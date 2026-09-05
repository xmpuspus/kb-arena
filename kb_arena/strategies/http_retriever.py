"""Bring-your-own-retriever strategy — benchmarks a user's own HTTP endpoint.

A user points KB Arena at an HTTP endpoint that owns and serves its own
index. query() POSTs a RetrieverQueryRequest and validates the JSON reply
against RetrieverQueryResponse before turning it into a RetrievalTrace, so a
reply that does not match the schema is an error, never a partial result.

Reuses the SSRF guard from kb_arena.ingest.parsers.web (PR #31): the same
_validate_url check, and the same _PinnedClient/_PinnedBackend that dial the
IP the check resolved instead of a fresh DNS answer at connect time. This
module does not add a second SSRF check.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, ValidationError

from kb_arena.exceptions import RetrieverContractError
from kb_arena.ingest.parsers.web import (
    ResponseTooLargeError,
    SSRFBlocked,
    _PinnedClient,
    _read_capped,
    _validate_url,
)
from kb_arena.models.document import Document
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.strategies.base import AnswerResult, Strategy, validate_top_k

SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the question using ONLY the provided context.\n"
    "If the context doesn't contain enough information, say so. Be concise and accurate."
)


class RetrieverQueryRequest(BaseModel):
    """What KB Arena sends to a bring-your-own-retriever endpoint."""

    query: str
    top_k: int
    corpus: str = "all"


class RetrievedChunkPayload(BaseModel):
    """One chunk a bring-your-own-retriever endpoint must return.

    source_doc_id and source_section_id must survive extraction, storage, and
    retrieval end to end, so both are required here. An endpoint that omits
    either one fails schema validation with a message naming the field.
    """

    chunk_id: str
    source_doc_id: str = Field(min_length=1)
    source_section_id: str = Field(min_length=1)
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieverQueryResponse(BaseModel):
    """The full response body a bring-your-own-retriever endpoint must return."""

    chunks: list[RetrievedChunkPayload]


class _Deadline:
    """A response that stops yielding raw bytes once the wall clock runs out.

    `_read_capped` reads through `iter_raw`, `headers`, `is_stream_consumed`
    and `_content`, so this proxies those four and nothing else. It is a
    wrapper rather than a change to `_read_capped`, because the web fetcher
    that owns that function has its own deadline in its own transport.
    """

    def __init__(self, response, deadline: float) -> None:
        self._response = response
        self._deadline = deadline

    def __getattr__(self, name: str):
        return getattr(self._response, name)

    def iter_raw(self):
        for chunk in self._response.iter_raw():
            if time.perf_counter() > self._deadline:
                raise httpx.ReadTimeout("the reply did not finish inside the request timeout")
            yield chunk
        if time.perf_counter() > self._deadline:
            raise httpx.ReadTimeout("the reply did not finish inside the request timeout")


class HTTPRetrieverStrategy(Strategy):
    """Benchmarks a user's own retriever behind an HTTP endpoint.

    build_index() is a no-op: the endpoint owns and serves its own index.
    query() enforces a per-request timeout and a total time budget across the
    life of the instance, so a slow endpoint fails the run instead of hanging
    it. Strategy instances are shared across concurrent requests (see
    base.py), so the budget spend is guarded by a lock.
    """

    name = "http_retriever"

    def __init__(
        self,
        endpoint_url: str,
        *,
        name: str | None = None,
        request_timeout_s: float = 10.0,
        total_budget_s: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__()
        self.endpoint_url = endpoint_url
        if name:
            self.name = name
        self.request_timeout_s = request_timeout_s
        self.total_budget_s = total_budget_s
        self._client = client
        self._llm = None
        self._spend_lock = threading.Lock()
        self._spent_s = 0.0

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = _PinnedClient()
        return self._client

    def _get_llm(self):
        if self._llm is None:
            from kb_arena.llm.client import LLMClient

            self._llm = LLMClient()
        return self._llm

    async def build_index(self, documents: list[Document]) -> None:
        """No-op: the endpoint owns and serves its own index."""
        return None

    def _reserve_budget(self) -> float:
        """Claim the timeout for one request, or refuse before dialing.

        Returns min(request_timeout_s, remaining budget). Raises when the
        budget is already spent, so a persistently slow endpoint stops the
        run instead of hanging it call after call.
        """
        with self._spend_lock:
            remaining = self.total_budget_s - self._spent_s
            if remaining <= 0:
                raise RetrieverContractError(
                    f"{self.endpoint_url}: total time budget of {self.total_budget_s}s "
                    "is already spent"
                )
            allowance = min(self.request_timeout_s, remaining)
            # Reserve it now. Reading the remainder and charging it later let
            # two concurrent callers each see one whole second of a one-second
            # budget, and the pair then spent two. Strategy instances are
            # shared across requests, so this is the ordinary case.
            self._spent_s += allowance
            return allowance

    def _charge_budget(self, elapsed_s: float, allowance: float) -> None:
        """Settle a reservation against the time the call actually took."""
        with self._spend_lock:
            self._spent_s += elapsed_s - allowance

    def _fetch_chunks(
        self, question: str, top_k: int, corpus: str, request_timeout: float | None = None
    ) -> list[RetrievedChunkPayload]:
        """POST the query, validate the reply, and return its chunks.

        Runs on a worker thread: httpx.Client.post() blocks the calling
        thread, and query() awaits this via asyncio.to_thread so the event
        loop stays free.
        """
        # `query` reserves the allowance and bounds the whole call with it. A
        # direct caller passes none, so the reservation happens here instead.
        if request_timeout is None:
            request_timeout = self._reserve_budget()
        client = self._get_client()
        host = (urlparse(self.endpoint_url).hostname or "").lower()

        request_start = time.perf_counter()
        try:
            client.pins[host] = _validate_url(self.endpoint_url)
            payload = RetrieverQueryRequest(query=question, top_k=top_k, corpus=corpus)
            # `client.post` buffers and decompresses the whole body before any
            # check runs, so a gzip bomb behind a compliant-looking endpoint
            # exhausts memory for a `top_k` of 1. `send` with `stream=True`
            # hands back the response before the body, and `_read_capped`
            # counts raw and decoded bytes on the way in. Same cap, same
            # reader the web fetcher already uses.
            request = client.build_request(
                "POST",
                self.endpoint_url,
                json=payload.model_dump(),
                timeout=request_timeout,
                # The sibling web fetcher asks for an unencoded body, and this
                # one must too. httpx advertises `br` when Brotli is installed,
                # and `_read_capped` has no Brotli decoder, so a compliant
                # endpoint honouring that header crashed the call.
                headers={"Accept-Encoding": "identity"},
            )
            response = client.send(request, stream=True, follow_redirects=False)
            try:
                # httpx applies its timeout to each socket wait, not to the
                # call. An endpoint dripping one byte every 0.9 seconds never
                # waits a full second, so it ran past the timeout AND past the
                # budget and then succeeded. The deadline is wall-clock.
                deadline = request_start + request_timeout
                capped = _Deadline(response, deadline)
                response._content = _read_capped(capped, self.endpoint_url)
                # The wrapper aborts a live stream mid-body. A response whose
                # body was already read never reaches that loop, so the clock
                # is checked once more here and both paths answer the same way.
                if time.perf_counter() > deadline:
                    raise httpx.ReadTimeout("the reply did not finish inside the request timeout")
            finally:
                response.close()
        except SSRFBlocked:
            raise
        except ResponseTooLargeError as exc:
            raise RetrieverContractError(f"{self.endpoint_url}: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise RetrieverContractError(
                f"{self.endpoint_url}: no reply within {request_timeout:.1f}s"
            ) from exc
        finally:
            self._charge_budget(time.perf_counter() - request_start, request_timeout)

        if response.status_code >= 300:
            raise RetrieverContractError(
                f"{self.endpoint_url}: returned status {response.status_code}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise RetrieverContractError(
                f"{self.endpoint_url}: reply body is not valid JSON"
            ) from exc

        try:
            parsed = RetrieverQueryResponse.model_validate(body)
        except ValidationError as exc:
            raise RetrieverContractError(
                f"{self.endpoint_url}: reply does not match the retriever contract: {exc}"
            ) from exc

        return parsed.chunks[:top_k]

    def _to_retrieved_chunks(self, payloads: list[RetrievedChunkPayload]) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id=payload.chunk_id,
                doc_id=payload.source_doc_id,
                content=payload.content,
                score=payload.score,
                rank=i + 1,
                source_strategy=self.name,
                metadata={
                    **payload.metadata,
                    "source_doc_id": payload.source_doc_id,
                    "source_section_id": payload.source_section_id,
                },
            )
            for i, payload in enumerate(payloads)
        ]

    async def query(self, question: str, top_k: int = 5, corpus: str = "all") -> AnswerResult:
        """POST to the endpoint, validate the reply, then generate an answer."""
        validate_top_k(top_k)
        start = self._start_timer()

        retrieval_start = time.perf_counter()
        # The deadline lives HERE, not inside the HTTP call. Two review rounds
        # found it missing one layer down: first the body read, then the wait
        # for headers, and httpx restarts its own timeout on every socket read
        # in both. A caller-side bound covers connect, headers and body at
        # once, and no future layer inside httpx can slip under it.
        allowance = self._reserve_budget()
        try:
            payloads = await asyncio.wait_for(
                asyncio.to_thread(self._fetch_chunks, question, top_k, corpus, allowance),
                timeout=allowance,
            )
        except TimeoutError as exc:
            raise RetrieverContractError(
                f"{self.endpoint_url}: no reply within {allowance:.1f}s"
            ) from exc
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        retrieved = self._to_retrieved_chunks(payloads)
        trace = RetrievalTrace(
            query=question, retrieved=retrieved, latency_ms=retrieval_ms, top_k=top_k
        )

        sources = list(dict.fromkeys(chunk.doc_id for chunk in retrieved if chunk.doc_id))
        context = "\n\n---\n\n".join(chunk.content for chunk in retrieved)

        llm = self._get_llm()
        gen_start = time.perf_counter()
        resp = await llm.generate(query=question, context=context, system_prompt=SYSTEM_PROMPT)
        gen_ms = (time.perf_counter() - gen_start) * 1000

        latency_ms = self._record_metrics(
            start, tokens=resp.total_tokens, cost=resp.cost_usd, sources=sources
        )
        return AnswerResult(
            answer=resp.text,
            sources=sources,
            retrieval=trace,
            strategy=self.name,
            latency_ms=latency_ms,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=resp.total_tokens,
            cost_usd=resp.cost_usd,
        )
