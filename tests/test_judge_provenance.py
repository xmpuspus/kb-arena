"""A judged score names its judge, its prompt, and its verdict, and the judge can run elsewhere."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from kb_arena.benchmark import evaluator as ev
from kb_arena.benchmark.evaluator import JUDGE_SYSTEM_PROMPT, evaluate
from kb_arena.llm import client as llm_client
from kb_arena.llm.client import LLMClient, LLMResponse
from kb_arena.models.benchmark import Constraints, GroundTruth
from kb_arena.settings import settings


class _FakeJudge:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    async def judge(self, **kwargs):
        self.calls += 1
        return LLMResponse(text=self.text, provider="openai", model="gpt-4o", cost_usd=0.001)


@pytest.mark.asyncio
async def test_a_judged_score_carries_its_provenance():
    verdict = {
        "accuracy": 0.8,
        "completeness": 0.5,
        "faithfulness": 1.0,
        "rationale": "Names the error.",
    }
    llm = _FakeJudge(json.dumps(verdict))

    score = await evaluate(
        "json.loads raises JSONDecodeError.",
        GroundTruth(answer="json.loads raises JSONDecodeError."),
        Constraints(),
        sources=[],
        llm=llm,
        question_text="What does json.loads raise?",
    )

    assert score.accuracy == 0.8
    assert score.judge_provider == "openai"
    assert score.judge_model == "gpt-4o"
    assert score.judge_prompt_hash == hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode()).hexdigest()[:16]
    assert score.judge_rationale == "Names the error."
    assert json.loads(score.judge_raw) == verdict


@pytest.mark.asyncio
async def test_a_verdict_without_a_rationale_still_scores():
    llm = _FakeJudge(json.dumps({"accuracy": 1.0, "completeness": 1.0, "faithfulness": 1.0}))

    score = await evaluate("a", GroundTruth(answer="a"), Constraints(), sources=[], llm=llm)

    assert score.accuracy == 1.0
    assert score.judge_rationale == ""
    assert score.judge_model == "gpt-4o"


@pytest.mark.asyncio
async def test_no_judge_means_empty_provenance():
    score = await evaluate("a", GroundTruth(answer="a"), Constraints(), sources=[], llm=None)

    assert score.judge_model == ""
    assert score.judge_prompt_hash == ""


def test_the_prompt_hash_moves_with_the_prompt(monkeypatch):
    before = ev._hash_text(ev.JUDGE_SYSTEM_PROMPT)
    monkeypatch.setattr(ev, "JUDGE_SYSTEM_PROMPT", ev.JUDGE_SYSTEM_PROMPT + " Be lenient.")
    assert ev._hash_text(ev.JUDGE_SYSTEM_PROMPT) != before


class _FakeProvider:
    def __init__(self, name: str):
        self.name = name
        self.calls: list[str] = []

    async def complete(self, model, system, user, max_tokens=4096, temperature=0):
        self.calls.append(model)
        return SimpleNamespace(
            text="{}", input_tokens=1, output_tokens=1, cache_creation_tokens=0, cache_read_tokens=0
        )


def _fake_factory(made: dict, keys: dict | None = None):
    def create_provider(provider_name: str, **kwargs):
        made[provider_name] = _FakeProvider(provider_name)
        if keys is not None:
            keys[provider_name] = kwargs.get("api_key")
        return made[provider_name]

    return create_provider


def test_the_judge_provider_reads_its_own_key_not_the_generic_one(monkeypatch):
    made: dict[str, _FakeProvider] = {}
    keys: dict[str, str | None] = {}
    monkeypatch.setattr("kb_arena.llm.providers.create_provider", _fake_factory(made, keys))
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "judge_provider", "openai")
    monkeypatch.setattr(settings, "llm_api_key", "generic-anthropic-key")
    monkeypatch.setattr(settings, "openai_api_key", "openai-key")

    LLMClient()

    assert keys["anthropic"] == "generic-anthropic-key"
    assert keys["openai"] == "openai-key"


def test_a_run_snapshot_names_the_judge(monkeypatch):
    from kb_arena.benchmark.runner import _config_snapshot

    llm = SimpleNamespace(judge_identity={"provider": "openai", "model": "gpt-judge"})
    snap = _config_snapshot(
        llm, top_k=5, split="", reference_free=False, cost_cap=0.0, parallel=True
    )

    assert snap["judge_provider"] == "openai"
    assert snap["judge_model"] == "gpt-judge"
    assert snap["llm_provider"] == settings.llm_provider


@pytest.mark.asyncio
async def test_the_judge_runs_on_its_own_provider(monkeypatch):
    made: dict[str, _FakeProvider] = {}
    monkeypatch.setattr("kb_arena.llm.providers.create_provider", _fake_factory(made))
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "judge_provider", "openai")
    monkeypatch.setattr(settings, "openai_judge_model", "gpt-judge")
    monkeypatch.setattr(settings, "generate_model", "claude-gen")

    client = LLMClient(api_key="k")
    judged = await client._call_once("judge", "sys", "user")
    generated = await client._call_once("generate", "sys", "user")

    assert made["openai"].calls == ["gpt-judge"]
    assert made["anthropic"].calls == ["claude-gen"]
    assert (judged.provider, judged.model) == ("openai", "gpt-judge")
    assert (generated.provider, generated.model) == ("anthropic", "claude-gen")


def test_an_empty_judge_provider_follows_the_generation_provider(monkeypatch):
    made: dict[str, _FakeProvider] = {}
    monkeypatch.setattr("kb_arena.llm.providers.create_provider", _fake_factory(made))
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "judge_provider", "")

    client = LLMClient(api_key="k")

    assert client._judge_provider is client._provider
    assert list(made) == ["anthropic"]
    assert llm_client._provider_setup("anthropic", "k")[1]["judge"] == settings.judge_model
