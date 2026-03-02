"""Comprehensive chat endpoint tests — dozens of scenarios."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Basic factoid questions
# ---------------------------------------------------------------------------


def test_factoid_returns_answer(client):
    r = client.post("/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"})
    assert r.status_code == 200
    assert r.json()["answer"]


def test_factoid_has_sources(client):
    r = client.post("/chat", json={"query": "What does json.loads do?", "strategy": "naive_vector"})
    assert isinstance(r.json()["sources"], list)


def test_factoid_has_latency_ms(client):
    r = client.post("/chat", json={"query": "What is json?", "strategy": "naive_vector"})
    assert r.json()["latency_ms"] >= 0


def test_factoid_has_tokens_used(client):
    r = client.post("/chat", json={"query": "What is json?", "strategy": "naive_vector"})
    assert isinstance(r.json()["tokens_used"], int)


def test_factoid_has_cost_usd(client):
    r = client.post("/chat", json={"query": "What is json?", "strategy": "naive_vector"})
    assert isinstance(r.json()["cost_usd"], (int, float))


@pytest.mark.parametrize(
    "question",
    [
        "What does json.loads do?",
        "What is the return type of os.path.join?",
        "What module is pathlib in?",
        "What is a Python decorator?",
        "What does the 'with' statement do?",
    ],
)
def test_factoid_questions_succeed(client, question):
    r = client.post("/chat", json={"query": question, "strategy": "naive_vector"})
    assert r.status_code == 200
    assert r.json()["answer"]


# ---------------------------------------------------------------------------
# Strategy routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy,expected_used",
    [
        ("naive_vector", "naive_vector"),
        ("contextual_vector", "contextual_vector"),
        ("qna_pairs", "qna_pairs"),
        ("knowledge_graph", "knowledge_graph"),
        ("hybrid", "hybrid"),
    ],
)
def test_strategy_routing(client, strategy, expected_used):
    r = client.post("/chat", json={"query": "What is X?", "strategy": strategy})
    assert r.status_code == 200
    assert r.json()["strategy_used"] == expected_used


def test_default_strategy_is_hybrid(client):
    r = client.post("/chat", json={"query": "What is X?"})
    assert r.status_code == 200
    assert r.json()["strategy_used"] == "hybrid"


def test_knowledge_graph_returns_graph_context(client):
    r = client.post(
        "/chat", json={"query": "What depends on json.loads?", "strategy": "knowledge_graph"}
    )
    assert r.status_code == 200
    data = r.json()
    # graph_context may be None (mock fallback) or populated
    assert "graph_context" in data


def test_comparison_question_routes_to_hybrid(client):
    r = client.post(
        "/chat",
        json={
            "query": "compare json.loads vs json.load",
            "strategy": "hybrid",
        },
    )
    assert r.status_code == 200
    assert r.json()["strategy_used"] == "hybrid"


# ---------------------------------------------------------------------------
# Multi-turn conversation — history handling
# ---------------------------------------------------------------------------


def test_history_empty_array_works(client):
    r = client.post(
        "/chat",
        json={
            "query": "What is json?",
            "strategy": "naive_vector",
            "history": [],
        },
    )
    assert r.status_code == 200


def test_history_two_turns(client):
    r = client.post(
        "/chat",
        json={
            "query": "What exceptions does it raise?",
            "strategy": "naive_vector",
            "history": [
                {"role": "user", "content": "What does json.loads do?"},
                {"role": "assistant", "content": "It deserializes JSON strings."},
            ],
        },
    )
    assert r.status_code == 200


def test_history_three_turns(client):
    r = client.post(
        "/chat",
        json={
            "query": "Can I use it with bytes?",
            "strategy": "naive_vector",
            "history": [
                {"role": "user", "content": "What does json.loads do?"},
                {"role": "assistant", "content": "It parses JSON strings."},
                {"role": "user", "content": "What does it return?"},
                {"role": "assistant", "content": "A Python object (dict, list, etc.)."},
            ],
        },
    )
    assert r.status_code == 200


def test_strategy_switch_mid_conversation(client):
    """Second turn uses a different strategy than first."""
    r1 = client.post(
        "/chat",
        json={
            "query": "What is json?",
            "strategy": "naive_vector",
        },
    )
    assert r1.status_code == 200

    r2 = client.post(
        "/chat",
        json={
            "query": "How does it compare to yaml?",
            "strategy": "hybrid",
            "history": [
                {"role": "user", "content": "What is json?"},
                {"role": "assistant", "content": r1.json()["answer"]},
            ],
        },
    )
    assert r2.status_code == 200
    assert r2.json()["strategy_used"] == "hybrid"


def test_history_twenty_turns_accepted(client):
    """Very long history is accepted without error."""
    history = []
    for i in range(20):
        history.append({"role": "user", "content": f"Question {i}?"})
        history.append({"role": "assistant", "content": f"Answer {i}."})

    r = client.post(
        "/chat",
        json={
            "query": "Final question?",
            "strategy": "naive_vector",
            "history": history,
        },
    )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Unicode and special characters
# ---------------------------------------------------------------------------


def test_unicode_chinese(client):
    r = client.post("/chat", json={"query": "什么是 JSON？", "strategy": "naive_vector"})
    assert r.status_code == 200


def test_unicode_arabic(client):
    r = client.post("/chat", json={"query": "ما هو json.loads؟", "strategy": "naive_vector"})
    assert r.status_code == 200


def test_unicode_emoji(client):
    r = client.post(
        "/chat", json={"query": "What does json.loads do? 🐍", "strategy": "naive_vector"}
    )
    assert r.status_code == 200


def test_special_chars_ampersand(client):
    r = client.post(
        "/chat", json={"query": "json & yaml & toml differences?", "strategy": "naive_vector"}
    )
    assert r.status_code == 200


def test_special_chars_html_entities(client):
    r = client.post(
        "/chat",
        json={"query": "<script>alert('xss')</script> What is json?", "strategy": "naive_vector"},
    )
    assert r.status_code == 200


def test_special_chars_sql_injection_attempt(client):
    r = client.post(
        "/chat", json={"query": "'; DROP TABLE documents; --", "strategy": "naive_vector"}
    )
    assert r.status_code == 200


def test_special_chars_quotes(client):
    r = client.post(
        "/chat",
        json={"query": "What is \"json.loads\" and 'json.load'?", "strategy": "naive_vector"},
    )
    assert r.status_code == 200


def test_very_short_query_hi(client):
    r = client.post("/chat", json={"query": "hi", "strategy": "naive_vector"})
    assert r.status_code == 200


def test_very_short_query_single_char(client):
    r = client.post("/chat", json={"query": "?", "strategy": "naive_vector"})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Content not in corpus
# ---------------------------------------------------------------------------


def test_out_of_corpus_question(client):
    """Questions about unrelated topics should still get a response (strategy handles it)."""
    r = client.post(
        "/chat",
        json={
            "query": "What is the capital of France?",
            "strategy": "naive_vector",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["answer"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_missing_query_field_returns_422(client):
    r = client.post("/chat", json={"strategy": "naive_vector"})
    assert r.status_code == 422


def test_null_query_returns_422(client):
    r = client.post("/chat", json={"query": None, "strategy": "naive_vector"})
    assert r.status_code == 422


def test_extra_fields_are_ignored(client):
    r = client.post(
        "/chat",
        json={
            "query": "What is X?",
            "strategy": "naive_vector",
            "extra_field": "should be ignored",
            "another": 42,
        },
    )
    assert r.status_code == 200


def test_invalid_strategy_returns_400_or_422(client):
    r = client.post("/chat", json={"query": "X?", "strategy": "nonexistent_strategy"})
    assert r.status_code in (400, 422)


def test_wrong_history_role_still_accepted(client):
    """Pydantic accepts any string for role — API doesn't validate enum."""
    r = client.post(
        "/chat",
        json={
            "query": "X?",
            "strategy": "naive_vector",
            "history": [{"role": "system", "content": "Be helpful."}],
        },
    )
    assert r.status_code == 200


def test_empty_history_message_content(client):
    r = client.post(
        "/chat",
        json={
            "query": "X?",
            "strategy": "naive_vector",
            "history": [{"role": "user", "content": ""}],
        },
    )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Rapid sequential requests
# ---------------------------------------------------------------------------


def test_ten_sequential_requests(client):
    for i in range(10):
        r = client.post(
            "/chat",
            json={
                "query": f"Question {i}?",
                "strategy": "naive_vector",
            },
        )
        assert r.status_code in (200, 429)


def test_requests_with_different_strategies_sequentially(client):
    strategies = ["naive_vector", "contextual_vector", "qna_pairs", "knowledge_graph", "hybrid"]
    for s in strategies:
        r = client.post("/chat", json={"query": "What is json?", "strategy": s})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Corpus field
# ---------------------------------------------------------------------------


def test_corpus_field_accepted(client):
    r = client.post(
        "/chat",
        json={
            "query": "What is json?",
            "strategy": "naive_vector",
            "corpus": "python-stdlib",
        },
    )
    assert r.status_code == 200


def test_corpus_default_works(client):
    r = client.post("/chat", json={"query": "What is json?", "strategy": "naive_vector"})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Large context
# ---------------------------------------------------------------------------


def test_large_context_in_history(client):
    long_content = "This is a very long message. " * 200  # ~5000 chars
    r = client.post(
        "/chat",
        json={
            "query": "Summarize the above.",
            "strategy": "naive_vector",
            "history": [
                {"role": "user", "content": long_content},
                {"role": "assistant", "content": "Here is a summary."},
            ],
        },
    )
    assert r.status_code == 200
