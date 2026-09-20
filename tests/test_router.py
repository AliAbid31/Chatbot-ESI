import pytest

from cissou.providers import AuthError, EchoProvider, LLMProvider, QuotaError, TransientError
from cissou.router import KeyState, LLMRouter, ProviderPool


class ScriptedProvider(LLMProvider):
    """Replays a scripted outcome per key so failover paths are deterministic."""

    def __init__(self, name, script):
        super().__init__(model="scripted")
        self.name = name
        self.script = script          # {api_key: Exception | str}
        self.calls = []

    def generate(self, api_key, system, history, message):
        self.calls.append(api_key)
        outcome = self.script[api_key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def pool(name, script):
    provider = ScriptedProvider(name, script)
    return provider, ProviderPool(
        provider=provider,
        keys=[KeyState(key=k, label=f"{name}#{i}") for i, k in enumerate(script, 1)],
    )


def test_rotates_past_a_quota_limited_key():
    provider, p = pool("gemini", {"k1": QuotaError("429"), "k2": "answer"})
    reply, name = LLMRouter([p]).generate("sys", [], "hi")
    assert (reply, name) == ("answer", "gemini")
    assert provider.calls == ["k1", "k2"]


def test_revoked_key_is_disabled_and_never_retried():
    provider, p = pool("gemini", {"k1": AuthError("suspended"), "k2": "ok"})
    router = LLMRouter([p])
    router.generate("sys", [], "hi")
    router.generate("sys", [], "hi again")
    assert provider.calls.count("k1") == 1, "dead key must not be retried"
    assert p.keys[0].disabled is True


def test_quota_key_only_cools_down():
    _, p = pool("gemini", {"k1": QuotaError("429"), "k2": "ok"})
    LLMRouter([p]).generate("sys", [], "hi")
    assert p.keys[0].disabled is False
    assert p.keys[0].cooldown_until > 0


def test_failover_to_the_second_provider():
    _, gemini = pool("gemini", {"g1": AuthError("dead")})
    groq_provider, groq = pool("groq", {"q1": "from groq"})
    reply, name = LLMRouter([gemini, groq]).generate("sys", [], "hi")
    assert (reply, name) == ("from groq", "groq")
    assert groq_provider.calls == ["q1"]


def test_transient_failure_moves_to_the_second_provider():
    _, gemini = pool("gemini", {"g1": TransientError("temporary outage")})
    groq_provider, groq = pool("groq", {"q1": "from groq"})
    reply, name = LLMRouter([gemini, groq]).generate("sys", [], "hi")
    assert (reply, name) == ("from groq", "groq")
    assert groq_provider.calls == ["q1"]


def test_transient_error_retries_the_same_key():
    class Flaky(LLMProvider):
        name = "flaky"

        def __init__(self):
            super().__init__(model="m")
            self.n = 0

        def generate(self, api_key, system, history, message):
            self.n += 1
            if self.n == 1:
                raise TransientError("hiccup")
            return "recovered"

    provider = Flaky()
    p = ProviderPool(provider=provider, keys=[KeyState(key="k1", label="flaky#1")])
    reply, _ = LLMRouter([p]).generate("sys", [], "hi")
    assert reply == "recovered"
    assert provider.n == 2


def test_degrades_to_offline_instead_of_crashing():
    _, p = pool("gemini", {"k1": AuthError("dead")})
    reply, name = LLMRouter([p]).generate("sys", [], "hi")
    assert name == "offline"
    assert "matching passage" in reply


def test_no_keys_configured_still_answers():
    reply, name = LLMRouter([]).generate("sys", [], "hi")
    assert name == "offline"


def test_round_robin_spreads_load_across_keys():
    provider, p = pool("gemini", {"k1": "a", "k2": "b", "k3": "c"})
    router = LLMRouter([p])
    for _ in range(3):
        router.generate("sys", [], "hi")
    assert provider.calls == ["k1", "k2", "k3"]


def test_status_never_leaks_key_material():
    _, p = pool("gemini", {"super-secret-key": "ok"})
    blob = str(LLMRouter([p]).status())
    assert "super-secret-key" not in blob


@pytest.mark.parametrize("exc,disabled", [(AuthError("x"), True), (QuotaError("x"), False)])
def test_error_taxonomy_drives_key_health(exc, disabled):
    _, p = pool("gemini", {"k1": exc, "k2": "ok"})
    LLMRouter([p]).generate("sys", [], "hi")
    assert p.keys[0].disabled is disabled


def test_quota_cooldown_honours_the_provider_hint():
    """A per-minute rate limit must not sideline a key for the max backoff."""
    import time as _time

    _, p = pool("groq", {"k1": QuotaError("429", retry_after=5), "k2": "ok"})
    LLMRouter([p]).generate("sys", [], "hi")
    remaining = p.keys[0].cooldown_until - _time.monotonic()
    assert 0 < remaining <= 5.5, f"expected ~5s cooldown, got {remaining:.1f}s"


def test_quota_cooldown_escalates_without_a_hint():
    from cissou.router import QUOTA_BACKOFF_SECONDS, KeyState, ProviderPool

    provider = ScriptedProvider("groq", {"k1": QuotaError("429")})
    p = ProviderPool(provider=provider, keys=[KeyState(key="k1", label="groq#1")])
    router = LLMRouter([p])

    import time as _time
    delays = []
    for _ in range(3):
        p.keys[0].cooldown_until = 0.0        # simulate the cooldown elapsing
        router.generate("sys", [], "hi")
        delays.append(p.keys[0].cooldown_until - _time.monotonic())

    assert delays[0] < delays[1] < delays[2], f"backoff must escalate, got {delays}"
    assert delays[0] <= QUOTA_BACKOFF_SECONDS[0] + 0.5


def test_success_clears_quota_strikes():
    _, p = pool("groq", {"k1": QuotaError("429"), "k2": "ok"})
    router = LLMRouter([p])
    router.generate("sys", [], "hi")
    assert p.keys[0].quota_strikes == 1
    p.keys[0].cooldown_until = 0.0
    p.provider.script["k1"] = "recovered"
    router.generate("sys", [], "hi")
    assert p.keys[0].quota_strikes == 0


def test_request_stops_after_the_attempt_budget():
    """A restart must not make one student wait through the whole dead pool."""
    script = {f"k{i}": AuthError("dead") for i in range(1, 9)}
    provider, p = pool("gemini", script)
    reply, name = LLMRouter([p], max_attempts=3).generate("sys", [], "hi")
    assert len(provider.calls) == 3, f"tried {len(provider.calls)} keys, budget was 3"
    assert name == "offline"
    assert "matching passage" in reply


def test_budget_spans_providers_not_per_provider():
    _, gemini = pool("gemini", {"g1": AuthError("x"), "g2": AuthError("x")})
    groq_provider, groq = pool("groq", {"q1": "late answer"})
    _, name = LLMRouter([gemini, groq], max_attempts=2).generate("sys", [], "hi")
    assert name == "offline"
    assert groq_provider.calls == [], "budget was spent before reaching groq"


def test_stream_does_not_leak_partial_answer_after_provider_failure():
    class PartialProvider(LLMProvider):
        name = "partial"

        def generate(self, api_key, system, history, message):
            return "fallback"

        def stream(self, api_key, system, history, message):
            yield "partial"
            raise TransientError("connection dropped")

    provider = PartialProvider(model="test")
    router = LLMRouter(
        [ProviderPool(provider=provider, keys=[KeyState(key="k", label="partial#1")])],
        fallback=EchoProvider(model="offline"),
    )

    output = "".join(fragment for fragment, _ in router.stream("<knowledge>fact</knowledge>", [], "question"))

    assert "partial" not in output
    assert "matching passage" in output


def test_gemini_service_unavailable_is_transient():
    from cissou.providers.gemini import classify

    assert isinstance(classify(RuntimeError("503 Service Unavailable")), TransientError)


def test_groq_network_access_denial_is_transient():
    from cissou.providers.groq import GroqProvider

    class Response:
        status_code = 403
        text = '{"error":{"message":"Access denied. Please check your network settings."}}'

    with pytest.raises(TransientError):
        GroqProvider._check(Response())


def test_groq_stream_decodes_utf8_bytes(monkeypatch):
    from cissou.providers.groq import GroqProvider

    class Response:
        status_code = 200
        text = ""

        def iter_lines(self, decode_unicode):
            assert decode_unicode is False
            yield b'data: {"choices":[{"delta":{"content":"sp\xc3\xa9cialit\xc3\xa9s"}}]}'
            yield b"data: [DONE]"

    monkeypatch.setattr("cissou.providers.groq.requests.post", lambda *args, **kwargs: Response())
    provider = GroqProvider(model="test")

    assert "spécialités" == "".join(provider.stream("key", "system", [], "question"))


def test_budget_does_not_block_a_healthy_first_key():
    provider, p = pool("gemini", {"k1": "fast answer"})
    reply, name = LLMRouter([p], max_attempts=1).generate("sys", [], "hi")
    assert (reply, name) == ("fast answer", "gemini")


def test_dead_keys_are_discovered_across_successive_requests():
    """Budgeted attempts still converge: each request retires a few more keys."""
    script = {f"k{i}": AuthError("dead") for i in range(1, 7)}
    script["k6"] = "alive"
    provider, p = pool("gemini", script)
    router = LLMRouter([p], max_attempts=2)
    names = [router.generate("sys", [], "hi")[1] for _ in range(4)]
    assert names[-1] == "gemini", f"never reached the healthy key: {names}"


def test_deadline_stops_further_attempts():
    _, p = pool("gemini", {"k1": AuthError("x"), "k2": "ok"})
    _, name = LLMRouter([p], deadline=0.0).generate("sys", [], "hi")
    assert name == "offline"


def test_stream_respects_the_budget():
    script = {f"k{i}": AuthError("dead") for i in range(1, 9)}
    provider, p = pool("gemini", script)
    out = list(LLMRouter([p], max_attempts=2).stream("sys", [], "hi"))
    assert len(provider.calls) <= 2
    assert out[-1][1] == "offline"


def test_each_provider_gets_its_own_output_budget():
    """Groq's free tier caps output tokens per minute below the Gemini budget."""
    from cissou.config import Settings
    from cissou.router import build_router

    settings = Settings(
        gemini_keys=["g1"], groq_keys=["q1"],
        max_output_tokens=1400, groq_max_output_tokens=900,
    )
    by_name = {p.provider.name: p.provider for p in build_router(settings).pools}
    assert by_name["gemini"].max_output_tokens == 1400
    assert by_name["groq"].max_output_tokens == 900


def test_groq_payload_does_not_inflate_max_tokens():
    """Requesting more than the account's OTPM limit gets the call rejected."""
    from cissou.providers import GroqProvider

    provider = GroqProvider(model="m", max_output_tokens=900)
    payload = provider._payload("sys", [], "hi", stream=False)
    assert payload["max_tokens"] == 900


def test_groq_payload_orders_system_history_then_message():
    from cissou.providers import GroqProvider

    provider = GroqProvider(model="m", max_output_tokens=900)
    payload = provider._payload("SYS", [{"role": "user", "content": "old"},
                                        {"role": "assistant", "content": "prev"}],
                                "new", stream=False)
    assert [m["role"] for m in payload["messages"]] == ["system", "user", "assistant", "user"]
    assert payload["messages"][0]["content"] == "SYS"
    assert payload["messages"][-1]["content"] == "new"
