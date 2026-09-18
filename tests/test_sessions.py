import time

from cissou.sessions import SessionStore


def test_sessions_are_isolated_between_users():
    store = SessionStore()
    a, b = store.get(None), store.get(None)
    assert a.session_id != b.session_id
    store.record(a, "hello from A", "reply A")
    assert store.history(b) == [], "history must not leak between sessions"


def test_history_round_trips():
    store = SessionStore()
    s = store.get(None)
    store.record(s, "q1", "a1")
    assert [t["role"] for t in store.history(s)] == ["user", "assistant"]


def test_history_is_trimmed_to_max_turns():
    store = SessionStore(max_turns=2)
    s = store.get(None)
    for i in range(6):
        store.record(s, f"q{i}", f"a{i}")
    history = store.history(s)
    assert len(history) == 4              # 2 turns == 2 user + 2 assistant
    assert history[0]["content"] == "q4"  # oldest dropped


def test_known_session_is_reused():
    store = SessionStore()
    s = store.get(None)
    store.record(s, "q", "a")
    assert store.get(s.session_id).session_id == s.session_id
    assert len(store.history(store.get(s.session_id))) == 2


def test_forged_session_id_gets_a_fresh_session():
    store = SessionStore()
    forged = store.get("../../etc/passwd")
    assert forged.session_id != "../../etc/passwd"
    assert len(forged.session_id) == 32


def test_lru_eviction_bounds_memory():
    store = SessionStore(max_sessions=3)
    ids = [store.get(None).session_id for _ in range(5)]
    assert len(store) == 3
    assert store.get(ids[0]).session_id == ids[0]  # re-minted, not resurrected
    assert store.history(store.get(ids[0])) == []


def test_expired_sessions_are_purged():
    store = SessionStore(ttl_seconds=0)
    s = store.get(None)
    store.record(s, "q", "a")
    time.sleep(0.01)
    assert store.history(store.get(s.session_id)) == []


def test_reset_clears_history():
    store = SessionStore()
    s = store.get(None)
    store.record(s, "q", "a")
    assert store.reset(s.session_id) is True
    assert store.reset(s.session_id) is False
