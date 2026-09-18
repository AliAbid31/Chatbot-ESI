import json


def test_health_reports_knowledge_stats(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok"
    assert body["assistant"] == "CISSOU"
    assert body["knowledge"]["chunks"] > 0


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"CISSOU" in response.data


def test_chat_returns_reply_and_session(client):
    body = client.post("/api/chat", json={"message": "Who developed CISSOU?"}).get_json()
    assert body["reply"]
    assert len(body["session_id"]) == 32
    assert body["provider"] == "offline"     # no keys configured in tests
    assert body["sources"]


def test_chat_grounds_the_offline_reply_in_retrieved_text(client):
    body = client.post("/api/chat", json={"message": "Who developed CISSOU?"}).get_json()
    assert "Badreddine" in body["reply"]


def test_session_id_is_reused_across_turns(client):
    first = client.post("/api/chat", json={"message": "What is 1CP?"}).get_json()
    second = client.post("/api/chat", json={"message": "And 2CP?",
                                            "session_id": first["session_id"]}).get_json()
    assert second["session_id"] == first["session_id"]


def test_empty_message_is_rejected(client):
    response = client.post("/api/chat", json={"message": "   "})
    assert response.status_code == 400
    assert response.get_json()["error"] == "empty_message"


def test_missing_body_is_rejected(client):
    assert client.post("/api/chat", data="not json",
                       content_type="application/json").status_code == 400


def test_oversized_message_is_rejected(client):
    response = client.post("/api/chat", json={"message": "x" * 5000})
    assert response.status_code == 413
    assert response.get_json()["error"] == "message_too_long"


def test_non_string_message_is_rejected(client):
    assert client.post("/api/chat", json={"message": 42}).status_code == 400


def test_search_endpoint_exposes_retrieval(client):
    body = client.get("/api/search?q=admission average").get_json()
    assert body["results"]
    assert "score" in body["results"][0]


def test_search_requires_a_query(client):
    assert client.get("/api/search?q=").status_code == 400


def test_session_reset(client):
    chat = client.post("/api/chat", json={"message": "hello"}).get_json()
    assert client.post("/api/session/reset",
                       json={"session_id": chat["session_id"]}).get_json()["cleared"] is True


def test_diagnostics_reports_router_state(client):
    body = client.get("/api/diagnostics").get_json()
    assert "router" in body and "knowledge" in body


def test_stream_emits_meta_delta_and_done(client):
    response = client.post("/api/chat/stream", json={"message": "What is CSE?"})
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.get_data(as_text=True).splitlines()
              if line.startswith("data: ")]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "meta" and kinds[-1] == "done"
    assert "delta" in kinds
    assert events[0]["session_id"]


def test_rate_limiter_blocks_a_flood(app):
    app.config["RATE_LIMITER"].per_minute = 3
    client = app.test_client()
    codes = [client.post("/api/chat", json={"message": "hi"}).status_code for _ in range(5)]
    assert 429 in codes
    assert codes[:3] == [200, 200, 200]
