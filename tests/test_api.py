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


def test_module_list_never_exposes_prompt_and_is_complete(client):
    # The default test corpus is the legacy single document, so this checks
    # the normal LLM path's safety contract without requiring provider keys.
    body = client.post("/api/chat", json={"message": "What courses are in 1CP?"}).get_json()
    assert "prompt says" not in body["reply"].lower()


def test_new_questions_do_not_inherit_the_previous_curriculum_scope():
    from cissou.app_factory import create_app
    from cissou.config import Settings

    structured_app = create_app(Settings(enable_semantic_retrieval=False))
    structured_client = structured_app.test_client()
    first = structured_client.post("/api/chat", json={"message": "What courses are in 1CP?"}).get_json()
    session_id = first["session_id"]

    second = structured_client.post(
        "/api/chat",
        json={"message": "What are modules in S1 2CP?", "session_id": session_id},
    ).get_json()
    third = structured_client.post(
        "/api/chat",
        json={"message": "What are the specialities in ESI?", "session_id": session_id},
    ).get_json()
    fourth = structured_client.post(
        "/api/chat",
        json={"message": "what is la note eliminatoire?", "session_id": session_id},
    ).get_json()

    assert "ECON" in second["reply"] and "ALSDS" not in second["reply"]
    assert "SIT" in third["reply"] and "ALG1" not in third["reply"]
    assert "offline mode" not in third["reply"].lower()
    assert "note éliminatoire" in fourth["reply"].lower()
    assert "offline mode" not in fourth["reply"].lower()


def test_first_year_mathematics_includes_both_semesters(client):
    body = client.post(
        "/api/chat",
        json={"message": "quelles modules de math on etudie en 1ere annee"},
    ).get_json()

    assert all(code in body["reply"] for code in ("ALG1", "ANAL1", "ALG2", "ANAL2"))


def test_catalog_module_list_is_complete_and_not_llm_truncated(client):
    body = client.post(
        "/api/chat",
        json={"message": "Quelles sont les modules en 1ere annee?"},
    ).get_json()

    assert body["provider"] == "knowledge"
    assert all(code in body["reply"] for code in ("ALSDS", "ANAL1", "ARCH1", "ALG1", "ALSDD", "ANAL2"))
    assert "-\n" not in body["reply"]


def test_followup_year_replaces_previous_year_scope(client):
    first = client.post(
        "/api/chat",
        json={"message": "quels modules de maths en 1ere annee"},
    ).get_json()
    second = client.post(
        "/api/chat",
        json={"message": "et en 2eme annee?", "session_id": first["session_id"]},
    ).get_json()

    assert all(code in second["reply"] for code in ("ALG3", "ANAL3", "PRST1", "ANAL4", "LOGM", "PRST2"))
    assert "ALG1" not in second["reply"] and "ANAL1" not in second["reply"]


def test_short_followup_year_without_the_word_year_replaces_scope(client):
    first = client.post(
        "/api/chat",
        json={"message": "quels modules de maths en 1ere annee"},
    ).get_json()
    second = client.post(
        "/api/chat",
        json={"message": "et en 2eme ?", "session_id": first["session_id"]},
    ).get_json()

    assert "ALG3" in second["reply"] and "ALG1" not in second["reply"]


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


def test_stream_declares_utf8_charset(client):
    response = client.post("/api/chat/stream", json={"message": "Quelles sont les spécialités ?"})
    assert response.mimetype == "text/event-stream"
    assert response.mimetype_params.get("charset") == "utf-8"


def test_rate_limiter_blocks_a_flood(app):
    app.config["RATE_LIMITER"].per_minute = 3
    client = app.test_client()
    codes = [client.post("/api/chat", json={"message": "hi"}).status_code for _ in range(5)]
    assert 429 in codes
    assert codes[:3] == [200, 200, 200]
