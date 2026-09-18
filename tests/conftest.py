import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cissou.config import Settings           # noqa: E402
from cissou.knowledge import load_knowledge  # noqa: E402
from cissou.retrieval import KnowledgeBase   # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "esi_knowledge.md"


@pytest.fixture(scope="session")
def document() -> str:
    return load_knowledge(DATA)


@pytest.fixture(scope="session")
def kb(document) -> KnowledgeBase:
    return KnowledgeBase(document)


@pytest.fixture
def app(monkeypatch):
    """App wired to the offline provider: no network, no keys, full pipeline."""
    for var in list(Settings.__annotations__):
        monkeypatch.delenv(var.upper(), raising=False)
    for i in [""] + [f"_{n}" for n in range(2, 21)]:
        monkeypatch.delenv(f"GOOGLE_GEMINI_API_KEY{i}", raising=False)
        monkeypatch.delenv(f"GROQ_API_KEY{i}", raising=False)

    from cissou.app_factory import create_app
    # Explicit override, not reliance on the default: semantic retrieval
    # needs a network call to fetch the embedding model on first use, which
    # CI may not have. Keeping this suite BM25-only preserves "80 tests,
    # fully offline, no keys, no network" regardless of what the shipped
    # default is. (Settings() ignores env by design here, same as before
    # this fixture existed — the env deletions above are defense in depth,
    # not the actual source of these settings.)
    application = create_app(Settings(enable_semantic_retrieval=False))
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def client(app):
    return app.test_client()
