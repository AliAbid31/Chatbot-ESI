from cissou.config import settings
from cissou.knowledge import KnowledgeBase
from cissou.retrieval import ESIBM25Retriever
from cissou.prompts import format_prompt
from cissou.sessions import SessionManager
from cissou.router import LLMRouter

class ChatService:
    def __init__(self):
        print("Loading ESI knowledge base...")
        self.kb = KnowledgeBase(settings.DATA_DIR)
        self.retriever = ESIBM25Retriever(self.kb.chunks)
        self.sessions = SessionManager()
        self.router = LLMRouter()
        print(f"Indexation finished : {len(self.kb.chunks)} chunks.")

    def answer_question(self, session_id: str, query: str) -> dict:
        session = self.sessions.get_or_create(session_id)
        
        # Détection de question de suivi (Ex: "et pour les filles ?")
        retrieval_query = query
        if len(query.split()) <= 4 and session.history:
            last_user_query = next((m["content"] for m in reversed(session.history) if m["role"] == "user"), "")
            retrieval_query = f"{last_user_query} {query}"

        # Récupération des extraits pertinents
        top_chunks = self.retriever.retrieve(retrieval_query, top_k=settings.TOP_K_CHUNKS)

        # Construction du prompt
        prompt = format_prompt(query, top_chunks, session.history)

        # Génération
        reply = self.router.generate(prompt)

        # Mise à jour de l'historique
        session.add_turn("user", query)
        session.add_turn("assistant", reply)

        sources = list(set([f"{c.source_file} ({c.breadcrumb})" for c in top_chunks]))

        return {
            "answer": reply,
            "sources": sources,
            "session_id": session_id
        }