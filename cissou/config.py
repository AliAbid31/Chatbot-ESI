import os 
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Paramètres RAG
    TOP_K_CHUNKS: int = 5
    MAX_CONTEXT_CHARS: int = 7000
    
    # Clés API & Modèles
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    GEMINI_MODEL: str = "gemini-3.6-flash"
    GROQ_MODEL: str = "qwen/qwen3.8-27b"

settings = Settings()