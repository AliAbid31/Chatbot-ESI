import requests
import re
from google import genai
from cissou.config import settings

class LLMRouter:
    def __init__(self):
        self.gemini_client = None
        if settings.GEMINI_API_KEY:
            self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def generate(self, prompt: str) -> str:
        # 1. Tentative avec Google Gemini
        if self.gemini_client:
            try:
                response = self.gemini_client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=prompt
                )
                if response.text:
                    return response.text
            except Exception as e:
                print(f"[Alerte Gemini] : {type(e).__name__}: {e} -> on Groq")

        # 2. Secours avec Groq (Llama 3)
        if settings.GROQ_API_KEY:
            try:
                headers = {
                    "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": settings.GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2
                }
                res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=15)
                if res.status_code == 200:
                    return res.json()["choices"][0]["message"]["content"]
                print(f"[Alerte Groq] HTTP {res.status_code}: {res.text[:300]}")
            except Exception as e:
                print(f"[Alerte Groq] : {type(e).__name__}: {e} -> on offline mode")

        # 3. Fallback Hors-Ligne
        match = re.search(r"<DOCUMENTS>\s*(.*?)\s*</DOCUMENTS>", prompt, re.S)
        context = match.group(1).strip() if match else ""
        return (
            "⚠️ Mode hors ligne : le fournisseur LLM est temporairement indisponible.\n\n"
            "Voici les passages les plus pertinents de la base ESI :\n\n"
            f"{context or 'Aucun passage pertinent trouvé.'}"
        )