import math
import re
import unicodedata
from collections import Counter
from typing import List, Tuple
from cissou.knowledge import Chunk

# Dictionnaire de correspondance bilingue et argotique ESI
BILINGUAL_DICTIONARY = {
    # Hébergement & Vie
    "chambre": ["room", "accommodation", "double", "individual", "hebergement", "residence"],
    "chambres": ["rooms", "accommodation", "double", "individual"],
    "dormir": ["residence", "accommodation", "bouraoui", "alia"],
    "filles": ["female", "alia", "girl", "women"],
    "garcons": ["male", "bouraoui", "boy", "men"],
    "douche": ["shower", "showers", "sanitary"],
    "eau": ["water", "interrupted", "schedule", "coupure"],
    "resto": ["restaurant", "meal", "breakfast", "lunch", "dinner"],
    "manger": ["restaurant", "meal", "food", "cafeteria"],
    
    # Transport
    "bus": ["kous", "transport", "shuttle", "navette"],
    "kous": ["bus", "transport", "navette", "liaison"],
    "taxi": ["transport", "rond-point", "oued", "smar"],
    
    # Scolarité & Concours
    "note": ["grade", "average", "marks", "eliminatoire", "elimination"],
    "notes": ["grades", "marks", "evaluation"],
    "eliminatoire": ["elimination", "threshold", "60%"],
    "rachat": ["compensation", "regulations"],
    "concours": ["competitive", "exam", "admission", "titre", "epreuve"],
    "stage": ["internship", "company", "mandatory", "discovery", "pfe"],
    "coef": ["coefficient", "weight"],
    "credits": ["credit", "ects", "validation"],
    
    # Clubs
    "club": ["clubs", "cse", "hackit", "datahack", "tresor", "shellmates", "etic"],
    "ia": ["ai", "artificial", "intelligence", "sid", "soai"],
    "securite": ["security", "cybersecurity", "shellmatess", "siq"],
}

def clean_text(text: str) -> str:
    """Supprime les accents et met en minuscules."""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.lower()

class ESIBM25Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.doc_len = []
        self.avg_doc_len = 0.0
        self.doc_freqs = Counter()
        self.tokenized_corpus = []

        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        text = clean_text(text)
        tokens = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text)
        expanded = list(tokens)
        
        # Expansion bilingue automatique
        for t in tokens:
            if t in BILINGUAL_DICTIONARY:
                expanded.extend(BILINGUAL_DICTIONARY[t])
        return expanded

    def _build_index(self):
        total_tokens = 0
        for chunk in self.chunks:
            # On donne une surpondération au titre et nom de fichier
            meta = f"{chunk.source_file} {chunk.breadcrumb} {chunk.breadcrumb}"
            full_text = f"{meta} {chunk.content}"
            
            tokens = self._tokenize(full_text)
            self.tokenized_corpus.append(tokens)
            self.doc_len.append(len(tokens))
            total_tokens += len(tokens)
            
            for token in set(tokens):
                self.doc_freqs[token] += 1

        self.avg_doc_len = total_tokens / max(1, len(self.chunks))

    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        query_tokens = self._tokenize(query)
        scores = []
        
        N = len(self.chunks)
        k1 = 1.6
        b = 0.75

        # Détection d'un code de module précis (ex: POO, ALSDS, RES1)
        target_code = None
        for word in re.findall(r"[A-Z0-9]{3,6}", query.upper()):
            if re.match(r'^[A-Z0-9]{3,6}$', word):
                target_code = word

        for idx, doc_tokens in enumerate(self.tokenized_corpus):
            score = 0.0
            doc_len = self.doc_len[idx]
            token_counts = Counter(doc_tokens)

            for q in query_tokens:
                if q in token_counts:
                    freq = token_counts[q]
                    df = self.doc_freqs.get(q, 0)
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                    num = freq * (k1 + 1.0)
                    den = freq + k1 * (1.0 - b + b * (doc_len / self.avg_doc_len))
                    score += idf * (num / den)

            # Bonus massif si le module demandé correspond au titre exact du chunk
            if target_code:
                heading_has_code = target_code in self.chunks[idx].breadcrumb.upper()
                exact_code = re.search(rf"(?<![A-Z0-9]){re.escape(target_code)}(?![A-Z0-9])",
                                       self.chunks[idx].content.upper())
                if heading_has_code:
                    score += 20.0
                if exact_code:
                    score += 12.0

            scores.append((self.chunks[idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, s in scores[:top_k] if s > 0]