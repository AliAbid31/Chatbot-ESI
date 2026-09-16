from dataclasses import dataclass
from pathlib import Path
import re
from typing import List

@dataclass
class Chunk:
    id: str
    source_file: str
    breadcrumb: str
    content: str

    def to_context(self) -> str:
        """Formate le passage pour l'injection dans le prompt du LLM."""
        return f"[[Source: {self.source_file} | Section: {self.breadcrumb}]]\n{self.content}\n"

class KnowledgeBase:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.chunks: List[Chunk] = []
        self._load_all_files()

    def _load_all_files(self):
        md_files = list(self.data_dir.glob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"Aucun fichier Markdown trouvé dans {self.data_dir}")

        for filepath in md_files:
            text = filepath.read_text(encoding="utf-8")
            if "Modules" in filepath.name:
                # Découpage spécifique par module pour préserver les coefficients
                self.chunks.extend(self._split_modules(filepath.name, text))
            else:
                # Découpage hiérarchique par titres (#, ##, ###)
                self.chunks.extend(self._split_hierarchical(filepath.name, text))

    def _split_modules(self, filename: str, text: str) -> List[Chunk]:
        """Isole chaque fiche module '### CODE — Nom' pour une précision absolue."""
        chunks = []
        # On découpe par niveau 3 (###) où se trouvent les fiches modules
        sections = re.split(r'\n(?=###\s+)', text)
        
        main_context = "ESI Modules Guide"
        current_year = "General"

        for idx, sec in enumerate(sections):
            sec = sec.strip()
            if not sec:
                continue
            
            # Détection de l'année (1CP, 2CP, 1CS, 2CS SID...)
            year_match = re.search(r'^#\s+(.+)$', sec, re.MULTILINE)
            if year_match:
                current_year = year_match.group(1).strip()

            chunks.append(Chunk(
                id=f"{filename}_mod_{idx}",
                source_file=filename,
                breadcrumb=f"{main_context} > {current_year}",
                content=sec
            ))
        return chunks

    def _split_hierarchical(self, filename: str, text: str) -> List[Chunk]:
        """Découpe les documents généraux en respectant la hiérarchie Markdown."""
        chunks = []
        lines = text.split("\n")
        
        h1, h2, h3 = filename.replace(".md", ""), "", ""
        buffer = []
        chunk_idx = 0

        def save_chunk():
            nonlocal chunk_idx
            content = "\n".join(buffer).strip()
            if len(content) > 50:  # Ignore les fragments vides
                path = " > ".join([p for p in [h1, h2, h3] if p])
                chunks.append(Chunk(
                    id=f"{filename}_{chunk_idx}",
                    source_file=filename,
                    breadcrumb=path,
                    content=content
                ))
                chunk_idx += 1
            buffer.clear()

        for line in lines:
            if line.startswith("# "):
                save_chunk()
                h1 = line[2:].strip()
                h2, h3 = "", ""
            elif line.startswith("## "):
                save_chunk()
                h2 = line[3:].strip()
                h3 = ""
            elif line.startswith("### "):
                save_chunk()
                h3 = line[4:].strip()
            else:
                buffer.append(line)

        save_chunk()
        return chunks