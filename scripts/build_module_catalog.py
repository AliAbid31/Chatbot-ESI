"""Build the row-oriented module catalog from modules_explained.md."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "documents" / "academic" / "modules_explained.md"
TARGET = ROOT / "data" / "documents" / "academic" / "modules_catalog.md"

TRACK_RE = re.compile(
    r"^# .*?\s+[—?]\s+(?P<level>1CP|2CP|1CS)|^# .*?2CS Speciality\s+[—?]\s+(?P<speciality>SID|SIT|SIL|SIQ)",
    re.MULTILINE,
)
MODULE_RE = re.compile(
    r"^### ([A-Z][A-Z0-9_]*) — (.+?) \((First|Second) Semester\)$",
    re.MULTILINE,
)
SECTION_RE = re.compile(r"^(?:##+\s+|---\s*$|#\s+)", re.MULTILINE)
COEFFICIENT_RE = re.compile(r"^\*\*Coefficient:\*\*\s*([^\s]+)", re.MULTILINE)


def parse_rows(text: str) -> list[tuple[str, str, str, str, str, str, str]]:
    tracks = list(TRACK_RE.finditer(text))
    rows = []
    for index, track in enumerate(tracks):
        end = tracks[index + 1].start() if index + 1 < len(tracks) else len(text)
        section = text[track.end():end]
        level = track.group("level") or "2CS"
        speciality = track.group("speciality") or "General"
        modules = list(MODULE_RE.finditer(section))
        for module_index, module in enumerate(modules):
            module_end = modules[module_index + 1].start() if module_index + 1 < len(modules) else len(section)
            section_end = SECTION_RE.search(section[module.end():module_end])
            if section_end:
                module_end = module.end() + section_end.start()
            block = section[module.end():module_end]
            coefficient = COEFFICIENT_RE.search(block)
            description = ""
            if coefficient:
                description = " ".join(
                    line.strip() for line in block[coefficient.end():].splitlines() if line.strip()
                )
            description = re.sub(r"\s+", " ", description).replace("|", "\\|")
            rows.append((
                level,
                speciality,
                "S1" if module.group(3) == "First" else "S2",
                module.group(1),
                module.group(2),
                coefficient.group(1) if coefficient else "N/A",
                description,
            ))
    return rows


def build_catalog(rows: list[tuple[str, str, str, str, str, str, str]]) -> str:
    lines = [
        "# ESI Module Catalog",
        "",
        "Canonical row-oriented catalog generated from `modules_explained.md`.",
        "Every row is one module. `S1` means first semester and `S2` means second semester.",
        "Descriptions and coefficients are copied from the maintained module guide; missing values remain `N/A`.",
        "",
    ]
    groups = [("1CP", "General"), ("2CP", "General"), ("1CS", "General"),
              ("2CS", "SID"), ("2CS", "SIT"), ("2CS", "SIL"), ("2CS", "SIQ")]
    for level, speciality in groups:
        lines.extend([
            f"## {level} - {speciality}",
            "",
            "| Niveau | Spécialité | Semestre | Code | Nom | Coefficient | Description |",
            "|---|---|---|---|---|---:|---|",
        ])
        for row in rows:
            if row[0] == level and row[1] == speciality:
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    rows = parse_rows(SOURCE.read_text(encoding="utf-8"))
    if len(rows) < 80:
        raise RuntimeError(f"Only parsed {len(rows)} modules; refusing to write an incomplete catalog")
    TARGET.write_text(build_catalog(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} module rows to {TARGET}")