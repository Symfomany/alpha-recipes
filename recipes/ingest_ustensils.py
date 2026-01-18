# recipes/ingest_ustensils.py

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from langchain_core.documents import Document

from .config import USTENSILS_VS, BASE_DIR


CSV_PATH = BASE_DIR / "files" / "ustensils.csv"


def ingest_ustensils() -> None:
    rprint(Panel.fit("[bold cyan]Ingestion du catalogue d'ustensiles → Chroma 'ustensils'[/bold cyan]"))

    if not CSV_PATH.exists():
        rprint(f"[red]CSV introuvable : {CSV_PATH}[/red]")
        return

    docs: List[Document] = []

    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    table = Table(title="Ustensiles détectés", show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("ID")
    table.add_column("Nom")
    table.add_column("Type")

    for idx, row in enumerate(rows, start=1):
        uid = row.get("id") or f"ust-{idx}"
        name = row.get("name") or "Ustensile"
        kind = row.get("kind") or ""
        tasks = row.get("tasks") or ""
        notes = row.get("notes") or ""
        url = row.get("url") or ""

        # texte indexé pour la similarité
        text = (
            f"{name} ({kind})\n\n"
            f"Tâches / usages : {tasks}\n\n"
            f"Notes : {notes}\n"
        )

        meta = {
            "id": uid,
            "name": name,
            "kind": kind,
            "tasks": tasks,
            "notes": notes,
            "url": url,
            "source": "ustensils",
        }

        docs.append(Document(page_content=text, metadata=meta))
        table.add_row(str(idx), uid, name, kind)

    rprint(table)

    rprint(
        Panel.fit(
            f"[cyan]Insertion dans Chroma 'ustensils'[/cyan] "
            f"({len(docs)} ustensiles)"
        )
    )
    USTENSILS_VS.add_documents(docs)

    rprint(Panel.fit("[bold green]Ingestion des ustensiles terminée ✅[/bold green]"))


if __name__ == "__main__":
    ingest_ustensils()
