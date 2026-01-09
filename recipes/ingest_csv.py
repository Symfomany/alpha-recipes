# recipes/ingest_csv.py

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from langchain_core.documents import Document

from config import RECIPES_VS, BASE_DIR


CSV_PATH = BASE_DIR / "data" / "recipes_salades.csv"


def ingest_salade_recipes() -> None:
    rprint(Panel.fit("[bold cyan]Ingestion des salades de saison → Chroma 'recipes'[/bold cyan]"))

    if not CSV_PATH.exists():
        rprint(f"[red]CSV introuvable : {CSV_PATH}[/red]")
        return

    docs: List[Document] = []

    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # tableau récap
    table = Table(title="Recettes détectées", show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("ID")
    table.add_column("Titre")
    table.add_column("Saison")
    table.add_column("Pers.")

    for idx, row in enumerate(rows, start=1):
        rid = row.get("id") or f"salade-{idx}"
        title = row.get("title") or "Salade"
        season = row.get("season") or "?"
        people = row.get("people") or "?"

        ingredients = (row.get("ingredients") or "").split(";")
        instructions = row.get("instructions") or ""

        # texte indexé pour le RAG
        text = (
            f"{title}\n\n"
            f"Saison : {season}\n"
            f"Portions : {people}\n\n"
            "Ingrédients :\n"
            + "\n".join(f"- {i.strip()}" for i in ingredients if i.strip())
            + "\n\nPréparation :\n"
            + instructions
        )

        meta = {
            "id": rid,
            "title": title,
            "season": season,
            "people": people,
            "source": "recipes",  # cohérent avec recipes_retriever
            "type": "salade",
        }

        docs.append(Document(page_content=text, metadata=meta))
        table.add_row(str(idx), rid, title, season, str(people))

    rprint(table)

    rprint(
        Panel.fit(
            f"[cyan]Insertion dans Chroma 'recipes'[/cyan] "
            f"({len(docs)} recettes de salades)"
        )
    )
    RECIPES_VS.add_documents(docs)

    rprint(Panel.fit("[bold green]Ingestion des salades terminée ✅[/bold green]"))


if __name__ == "__main__":
    ingest_salade_recipes()
