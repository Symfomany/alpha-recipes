# recipes/ingest_pdfs.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader


from .config import COOKBOOKS_VS, BASE_DIR  # adapté à ton chemin actuel


PDF_DIR = BASE_DIR / "pdfs"
MIN_TOKENS = 50  # on ignore les pages trop courtes (page de garde, pub, etc.)


def infer_category_and_title(stem: str) -> Tuple[str, str]:
    s = stem.lower()
    if "noel" in s or "noël" in s:
        return "noel", "Recettes magiques de Noël"
    if "italien" in s or "pates" in s or "pâtes" in s or "pasta" in s:
        return "italien", "Recettes italiennes (pâtes, bolo, etc.)"
    return "autre", stem.replace("_", " ").title()


def _token_len(text: str) -> int:
    return len((text or "").split())


def ingest_cookbook_pdfs() -> None:
    rprint(Panel.fit("[bold cyan]Ingestion des PDFs de cuisine → Chroma 'cookbooks'[/bold cyan]"))

    if not PDF_DIR.exists():
        rprint(f"[red]Dossier PDF inexistant : {PDF_DIR}[/red]")
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        rprint(f"[yellow]Aucun PDF trouvé dans {PDF_DIR}[/yellow]")
        return

    # Tableau récap des fichiers
    table = Table(title="PDFs détectés", show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("Fichier")
    table.add_column("Taille (Ko)")
    for idx, path in enumerate(pdf_files, start=1):
        size_kb = round(path.stat().st_size / 1024, 1)
        table.add_row(str(idx), path.name, str(size_kb))
    rprint(table)

    all_docs: List[Document] = []
    total_pages = 0
    total_kept = 0

    for path in pdf_files:
        category, title = infer_category_and_title(path.stem)

        rprint(Panel.fit(f"[bold green]Chargement[/bold green] {path.name}"))
        loader = PyPDFLoader(str(path))

        # 1) on charge les pages (1 Document par page)
        pages = loader.load()
        nb_pages = len(pages)
        total_pages += nb_pages

        kept_for_file = 0
        docs_for_file: List[Document] = []

        for page_idx, page in enumerate(pages, start=1):
            text = page.page_content or ""
            if _token_len(text) < MIN_TOKENS:
                # On ignore les pages trop courtes
                continue

            page.metadata = page.metadata or {}
            page.metadata.update(
                {
                    "id": path.stem,            # ex: recettes_italien
                    "filename": path.name,      # ex: recettes_italien.pdf
                    "source": "cookbook_pdf",
                    "category": category,       # "noel" / "italien" / "autre"
                    "book_title": title,
                    "page": page_idx,
                    "page_label": page.metadata.get("page_label", str(page_idx)),
                    "chunk_index": page_idx,    # 1 chunk = 1 page
                    "total_pages": nb_pages,
                }
            )
            docs_for_file.append(page)
            kept_for_file += 1

        total_kept += kept_for_file
        all_docs.extend(docs_for_file)

        rprint(
            f"  → [cyan]{nb_pages} pages[/cyan], "
            f"[green]{kept_for_file} pages retenues >= {MIN_TOKENS} tokens[/green] "
            f"pour [bold]{title}[/bold] (catégorie: {category})"
        )

    rprint(
        Panel.fit(
            f"[cyan]Insertion dans Chroma[/cyan] "
            f"({len(all_docs)} documents/pages, {total_pages} pages au total, "
            f"{total_kept} pages gardées après filtre longueur ≥ {MIN_TOKENS})"
        )
    )
    COOKBOOKS_VS.add_documents(all_docs)

    rprint(Panel.fit("[bold green]Ingestion cookbooks terminée ✅[/bold green]"))


if __name__ == "__main__":
    ingest_cookbook_pdfs()
