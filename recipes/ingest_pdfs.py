# recipes/ingest_pdfs.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import COOKBOOKS_VS, BASE_DIR  # adapté à ton chemin actuel


PDF_DIR = BASE_DIR / "pdfs"

# splitter pour découper chaque page en chunks RAG
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", "!", "?", " "],
)


def infer_category_and_title(stem: str) -> Tuple[str, str]:
    """
    Déduit une catégorie et un titre humain à partir du nom du fichier PDF.
    """
    s = stem.lower()
    if "noel" in s or "noël" in s:
        return "noel", "Recettes magiques de Noël"
    if "italien" in s or "pates" in s or "pâtes" in s or "pasta" in s:
        return "italien", "Recettes italiennes (pâtes, bolo, etc.)"
    return "autre", stem.replace("_", " ").title()


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
    total_chunks = 0

    for path in pdf_files:
        category, title = infer_category_and_title(path.stem)

        rprint(Panel.fit(f"[bold green]Chargement[/bold green] {path.name}"))
        loader = PyPDFLoader(str(path))

        # 1) on charge les pages
        pages = loader.load()  # 1 Document par page
        nb_pages = len(pages)
        total_pages += nb_pages

        # 2) on les découpe en chunks RAG
        docs = text_splitter.split_documents(pages)
        nb_chunks = len(docs)
        total_chunks += nb_chunks

        rprint(
            f"  → [cyan]{nb_pages} pages[/cyan], "
            f"[magenta]{nb_chunks} chunks[/magenta] pour "
            f"[bold]{title}[/bold] (catégorie: {category})"
        )

        # Ajout des métadatas + debug
        for i, d in enumerate(docs, start=1):
            d.metadata = d.metadata or {}
            d.metadata.update(
                {
                    "id": path.stem,          # ex: recettes_magiques_noel
                    "filename": path.name,    # ex: recettes_magiques_noel.pdf
                    "source": "cookbook_pdf",
                    "category": category,     # "noel" / "italien" / "autre"
                    "book_title": title,
                    "chunk_index": i,
                }
            )
        all_docs.extend(docs)

        rprint(
            f"  → [green]{nb_chunks} chunks[/green] ajoutés au buffer "
            f"(total buffer: {len(all_docs)})"
        )

    rprint(
        Panel.fit(
            f"[cyan]Insertion dans Chroma[/cyan] "
            f"({len(all_docs)} documents/chunks, {total_pages} pages au total, "
            f"{total_chunks} chunks cumulés)"
        )
    )
    COOKBOOKS_VS.add_documents(all_docs)

    rprint(Panel.fit("[bold green]Ingestion cookbooks terminée ✅[/bold green]"))


if __name__ == "__main__":
    ingest_cookbook_pdfs()
