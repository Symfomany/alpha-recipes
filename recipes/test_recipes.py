# recipes/test_recipes.py

from rich import print as rprint
from rich.panel import Panel

from config import RECIPES_VS
from langchain_core.documents import Document


def main() -> None:
    query = "salade de quinoa aux agrumes pour 4 personnes en été"
    rprint(
        Panel.fit(
            f"[bold cyan]Test LOCAL_RECIPES RAG[/bold cyan]\n[white]{query}[/white]"
        )
    )

    # top‑10, avec scores
    docs_with_scores: list[tuple[Document, float]] = RECIPES_VS.similarity_search_with_score(
        query=query,
        k=10,
    )

    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata or {}
        header = (
            f"[bold green]Doc {i}[/bold green]  "
            f"[yellow]score (distance) = {score:.4f}[/yellow]"
        )
        rprint(
            Panel.fit(
                f"{header}\n"
                f"[bold]id[/bold]: {meta.get('id')}\n"
                f"[bold]title[/bold]: {meta.get('title')}\n"
                f"[bold]season[/bold]: {meta.get('season')}\n"
                f"[bold]people[/bold]: {meta.get('people')}\n"
                f"[bold]source[/bold]: {meta.get('source')}\n\n"
                f"[bold]Preview:[/bold]\n{doc.page_content[:600]}..."
            )
        )


if __name__ == "__main__":
    main()
