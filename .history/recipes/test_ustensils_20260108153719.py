# recipes/test_ustensils.py

from rich import print as rprint
from rich.panel import Panel

from config import USTENSILS_VS
from langchain_core.documents import Document


def main() -> None:
    query = "purée pour 6 personnes"
    rprint(
        Panel.fit(
            f"[bold cyan]Test USTENSILS RAG[/bold cyan]\n[white]{query}[/white]"
        )
    )

    docs_with_scores: list[tuple[Document, float]] = USTENSILS_VS.similarity_search_with_score(
        query=query,
        k=5,
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
                f"[bold]name[/bold]: {meta.get('name')}\n"
                f"[bold]kind[/bold]: {meta.get('kind')}\n"
                f"[bold]tasks[/bold]: {meta.get('tasks')}\n"
                f"[bold]url[/bold]: {meta.get('url')}\n\n"
                f"[bold]Preview:[/bold]\n{doc.page_content[:300]}..."
            )
        )


if __name__ == "__main__":
    main()
