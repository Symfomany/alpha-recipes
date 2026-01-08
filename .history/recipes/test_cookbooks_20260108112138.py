from rich import print as rprint
from rich.panel import Panel

from config import COOKBOOKS_VS
from schema import RetrievedDoc  # si tu l'utilises ailleurs


def main() -> None:
    query = "Je veux faire des lasagnes à la bolognaises pour 4 personnes avec une option végétarienne."
    rprint(
        Panel.fit(
            f"[bold cyan]Test cookbooks RAG[/bold cyan]\n[white]{query}[/white]"
        )
    )

    # Retourne une liste de tuples (Document, score)
    docs_with_scores = COOKBOOKS_VS.similarity_search_with_score(
        query=query,
        k=3,  # top‑3
    )  # score = distance (plus petit = plus proche, avec Chroma) [web:153][web:159][web:162]

    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        header = (
            f"[bold green]Doc {i}[/bold green]  "
            f"[yellow]score (distance) = {score:.4f}[/yellow]"
        )
        meta_str = "\n".join(f"{k}: {v}" for k, v in (doc.metadata or {}).items())
        content_preview = (doc.page_content or "")[:8600] + "..."

        rprint(
            Panel.fit(
                f"{header}\n\n[bold]Metadata:[/bold]\n{meta_str}\n\n"
                f"[bold]Preview:[/bold]\n{content_preview}"
            )
        )


if __name__ == "__main__":
    main()
