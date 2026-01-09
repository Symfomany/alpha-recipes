from rich import print as rprint
from rich.panel import Panel

from config import COOKBOOKS_VS
from schema import RetrievedDoc  # si tu l'utilises ailleurs



MIN_TOKENS = 50  # seuil de longueur minimale


def doc_length_tokens(text: str) -> int:
    # approximation simple : split sur les espaces
    return len(text.split())


def main() -> None:
    query = "Je veux faire des tartines d'avocat"
    rprint(
        Panel.fit(
            f"[bold cyan]Test cookbooks RAG[/bold cyan]\n[white]{query}[/white]"
        )
    )

    # Retourne une liste de tuples (Document, score)
    docs_with_scores = COOKBOOKS_VS.similarity_search_with_score(
        query=query,
        k=10,  # top‑3
    )  # score = distance (plus petit = plus proche, avec Chroma) [web:153][web:159][web:162]

      # filtrage sur la longueur, puis on garde les 3 meilleurs
    filtered = [
        (doc, score)
        for (doc, score) in docs_with_scores
        if doc_length_tokens(doc.page_content or "") >= MIN_TOKENS
    ][:3]

    for i, (doc, score) in enumerate(filtered, start=1):
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
