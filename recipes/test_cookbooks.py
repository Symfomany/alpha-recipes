from rich import print as rprint
from rich.panel import Panel

from config import COOKBOOKS_VS
from schema import RetrievedDoc


def main():
    query = "recette des pâtes Carbonara à base de spaghettis"
    rprint(Panel.fit(f"[bold cyan]Test cookbooks RAG[/bold cyan]\n{query}"))

    docs = COOKBOOKS_VS.similarity_search(query, k=5)
    for i, d in enumerate(docs, start=1):
        rprint(Panel.fit(f"[bold green]Doc {i}[/bold green]"))
        rprint(d.metadata)
        rprint(d.page_content[:600] + "...")


if __name__ == "__main__":
    main()
