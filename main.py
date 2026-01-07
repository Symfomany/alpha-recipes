from __future__ import annotations

import asyncio

from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from langchain_core.messages import HumanMessage

from recipes.graph_builder import build_graph_async
from recipes.schema import RecipeState


async def run_stream(query: str) -> RecipeState:
    graph = await build_graph_async()

    rprint("\n[bold cyan]Graph ASCII[/bold cyan]\n")
    graph.get_graph().print_ascii()

    state: RecipeState = {
        "query": query,
        "messages": [HumanMessage(content=query)],
    }
    config = {"configurable": {"thread_id": "demo_stream"}}

    rprint(Panel.fit("[bold cyan]Streaming du graphe recettes[/bold cyan]"))

    # Stream des updates node par node
    async for chunk in graph.astream(
        state,
        config=config,
        stream_mode="updates",  # ou "values"
    ):
        for node, update in chunk.items():
            rprint(Panel.fit(f"[bold green]{node}[/bold green]"))

            if "retrieved_docs" in update:
                rprint(f"[yellow]retrieved_docs[/yellow]: "
                       f"{len(update['retrieved_docs'])} docs")

            if "candidate_recipes" in update:
                rprint("[magenta]Candidats recettes (résumé brut):[/magenta]")
                rprint(update["candidate_recipes"])

            if "shopping_list" in update:
                rprint("[cyan]Liste de courses (partielle ou finale)[/cyan]")
                table = Table(show_lines=True)
                table.add_column("Type")
                table.add_column("Nom")
                table.add_column("Quantité")
                for item in update["shopping_list"]:
                    kind = "Ustensile" if item.get("is_ustensil") else "Ingrédient"
                    table.add_row(
                        kind,
                        item.get("name", ""),
                        item.get("quantity", ""),
                    )
                rprint(table)

            if "cooking_steps" in update:
                rprint("[bold]Étapes de cuisson (partielles ou finales):[/bold]")
                for line in update["cooking_steps"][:10]:
                    rprint(f"- {line}")

    # État final (optionnel si tu veux le réutiliser)
    result: RecipeState = await graph.ainvoke(state, config=config)
    return result


if __name__ == "__main__":
    # user_query = input(
    #     "Décris ta situation (ingrédients, personnes, temps, contraintes) :\n> "
    # )
    user_query = "Cuisiner un poulet tikka massala pour 6 personnes"
    asyncio.run(run_stream(user_query))
