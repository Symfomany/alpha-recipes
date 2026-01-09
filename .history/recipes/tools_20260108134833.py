"""
recipes/tools.py

Définition des tools utilisés par l'Agentic RAG :
- retrievers (recettes, cookbooks, ustensiles)
- Tavily web search
- nutrition simple
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.tools import tool
from langchain_core.documents import Document

from .config import RECIPES_VS, COOKBOOKS_VS, USTENSILS_VS, TAVILY_TOOL


# --- retrievers basés sur Chroma ---


@tool("recipes_retriever", return_direct=False)
def recipes_retriever(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Recherche des recettes (vector store local) pertinentes pour la requête."""
    docs: List[Document] = RECIPES_VS.similarity_search(query, k=k)
    rprint(f"[recipes/tools] recipes_retriever: found {len(docs)} docs for query '{query}'")
    return [
        {
            "id": d.metadata.get("id", d.page_content[:50]),
            "source": "recipes",
            "content": d.page_content,
            "metadata": d.metadata,
        }
        for d in docs
    ]


@tool("cookbooks_retriever", return_direct=False)
def cookbooks_retriever(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Recherche dans les PDFs / livres de cuisine vectorisés."""
    docs: List[Document] = COOKBOOKS_VS.similarity_search(query, k=k)
    return [
        {
            "id": d.metadata.get("id", d.page_content[:50]),
            "source": "cookbook_pdf",
            "content": d.page_content,
            "metadata": d.metadata,
        }
        for d in docs
    ]


@tool("ustensils_retriever", return_direct=False)
def ustensils_retriever(task: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Suggère des ustensiles adaptés à une tâche (ex: 'purée pour 6 personnes').
    Utilise le vector store ustensiles (scrap + CSV).
    """
    docs: List[Document] = USTENSILS_VS.similarity_search(task, k=k)
    return [
        {
            "id": d.metadata.get("id", d.page_content[:50]),
            "name": d.metadata.get("name"),
            "kind": d.metadata.get("kind"),
            "source": "ustensils",
            "content": d.page_content,
            "metadata": d.metadata,
        }
        for d in docs
    ]


# --- Tavily (web search) ---


@tool("web_search", return_direct=False)
def web_search(query: str) -> Any:
    """Recherche web via Tavily pour compléter le RAG (ingrédients rares, tendances…)."""
    return TAVILY_TOOL.invoke({"query": query})


# --- nutrition simple (placeholder) ---


@tool("nutrition_tool", return_direct=False)
def nutrition_tool(ingredients: List[str]) -> str:
    """
    Fournit un résumé nutritionnel grossier à partir de la liste d'ingrédients.

    Implémentation simplifiée (à remplacer par un vrai appel API si besoin).
    """
    joined = ", ".join(ingredients)
    return (
        f"Résumé nutritionnel approximatif pour : {joined}. "
        "Attention aux allergènes éventuels et à la quantité de matières grasses."
    )
