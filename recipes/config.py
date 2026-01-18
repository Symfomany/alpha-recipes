"""
recipes/app/config.py

Configuration globale : LLM Mistral 3B local, embeddings, vector stores,
Tavily, checkpointer SQLite (async) pour LangGraph.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite  # type: ignore
from langgraph.checkpoint.memory import MemorySaver

from check import _log_cuda_status


# --- chemins & .env ---

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
CHECKPOINT_DB = DATA_DIR / "recipes_checkpoints.sqlite"

DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

load_dotenv()



    
# appeler le check au chargement du module
_log_cuda_status()

# --- LLM principal : Mistral 3B local via Ollama ---


def get_llm() -> Ollama:
    """
    Retourne le LLM principal (Mistral 3B local via Ollama).

    Assure-toi que le modèle 'ministral-3:3b' est présent côté Ollama :
        ollama pull ministral-3:3b
    """
    model_name = os.getenv("MISTRAL_LOCAL_MODEL", "ministral-3:3b")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    return Ollama(
        model=model_name,
        temperature=temperature,
    )


# --- embeddings & vector stores ---


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Embeddings pour les vector stores (recettes, PDFs, ustensiles).

    Ici HuggingFaceEmbeddings, 100 % local.
    """
    model_name = os.getenv(
        "EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    return HuggingFaceEmbeddings(model_name=model_name)


def get_vectorstores() -> Tuple[Chroma, Chroma, Chroma]:
    """
    Initialise / ouvre les vector stores :
    - recipes   : recettes scrapées / JSON-LD
    - cookbooks : PDFs de cuisine
    - ustensils : catalogue d'ustensiles

    Chaque store est dans un sous-dossier de CHROMA_DIR.
    """
    embeddings = get_embeddings()

    recipes_vs = Chroma(
        collection_name="pdfs",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR / "recipes"),
    )

    cookbooks_vs = Chroma(
        collection_name="cookbooks",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR / "cookbooks"),
    )

    ustensils_vs = Chroma(
        collection_name="ustensils",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR / "ustensils"),
    )

    return recipes_vs, cookbooks_vs, ustensils_vs


# --- Tavily (web search) ---


def get_tavily_tool() -> TavilySearch:
    """
    Tool Tavily pour la recherche web (Adaptive / Agentic RAG).

    Nécessite TAVILY_API_KEY dans l'environnement.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise RuntimeError("TAVILY_API_KEY manquant pour Tavily.")

    # La clé est lue automatiquement par TavilySearch
    return TavilySearch(
        max_results=5,
        include_answer=True,
        include_raw_content=False,
        include_images=False,
    )


# --- checkpointer SQLite async pour LangGraph ---


async def get_async_checkpointer() -> AsyncSqliteSaver:
    """
    Checkpointer async pour LangGraph (obligatoire pour .astream / .ainvoke).

    Le fichier est créé dans data/recipes_checkpoints.sqlite.
    """
    conn = await aiosqlite.connect(str(CHECKPOINT_DB))
    return AsyncSqliteSaver(conn)



# Getter Checkpointer of Graph
def get_memory_checkpointer() -> MemorySaver:
    return MemorySaver()


# --- helpers globaux (sync) ---

# Ces objets sont utilisables directement dans nodes/tools.
LLM = get_llm()
RECIPES_VS, COOKBOOKS_VS, USTENSILS_VS = get_vectorstores()
TAVILY_TOOL = get_tavily_tool()
