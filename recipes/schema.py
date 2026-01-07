"""
recipes/app/schema.py

Schéma de state pour le graphe LangGraph du compagnon de recettes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


# --- constantes de nœuds (pour graph_builder.py) ---

START = "START"
END = "END"

ANALYZE = "ANALYZE_REQUEST"
CLASSIFY_RAG = "CLASSIFY_RAG"

RETRIEVE_RECIPES = "RETRIEVE_RECIPES"
RETRIEVE_COOKBOOKS = "RETRIEVE_COOKBOOKS"
RETRIEVE_WEB = "RETRIEVE_WEB"

GRADE_RETRIEVAL = "GRADE_RETRIEVAL"
REWRITE_QUERY = "REWRITE_QUERY"
CLARIFY_USER = "CLARIFY_USER"

AGENT = "AGENT_NODE"
USTENSILS = "USTENSILS_NODE"
NUTRITION = "NUTRITION_NODE"

PLAN_BATCH = "PLAN_BATCH_COOKING"
SHOPPING = "BUILD_SHOPPING_LIST"
STEPS = "GENERATE_STEPS"
SAVE_STATE = "SAVE_SESSION"


# --- types utilitaires ---

RagStrategy = Literal[
    "NO_RAG",          # réponse directe
    "LOCAL_RECIPES",   # vector store recettes
    "COOKBOOKS",       # PDF / fiches techniques
    "WEB",             # Tavily
]

RetrievalQuality = Literal["GOOD", "BAD", "AMBIGUOUS"]


class RetrievedDoc(TypedDict, total=False):
    id: str
    source: str          # "recipes", "cookbooks", "web"
    content: str
    metadata: Dict[str, Any]


class CandidateRecipe(TypedDict, total=False):
    id: str
    title: str
    summary: str
    steps: List[str]
    ingredients: List[str]
    score: float
    source: str          # "recipes", "cookbooks", "web"
    url: Optional[str]


class UstensilInfo(TypedDict, total=False):
    id: str
    name: str
    kind: str            # presse-purée, moulin, mortier, etc.
    required_for: List[str]   # ids de recettes / techniques
    has_user: bool
    suggestion_url: Optional[str]   # lien Cuisine Addict
    notes: Optional[str]


class ShoppingItem(TypedDict, total=False):
    name: str
    quantity: str
    category: str        # légumes, épicerie, frais, ustensiles
    is_ustensil: bool
    optional: bool


# --- state principal du graphe ---

class RecipeState(TypedDict, total=False):
    """
    State passé entre les nœuds du graphe.

    Il doit rester JSON-serializable pour la persistance (SqliteSaver).
    """

    # entrée utilisateur brute + historisée
    query: str
    messages: List[Any]  # messages LangChain (HumanMessage, AIMessage, ...)

    # analyse / normalisation
    normalized_request: Optional[str]
    people: Optional[int]
    max_time_minutes: Optional[int]
    diet: Optional[str]              # vegan, végétarien, sans lactose, etc.
    allergies: List[str]
    equipment_available: List[str]   # inventaire basique (four, mixeur…)

    # stratégie RAG / corrective
    rag_strategy: Optional[RagStrategy]
    retrieved_docs: List[RetrievedDoc]
    retrieval_quality: Optional[RetrievalQuality]
    clarification_needed: bool
    clarification_question: Optional[str]

    # recettes candidates / choisies
    candidate_recipes: List[CandidateRecipe]
    chosen_recipe: Optional[CandidateRecipe]

    # batch cooking
    batch_plan: List[CandidateRecipe]   # plusieurs recettes organisées
    batch_notes: Optional[str]

    # ustensiles & nutrition
    ustensils_needed: List[UstensilInfo]
    nutrition_summary: Optional[str]

    # liste de courses
    shopping_list: List[ShoppingItem]

    # instructions finales
    cooking_steps: List[str]        # étapes détaillées
    timelines: Optional[str]        # planning (parallelisation, batch)
    tips: Optional[str]             # conseils du chef

    # persistance / métadonnées
    thread_id: Optional[str]        # pour SqliteSaver
    error: Optional[str]            # message d'erreur éventuel
