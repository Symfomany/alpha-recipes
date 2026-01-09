"""
recipes/app/nodes.py

Implémentation des nœuds du graphe recipes.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage

from .config import LLM
from .schema import (
    RecipeState,
    RagStrategy,
    RetrievalQuality,
    RetrievedDoc,
    CandidateRecipe,
    UstensilInfo,
    ShoppingItem,
)
from . import tools
from rich import print as rprint


# --- helpers LLM ---


def _log_node(name: str) -> None:
    rprint(f"[bold magenta]→ NODE[/bold magenta] [cyan]{name}[/cyan]")


def _llm_chat(messages: List[Any]) -> str:
    """Appel simple au LLM avec des messages LangChain."""
    resp = LLM.invoke(messages)
    if isinstance(resp, str):
        return resp
    return resp.content  # ChatMessage


# --- ANALYZE_REQUEST ---


def analyze_request_node(state: RecipeState) -> RecipeState:
    """
    Normalise la demande utilisateur : personnes, temps, régime, matériel, etc.
    Prompt simple pour extraire les méta-infos.
    """
    _log_node("ANALYZE_REQUEST")
    query = state.get("query") or ""
    messages = [
        HumanMessage(
            content=(
                "Tu es un assistant culinaire. Analyse la demande suivante et "
                "retourne un JSON avec les clés: normalized_request, people, "
                "max_time_minutes, diet, allergies (liste), equipment_available (liste).\n\n"
                f"Demande: {query}"
            )
        )
    ]
    text = _llm_chat(messages)

    # Pour rester simple, on laisse le parsing JSON à plus tard;
    # ici on stocke le texte brut.
    state["normalized_request"] = text
    state.setdefault("messages", []).append(HumanMessage(content=query))
    state["messages"].append(AIMessage(content=text))
    return state


# --- CLASSIFY_RAG (Adaptive RAG) ---

# LOCAL_RECIPES → vector store recettes
# COOKBOOKS → vector store PDFs / livres de cuisine
# WEB → Tavily (recherche web)
def classify_rag_node(state: RecipeState) -> RecipeState:
    """
    Choisit la stratégie RAG : NO_RAG / LOCAL_RECIPES / COOKBOOKS / WEB.
    """
    _log_node("CLASSIFY_RAG")
    query = state.get("query") or ""
    messages = [
        HumanMessage(
            content=(
                "Tu es un routeur RAG spécialisé en cuisine.\n"
                "Tu dois décider de la meilleure source principale pour répondre "
                "à la question suivante.\n\n"
                "Choisis exactement UN token parmi :\n"
                "- NO_RAG      : répondre uniquement avec tes connaissances générales\n"
                "- LOCAL_RECIPES : recettes locales (base de recettes classiques)\n"
                "- COOKBOOKS   : livres de cuisine / PDF (recettes détaillées)\n"
                "- WEB         : recherche web (Tavily) pour informations récentes\n\n"
                "Réponds UNIQUEMENT par l’un de ces tokens (sans explication).\n\n"
                f"Question utilisateur : {query}"
            )
        )
    ]
    strategy_text = _llm_chat(messages).strip().upper()
    valid: List[RagStrategy] = ["NO_RAG", "LOCAL_RECIPES", "COOKBOOKS", "WEB"]  # type: ignore
    # Fallback raisonnable si le LLM sort autre chose
    strategy: RagStrategy = (
        strategy_text if strategy_text in valid else "LOCAL_RECIPES"  # type: ignore
    )
    _log_node("strategy chosen: " + strategy)
    state["rag_strategy"] = strategy
    return state


# --- RETRIEVE_* ---

def retrieve_recipes_node(state: RecipeState) -> RecipeState:
    _log_node("RETRIEVE_RECIPES")
    query = state.get("query") or ""

    # RAG local sur la collection "recipes"
    docs: list[Document] = RECIPES_VS.similarity_search(query, k=5)
    retrieved: list[RetrievedDoc] = [
        {
            "id": d.metadata.get("id", d.page_content[:50]),
            "source": "recipes",
            "content": d.page_content,
            "metadata": d.metadata,
        }
        for d in docs
    ]

    _log_node(f"Found {len(retrieved)} docs in RECIPES_VS")
    return {"retrieved_docs": retrieved}


def retrieve_cookbooks_node(state: RecipeState) -> RecipeState:
    _log_node("RETRIEVE_COOKBOOKS")
    query = state.get("query") or ""

    docs: list[Document] = COOKBOOKS_VS.similarity_search(query, k=5)
    retrieved: list[RetrievedDoc] = [
        {
            "id": d.metadata.get("id", d.page_content[:50]),
            "source": "cookbook_pdf",
            "content": d.page_content,
            "metadata": d.metadata,
        }
        for d in docs
    ]

    _log_node(f"Found {len(retrieved)} docs in COOKBOOKS_VS")
    return {"retrieved_docs": retrieved}


def retrieve_web_node(state: RecipeState) -> RecipeState:
    _log_node("RETRIEVE_WEB")
    query = state.get("query") or ""
    result = tools.web_search.invoke({"query": query})

    docs: list[RetrievedDoc] = [
        {
            "id": "tavily-0",
            "source": "web",
            "content": str(result),
            "metadata": {},
        }
    ]
    _log_node("Found 1 web doc (Tavily)")
    return {"retrieved_docs": docs}


# --- GRADE_RETRIEVAL (Corrective RAG) ---

def grade_retrieval_node(state: RecipeState) -> RecipeState:
    _log_node("GRADE_RETRIEVAL")
    query = state.get("query") or ""
    docs = state.get("retrieved_docs", [])
    text_docs = "\n\n".join(d.get("content", "") for d in docs)[:4000]

    messages = [
        HumanMessage(
            content=(
                "Tu es un évaluateur de RAG pour la cuisine.\n"
                "Les documents suivants sont-ils suffisants, insuffisants ou ambigus "
                "pour répondre à la question ?\n"
                "Réponds uniquement par: GOOD, BAD ou AMBIGUOUS.\n\n"
                f"Question: {query}\n\nDocs:\n{text_docs}"
            )
        )
    ]
    quality_text = _llm_chat(messages).strip().upper()
    valid: List[RetrievalQuality] = ["GOOD", "BAD", "AMBIGUOUS"]  # type: ignore
    quality: RetrievalQuality = (
        quality_text if quality_text in valid else "GOOD"  # type: ignore
    )
    return {
        "retrieval_quality": quality,
        "clarification_needed": quality == "AMBIGUOUS",
    }


def rewrite_query_node(state: RecipeState) -> RecipeState:
    """Réécrit la requête pour un meilleur retrieval."""
    query = state.get("query") or ""
    messages = [
        HumanMessage(
            content=(
                "Réécris la question suivante pour qu'elle soit plus précise pour un "
                "moteur de recherche de recettes. Garde le français.\n\n"
                f"Question: {query}"
            )
        )
    ]
    new_query = _llm_chat(messages).strip()
    return {"query": new_query}


def clarify_user_node(state: RecipeState) -> RecipeState:
    """
    Prépare une question de clarification pour l'utilisateur
    (à afficher côté UI).
    """
    query = state.get("query") or ""
    messages = [
        HumanMessage(
            content=(
                "La question de cuisine suivante est ambiguë. Formule UNE seule question "
                "courte pour clarifier les contraintes (régime, matériel, temps, etc.).\n\n"
                f"Question: {query}"
            )
        )
    ]
    question = _llm_chat(messages).strip()
    return {
        "clarification_question": question,
        "clarification_needed": True,
    }


# --- AGENT_NODE (Agentic RAG simplifié) ---


def agent_node(state: RecipeState) -> RecipeState:
    """
    Agent qui combine les docs RAG + connaissances LLM
    pour proposer 1..N recettes candidates.
    """
    _log_node("AGENT")
    query = state.get("query") or ""
    # _log_node("Query" + query)
    docs = state.get("retrieved_docs", [])
    context = "\n\n".join(d.get("content", "") for d in docs)[:6000]

    messages = [
        HumanMessage(
            content=(
                "Tu es un chef assistant. À partir de la question de l'utilisateur "
                "et du contexte (recettes / techniques), propose 3 recettes candidates.\n"
                "Réponds sous forme de liste numérotée avec pour chaque recette : "
                "titre, résumé, liste d'ingrédients, temps total.\n\n"
                f"Question: {query}\n\nContexte:\n{context}"
            )
        )
    ]
    text = _llm_chat(messages)

    # On stocke brut dans candidate_recipes_text pour commencer.
    candidate: CandidateRecipe = {
        "id": "candidate-raw",
        "title": "Candidats décrits en texte libre",
        "summary": text,
        "steps": [],
        "ingredients": [],
        "score": 0.0,
        "source": "llm",
        "url": None,
    }

    return {"candidate_recipes": [candidate]}


# --- USTENSILS_NODE ---


def ustensils_node(state: RecipeState) -> RecipeState:
    """
    Vérifie / propose des ustensiles à partir de la demande
    et des recettes candidates.
    """
    query = state.get("query") or ""
    task = f"{query} (batch cooking / préparation proposée)"
    raw = tools.ustensils_retriever.invoke({"task": task})
    ustensils: List[UstensilInfo] = []
    for u in raw:  # type: ignore
        ustensils.append(
            {
                "id": u.get("id", ""),
                "name": u.get("name"),
                "kind": u.get("kind"),
                "required_for": [],
                "has_user": False,
                "suggestion_url": u.get("metadata", {}).get("url"),
                "notes": u.get("content"),
            }
        )
    state["ustensils_needed"] = ustensils
    return state


# --- NUTRITION_NODE ---


def nutrition_node(state: RecipeState) -> RecipeState:
    """
    Résumé nutritionnel à partir des ingrédients (si dispo).
    """
    ingredients: List[str] = []
    for c in state.get("candidate_recipes", []):
        ingredients.extend(c.get("ingredients", []))
    if not ingredients:
        state["nutrition_summary"] = None
        return state

    summary = tools.nutrition_tool.invoke({"ingredients": ingredients})
    state["nutrition_summary"] = str(summary)
    return state


# --- PLAN_BATCH_COOKING ---


def plan_batch_cooking_node(state: RecipeState) -> RecipeState:
    """
    Organise 1..N recettes en plan de batch cooking simple.
    """
    candidates = state.get("candidate_recipes", [])
    state["batch_plan"] = candidates
    state["batch_notes"] = (
        "Plan de batch cooking généré automatiquement sur la base des recettes candidates."
    )
    return state


# --- BUILD_SHOPPING_LIST ---


def build_shopping_list_node(state: RecipeState) -> RecipeState:
    """
    Construit une liste de courses simplifiée à partir des recettes + ustensiles.
    """
    items: List[ShoppingItem] = []
    for c in state.get("candidate_recipes", []):
        for ing in c.get("ingredients", []):
            items.append(
                {
                    "name": ing,
                    "quantity": "",
                    "category": "ingrédients",
                    "is_ustensil": False,
                    "optional": False,
                }
            )

    for u in state.get("ustensils_needed", []):
        items.append(
            {
                "name": u.get("name", ""),
                "quantity": "1",
                "category": "ustensiles",
                "is_ustensil": True,
                "optional": True,
            }
        )

    state["shopping_list"] = items
    return state


# --- GENERATE_STEPS ---


def generate_steps_node(state: RecipeState) -> RecipeState:
    """
    Génère les étapes détaillées de cuisson + planning.
    """
    query = state.get("query") or ""
    plan = state.get("batch_plan", [])
    context = "\n\n".join(c.get("summary", "") for c in plan)

    messages = [
        HumanMessage(
            content=(
                "À partir du plan de recettes suivant, génère des étapes détaillées "
                "pour cuisiner efficacement (batch cooking si pertinent).\n"
                "Structure ta réponse en étapes numérotées, avec temps estimé.\n\n"
                f"Demande utilisateur: {query}\n\nPlan:\n{context}"
            )
        )
    ]
    text = _llm_chat(messages)
    state["cooking_steps"] = text.split("\n")
    state["timelines"] = "Planning indicatif généré dans les étapes."
    state["tips"] = "Ajuste les temps de cuisson selon la puissance de ton four / plaques."
    return state


# --- SAVE_SESSION ---


def save_session_node(state: RecipeState) -> RecipeState:
    """
    Pour l'instant, ne fait que marquer la fin; la persistance est gérée
    par SqliteSaver dans LangGraph.
    Tu pourras plus tard appeler ici database.py pour sauver favoris / historique.
    """
    return state
