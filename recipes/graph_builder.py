"""
recipes/app/graph_builder.py

Construction du StateGraph LangGraph pour le compagnon de recettes.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from rich import print as rprint
from recipes.config import  get_memory_checkpointer

from .schema import (
    RecipeState,
    ANALYZE,
    CLASSIFY_RAG,
    RETRIEVE_RECIPES,
    RETRIEVE_COOKBOOKS,
    RETRIEVE_WEB,
    GRADE_RETRIEVAL,
    REWRITE_QUERY,
    CLARIFY_USER,
    AGENT,
    USTENSILS,
    NUTRITION,
    PLAN_BATCH,
    SHOPPING,
    STEPS,
    SAVE_STATE,
)
from . import nodes

def debug_print_graph_ascii() -> None:
    graph = build_graph()
    rprint("\n[bold cyan]Graph ASCII[/bold cyan]\n")
    graph.get_graph().print_ascii()

async def build_graph_async():
    builder = StateGraph(RecipeState)
    
    # --- nœuds principaux ---
    builder.add_node(ANALYZE, nodes.analyze_request_node)
    builder.add_node(CLASSIFY_RAG, nodes.classify_rag_node)

    builder.add_node(RETRIEVE_RECIPES, nodes.retrieve_recipes_node)
    builder.add_node(RETRIEVE_COOKBOOKS, nodes.retrieve_cookbooks_node)
    builder.add_node(RETRIEVE_WEB, nodes.retrieve_web_node)

    builder.add_node(GRADE_RETRIEVAL, nodes.grade_retrieval_node)
    builder.add_node(REWRITE_QUERY, nodes.rewrite_query_node)
    builder.add_node(CLARIFY_USER, nodes.clarify_user_node)

    builder.add_node(AGENT, nodes.agent_node)
    builder.add_node(USTENSILS, nodes.ustensils_node)
    builder.add_node(NUTRITION, nodes.nutrition_node)

    builder.add_node(PLAN_BATCH, nodes.plan_batch_cooking_node)
    builder.add_node(SHOPPING, nodes.build_shopping_list_node)
    builder.add_node(STEPS, nodes.generate_steps_node)
    builder.add_node(SAVE_STATE, nodes.save_session_node)

    # --- edges ---

    builder.set_entry_point(ANALYZE)
    builder.add_edge(ANALYZE, CLASSIFY_RAG)

    # Adaptive RAG routing
    builder.add_conditional_edges(
        CLASSIFY_RAG,
        lambda s: s.get("rag_strategy", "LOCAL_RECIPES"),
        {
            "NO_RAG": AGENT,
            "LOCAL_RECIPES": RETRIEVE_RECIPES,
            "COOKBOOKS": RETRIEVE_COOKBOOKS,
            "WEB": RETRIEVE_WEB,
        },
    )

    # After any retrieval -> grade
    builder.add_edge(RETRIEVE_RECIPES, GRADE_RETRIEVAL)
    builder.add_edge(RETRIEVE_COOKBOOKS, GRADE_RETRIEVAL)
    builder.add_edge(RETRIEVE_WEB, GRADE_RETRIEVAL)

    # Corrective RAG routing
    def _route_quality(state: RecipeState) -> str:
        return (state.get("retrieval_quality") or "GOOD").upper()

    builder.add_conditional_edges(
        GRADE_RETRIEVAL,
        _route_quality,
        {
            "GOOD": AGENT,
            "BAD": REWRITE_QUERY,
            "AMBIGUOUS": CLARIFY_USER,
        },
    )

    # BAD -> réécriture puis reclassification
    builder.add_edge(REWRITE_QUERY, CLASSIFY_RAG)

    # AMBIGUOUS -> on renvoie vers AGENT quand même pour proposer quelque chose
    builder.add_edge(CLARIFY_USER, AGENT)

    # Agent -> ustensiles + nutrition
    builder.add_edge(AGENT, USTENSILS)
    builder.add_edge(USTENSILS, NUTRITION)

    # Suite finale
    builder.add_edge(NUTRITION, PLAN_BATCH)
    builder.add_edge(PLAN_BATCH, SHOPPING)
    builder.add_edge(SHOPPING, STEPS)
    builder.add_edge(STEPS, SAVE_STATE)
    builder.add_edge(SAVE_STATE, END)
    
    checkpointer = get_memory_checkpointer()
    return builder.compile(checkpointer=checkpointer)
    
    return graph

def build_graph():
    builder = StateGraph(RecipeState)

    # --- nœuds principaux ---
    builder.add_node(ANALYZE, nodes.analyze_request_node)
    builder.add_node(CLASSIFY_RAG, nodes.classify_rag_node)

    builder.add_node(RETRIEVE_RECIPES, nodes.retrieve_recipes_node)
    builder.add_node(RETRIEVE_COOKBOOKS, nodes.retrieve_cookbooks_node)
    builder.add_node(RETRIEVE_WEB, nodes.retrieve_web_node)

    builder.add_node(GRADE_RETRIEVAL, nodes.grade_retrieval_node)
    builder.add_node(REWRITE_QUERY, nodes.rewrite_query_node)
    builder.add_node(CLARIFY_USER, nodes.clarify_user_node)

    builder.add_node(AGENT, nodes.agent_node)
    builder.add_node(USTENSILS, nodes.ustensils_node)
    builder.add_node(NUTRITION, nodes.nutrition_node)

    builder.add_node(PLAN_BATCH, nodes.plan_batch_cooking_node)
    builder.add_node(SHOPPING, nodes.build_shopping_list_node)
    builder.add_node(STEPS, nodes.generate_steps_node)
    builder.add_node(SAVE_STATE, nodes.save_session_node)

    # --- edges ---

    builder.set_entry_point(ANALYZE)
    builder.add_edge(ANALYZE, CLASSIFY_RAG)

    # Adaptive RAG routing
    builder.add_conditional_edges(
        CLASSIFY_RAG,
        lambda s: s.get("rag_strategy", "LOCAL_RECIPES"),
        {
            "NO_RAG": AGENT,
            "LOCAL_RECIPES": RETRIEVE_RECIPES,
            "COOKBOOKS": RETRIEVE_COOKBOOKS,
            "WEB": RETRIEVE_WEB,
        },
    )

    # After any retrieval -> grade
    builder.add_edge(RETRIEVE_RECIPES, GRADE_RETRIEVAL)
    builder.add_edge(RETRIEVE_COOKBOOKS, GRADE_RETRIEVAL)
    builder.add_edge(RETRIEVE_WEB, GRADE_RETRIEVAL)

    # Corrective RAG routing
    def _route_quality(state: RecipeState) -> str:
        return (state.get("retrieval_quality") or "GOOD").upper()

    builder.add_conditional_edges(
        GRADE_RETRIEVAL,
        _route_quality,
        {
            "GOOD": AGENT,
            "BAD": REWRITE_QUERY,
            "AMBIGUOUS": CLARIFY_USER,
        },
    )

    # BAD -> réécriture puis reclassification
    builder.add_edge(REWRITE_QUERY, CLASSIFY_RAG)

    # AMBIGUOUS -> on renvoie vers AGENT quand même pour proposer quelque chose
    builder.add_edge(CLARIFY_USER, AGENT)

    # Agent -> ustensiles + nutrition
    builder.add_edge(AGENT, USTENSILS)
    builder.add_edge(USTENSILS, NUTRITION)

    # Suite finale
    builder.add_edge(NUTRITION, PLAN_BATCH)
    builder.add_edge(PLAN_BATCH, SHOPPING)
    builder.add_edge(SHOPPING, STEPS)
    builder.add_edge(STEPS, SAVE_STATE)
    builder.add_edge(SAVE_STATE, END)
    
    graph = builder.compile()
    
    return graph
