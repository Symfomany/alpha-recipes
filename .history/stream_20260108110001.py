from __future__ import annotations

import streamlit as st
from langchain_core.messages import HumanMessage

from recipes.graph_builder import build_graph
from recipes.schema import RecipeState


# ---------- UI UTILS ----------

def _inject_kitchen_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #ffe9d6 0, #fff7ec 40%, #ffffff 100%);
        }
        .recipe-card {
            background-color: #ffffffcc;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        .step-badge {
            display: inline-block;
            background-color: #ffb347;
            color: #fff;
            border-radius: 999px;
            padding: 0.15rem 0.6rem;
            font-size: 0.8rem;
            margin-right: 0.4rem;
        }
        .shopping-pill {
            display: inline-block;
            background-color: #fff3d9;
            border-radius: 999px;
            padding: 0.2rem 0.7rem;
            margin: 0.15rem;
            font-size: 0.85rem;
        }
        .ust-pill {
            display: inline-block;
            background-color: #e7f0ff;
            border-radius: 999px;
            padding: 0.2rem 0.7rem;
            margin: 0.15rem;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- GRAPH STREAMING ----------

def run_graph_stream(query: str) -> RecipeState:
    graph = build_graph()

    state: RecipeState = {
        "query": query,
        "messages": [HumanMessage(content=query)],
    }
    config = {"configurable": {"thread_id": "streamlit_session"}}

    final_state: RecipeState = state

    col_status, col_meta = st.columns([3, 2])
    with col_status:
        placeholder_status = st.empty()
    with col_meta:
        st.caption("👩‍🍳 Ton compagnon s’occupe de tout : choix des recettes, plan de cuisson, liste de courses et ustensiles.")

    tab_recette, tab_etapes, tab_courses, tab_logs = st.tabs(
        ["🍽️ Recette & plan", "🥣 Étapes détaillées", "🛒 Courses & ustensiles", "🔍 Logs"]
    )

    with tab_recette:
        placeholder_summary = st.empty()
        placeholder_plan = st.empty()
    with tab_etapes:
        placeholder_steps = st.empty()
    with tab_courses:
        placeholder_shopping = st.empty()
        placeholder_ust = st.empty()
    with tab_logs:
        placeholder_log = st.expander("Voir les logs techniques", expanded=False)

    with placeholder_status.container():
        st.info("Le chef réfléchit à la meilleure stratégie pour ton repas...")

    for chunk in graph.stream(state, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            # Logs détaillés
            with placeholder_log:
                st.write(f"**Node exécuté :** `{node}`")
                st.json(update)

            final_state.update(update)

            # ----- Recettes candidates & résumé -----
            if "candidate_recipes" in update:
                with placeholder_summary:
                    st.subheader("🥗 Propositions de recettes")
                    for idx, c in enumerate(update["candidate_recipes"], start=1):
                        title = c.get("title", "Recette")
                        summary = c.get("summary", "")
                        col_l, col_r = st.columns([4, 1])
                        with col_l:
                            st.markdown(
                                f"""
                                <div class="recipe-card">
                                    <h4>{idx}. {title}</h4>
                                    <p style="margin-bottom:0;">{summary}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        with col_r:
                            st.metric("Nbr pers.", c.get("servings", "
