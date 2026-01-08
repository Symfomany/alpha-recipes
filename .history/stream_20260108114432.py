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
            background: radial-gradient(circle at top left,
                                        #141824 0,
                                        #050711 40%,
                                        #050711 100%);
        }
        .recipe-card {
            background-color: #141824;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.45);
        }
        .step-badge {
            display: inline-block;
            background-color: #ffb347;
            color: #050711;
            border-radius: 999px;
            padding: 0.15rem 0.6rem;
            font-size: 0.8rem;
            margin-right: 0.4rem;
        }
        .shopping-pill {
            display: inline-block;
            background-color: #22263a;
            border-radius: 999px;
            padding: 0.2rem 0.7rem;
            margin: 0.15rem;
            font-size: 0.85rem;
        }
        .ust-pill {
            display: inline-block;
            background-color: #1b283b;
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
        placeholder_sources = st.empty()
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
            # ---------- SOURCES RAG (PDFs) ----------
            # On suppose que les nœuds RAG remplissent final_state["retrieved_docs"]
            # comme une liste de Documents LangChain avec metadata["source"] / ["filename"].
            if "retrieved_docs" in update:
                docs = update["retrieved_docs"] or []
                pdf_filenames = {
                    d.metadata.get("filename")
                    for d in docs
                    if d.metadata.get("source") == "cookbook_pdf"
                    and d.metadata.get("filename")
                }
                if pdf_filenames:
                    # on garde un seul affichage global des sources cookbooks
                    with placeholder_sources:
                        st.markdown("#### 📚 Recettes inspirées de")
                        for fname in sorted(pdf_filenames):
                            st.markdown(f"- `{fname}`")

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
                            st.metric("Nbr pers.", c.get("servings", "–"))

            # ----- Batch cooking plan -----
            if "batch_plan" in update:
                with placeholder_plan:
                    st.subheader("🧩 Plan de batch cooking")
                    notes = final_state.get("batch_notes") or ""
                    if notes:
                        st.info(notes)
                    for c in update["batch_plan"]:
                        st.markdown(f"- **{c.get('title', 'Recette')}**")

            # ----- Étapes de cuisson -----
            if "cooking_steps" in update:
                with placeholder_steps:
                    st.subheader("🔥 Étapes de cuisson")
                    for i, line in enumerate(update["cooking_steps"], start=1):
                        if line.strip():
                            st.markdown(
                                f"""
                                <div class="recipe-card">
                                    <span class="step-badge">Étape {i}</span>
                                    {line}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            # ----- Liste de courses -----
            if "shopping_list" in update:
                with placeholder_shopping:
                    st.subheader("🛒 Liste de courses")
                    ing = [i for i in update["shopping_list"] if not i.get("is_ustensil")]
                    ust = [i for i in update["shopping_list"] if i.get("is_ustensil")]

                    if ing:
                        st.markdown("**Ingrédients :**")
                        line = ""
                        for item in ing:
                            label = f"{item.get('name','')} {item.get('quantity','')}".strip()
                            line += f'<span class="shopping-pill">{label}</span>'
                        st.markdown(line, unsafe_allow_html=True)

                    if ust:
                        st.markdown("**Ustensiles éventuels :**")
                        line = ""
                        for item in ust:
                            label = f"{item.get('name','')} {item.get('quantity','')}".strip()
                            line += f'<span class="ust-pill">{label}</span>'
                        st.markdown(line, unsafe_allow_html=True)

            # ----- Ustensiles recommandés (avec liens) -----
            if "ustensils_needed" in update:
                with placeholder_ust:
                    st.subheader("🔧 Ustensiles recommandés")
                    for u in update["ustensils_needed"]:
                        name = u.get("name") or "Ustensile"
                        kind = u.get("kind") or ""
                        url = (u.get("suggestion_url")
                               or u.get("metadata", {}).get("url"))
                        base = f"**{name}**"
                        if kind:
                            base += f" – {kind}"
                        if url:
                            base += f" – [voir un exemple]({url})"
                        st.markdown(f"- {base}")

    with placeholder_status.container():
        st.success("Service terminé ✅ Bon appétit !")

    return final_state


# ---------- MAIN APP ----------

def main():
    st.set_page_config(
        page_title="Chef Alpha – Compagnon de cuisine",
        layout="wide",
        page_icon="🍳",
    )

    _inject_kitchen_style()

    with st.sidebar:
        st.title("👨‍🍳 Chef Alpha")
        st.caption("Planifie ton repas, optimise ton temps et génère automatiquement la liste de courses.")

        default_query = (
            "Je veux préparer un poulet rôti convivial pour 4 personnes ce soir, "
            "avec des légumes de saison et un dessert simple."
        )
        query = st.text_area(
            "Décris ta situation (ingrédients, personnes, temps, contraintes alimentaires)",
            value=default_query,
            height=220,
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run = st.button("🚀 Lancer")
        with col_btn2:
            light_example = st.button("🎲 Idée au hasard")

        if light_example and not run:
            st.info("Tu peux demander par exemple : *Repas complet végétarien pour 2 personnes en 30 minutes max.*")

        st.markdown("---")
        st.caption("Propulsé par LangGraph, Mistral et un peu de magie culinaire ✨")

    st.title("🍳 Chef Alpha – Assistant de cuisine intelligent")
    st.markdown(
        "Transforme ce que tu as dans ton frigo en un plan de repas complet : **recettes**, "
        "**organisation du batch cooking**, **liste de courses** et **ustensiles**."
    )

    if run:
        run_graph_stream(query)
    else:
        st.info(
            "Commence par décrire ton contexte dans la barre latérale, puis clique sur **Lancer**.\n\n"
            "Par exemple : *« J’ai des pâtes, des tomates, de la feta et 20 minutes devant moi. »*"
        )


if __name__ == "__main__":
    main()
