from __future__ import annotations

import streamlit as st
from langchain_core.messages import HumanMessage

from recipes.graph_builder import build_graph
from recipes.schema import RecipeState

"""
Tu es mon chef assistant connecté. Commence par chercher sur le web 3 idées de salades d’été originales et fraîches adaptées aux fortes chaleurs, en tenant compte des tendances récentes. Ensuite, compare ces idées avec les recettes de salades déjà présentes dans ta base locale et dans mes PDFs de recettes, et sélectionne la combinaison la plus cohérente pour un repas du soir pour 6 personnes. Optimise la préparation pour réduire au maximum le temps actif en cuisine (batch cooking des étapes communes : découpe, cuisson, sauces). Termine en générant un tableau de synthèse des ingrédients nécessaires avec colonnes : ingrédient, quantité totale pour 6 personnes, catégorie (légume, fromage, céréale, assaisonnement, autre).

"""



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

    # ---------- LAYOUT HAUT ----------
    col_status, col_meta = st.columns([3, 2])
    with col_status:
        placeholder_status = st.empty()
    with col_meta:
        st.caption(
            "👩‍🍳 Ton compagnon s’occupe de tout : "
            "choix des recettes, plan de cuisson, liste de courses et ustensiles."
        )

    # ---------- TABS ----------
    tab_recette, tab_etapes, tab_courses, tab_logs = st.tabs(
        [
            "🍽️ Recette & plan",
            "🥣 Étapes détaillées",
            "🛒 Courses & ustensiles",
            "🔍 Logs",
        ]
    )

    with tab_recette:
        placeholder_summary = st.empty()
        placeholder_plan = st.empty()
        placeholder_sources = st.empty()
        placeholder_rag_titles = st.empty()
        placeholder_web = st.empty()

    with tab_etapes:
        placeholder_steps = st.empty()

    with tab_courses:
        placeholder_shopping = st.empty()
        placeholder_ust = st.empty()

    with tab_logs:
        placeholder_log = st.expander("Voir les logs techniques", expanded=False)

    # ---------- STATUS INIT ----------
    with placeholder_status.container():
        st.info("Le chef réfléchit à la meilleure stratégie pour ton repas...")

    # ---------- STREAM DU GRAPH ----------
    for chunk in graph.stream(state, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            # Logs bruts
            with placeholder_log:
                st.write(f"**Node exécuté :** `{node}`")
                st.json(update)

            final_state.update(update)

            # ---------- RETRIEVED_DOCS (RAG + WEB) ----------
            if "retrieved_docs" in update:
                docs = update["retrieved_docs"] or []

                # log debug
                with placeholder_log:
                    st.write(
                        f"🔎 retrieved_docs reçu depuis `{node}` : "
                        f"{len(docs)} documents"
                    )

                pdf_filenames = set()
                rag_titles: list[str] = []
                web_results: list[dict] = []

                for d in docs:
                    if not isinstance(d, dict):
                        continue
                    meta = d.get("metadata", {}) or {}

                    # titres de recettes (LOCAL_RECIPES + COOKBOOKS)
                    title = meta.get("title")
                    if title:
                        rag_titles.append(title)

                    # sources PDF cookbook
                    if (
                        meta.get("source") == "cookbook_pdf"
                        and meta.get("filename")
                    ):
                        pdf_filenames.add(meta["filename"])

                    # résultats Tavily (web)
                    if d.get("source") == "web":
                        raw = meta.get("raw", {})
                        if isinstance(raw, list):
                            web_results.extend(raw)
                        elif isinstance(raw, dict):
                            web_results.extend(raw.get("results", []) or [])

                # titres RAG
                if rag_titles:
                    with placeholder_rag_titles:
                        st.markdown("#### 📖 Recettes RAG retrouvées")
                        for t in rag_titles:
                            st.markdown(f"- {t}")

                # PDFs cookbook
                if pdf_filenames:
                    with placeholder_sources:
                        st.markdown("#### 📚 Recettes inspirées de")
                        for fname in sorted(pdf_filenames):
                            st.markdown(f"- `{fname}`")

                # résultats web Tavily
                if web_results:
                    with placeholder_web:
                        st.markdown("#### 🌐 Résultats web (Tavily)")
                        for r in web_results[:3]:
                            title = r.get("title") or "Résultat web"
                            url = r.get("url") or ""
                            snippet = (
                                r.get("content")
                                or r.get("snippet")
                                or ""
                            )
                            st.markdown(f"**{title}**")
                            if url:
                                st.markdown(f"[Voir la source]({url})")
                            if snippet:
                                short = snippet[:300]
                                if len(snippet) > 300:
                                    short += "…"
                                st.caption(short)
                            st.markdown("---")

            # ---------- RECETTES CANDIDATES ----------
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

            # ---------- PLAN BATCH COOKING ----------
            if "batch_plan" in update:
                with placeholder_plan:
                    st.subheader("🧩 Plan de batch cooking")
                    notes = final_state.get("batch_notes") or ""
                    if notes:
                        st.info(notes)
                    for c in update["batch_plan"]:
                        st.markdown(f"- **{c.get('title', 'Recette')}**")

            # ---------- ÉTAPES ----------
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

            # ---------- COURSES ----------
            if "shopping_list" in update:
                with placeholder_shopping:
                    st.subheader("🛒 Liste de courses")
                    ing = [
                        i
                        for i in update["shopping_list"]
                        if not i.get("is_ustensil")
                    ]
                    ust = [
                        i
                        for i in update["shopping_list"]
                        if i.get("is_ustensil")
                    ]

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

            # ---------- USTENSILES RECOMMANDÉS ----------
            if "ustensils_needed" in update:
                with placeholder_ust:
                    st.subheader("🔧 Ustensiles recommandés")
                    for u in update["ustensils_needed"]:
                        name = u.get("name") or "Ustensile"
                        kind = u.get("kind") or ""
                        url = (
                            u.get("suggestion_url")
                            or u.get("metadata", {}).get("url")
                        )
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
            "Rechercher sur le web une recette de  salade d'été, puis regardes les recettes de salades que j'ai deja et optimise la préparation pour 6 personnes ce soir "
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
