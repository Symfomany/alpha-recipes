from __future__ import annotations

import streamlit as st
from langchain_core.messages import HumanMessage

from recipes.graph_builder import build_graph
from recipes.schema import RecipeState


def run_graph_stream(query: str) -> RecipeState:
    graph = build_graph()

    state: RecipeState = {
        "query": query,
        "messages": [HumanMessage(content=query)],
    }
    config = {"configurable": {"thread_id": "streamlit_session"}}

    final_state: RecipeState = state

    # Zones de sortie
    placeholder_status = st.empty()
    placeholder_summary = st.empty()
    placeholder_plan = st.empty()
    placeholder_steps = st.empty()
    placeholder_shopping = st.empty()
    placeholder_ust = st.empty()
    placeholder_log = st.expander("🔍 Logs détaillés", expanded=False)

    with placeholder_status.container():
        st.info("Le chef réfléchit...")

    for chunk in graph.stream(state, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            # Log brut (optionnel)
            with placeholder_log:
                st.write(f"**Node:** `{node}`")
                st.json(update)

            final_state.update(update)

            # Résumé / recettes candidates
            if "candidate_recipes" in update:
                with placeholder_summary:
                    st.markdown("## 🍽️ Propositions de recettes")
                    for idx, c in enumerate(update["candidate_recipes"], start=1):
                        st.markdown(f"**{idx}. {c.get('title','Recette')}**")
                        if c.get("summary"):
                            st.write(c["summary"])

            # Batch plan
            if "batch_plan" in update:
                with placeholder_plan:
                    st.markdown("## 🧩 Plan Batch Cooking")
                    notes = final_state.get("batch_notes") or ""
                    if notes:
                        st.info(notes)
                    for c in update["batch_plan"]:
                        st.markdown(f"- {c.get('title','Recette')}")

            # Étapes de cuisson
            if "cooking_steps" in update:
                with placeholder_steps:
                    st.markdown("## 🥘 Étapes de cuisson")
                    for line in update["cooking_steps"]:
                        if line.strip():
                            st.markdown(f"- {line}")

            # Liste de courses
            if "shopping_list" in update:
                with placeholder_shopping:
                    st.markdown("## 🛒 Liste de courses")
                    ing = [i for i in update["shopping_list"] if not i.get("is_ustensil")]
                    ust = [i for i in update["shopping_list"] if i.get("is_ustensil")]

                    if ing:
                        st.markdown("**Ingrédients :**")
                        for item in ing:
                            st.write(f"- {item.get('name','')} {item.get('quantity','')}")

                    if ust:
                        st.markdown("**Ustensiles (facultatifs) :**")
                        for item in ust:
                            st.write(f"- {item.get('name','')} {item.get('quantity','')}")

            # Ustensiles détaillés
            if "ustensils_needed" in update:
                with placeholder_ust:
                    st.markdown("## 🔧 Ustensiles recommandés")
                    for u in update["ustensils_needed"]:
                        name = u.get("name") or "Ustensile"
                        kind = u.get("kind") or ""
                        url = (u.get("suggestion_url")
                               or u.get("metadata", {}).get("url"))
                        txt = f"- **{name}** ({kind})"
                        if url:
                            txt += f" – [voir]({url})"
                        st.markdown(txt)

    with placeholder_status.container():
        st.success("Workflow terminé ✅")

    return final_state


def main():
    st.set_page_config(
        page_title="Compagnon de recettes (LangGraph)",
        layout="wide",
        page_icon="🍳",
    )

    st.sidebar.markdown("### Paramètres")
    default_query = "Je veux faire un bon plat pour ce soir"
    query = st.sidebar.text_area(
        "Décris ta situation (ingrédients, personnes, temps, contraintes)",
        value=default_query,
        height=120,
    )
    run = st.sidebar.button("🚀 Lancer le workflow")

    st.title("🍳 Compagnon de recettes – LangGraph + Mistral 3B")

    if run:
        tabs = st.tabs(
            ["Recette & plan", "Étapes", "Courses & ustensiles", "Logs"]
        )
        with tabs[0]:
            # résumé, batch plan seront remplis par run_graph_stream
            pass
        with tabs[1]:
            pass
        with tabs[2]:
            pass
        with tabs[3]:
            st.write("Les logs seront affichés en bas de page.")

        # Exécute le graph (qui remplira les placeholders)
        run_graph_stream(query)
    else:
        st.info("Saisis une demande dans la barre latérale puis clique sur **Lancer le workflow**.")


if __name__ == "__main__":
    main()
