import streamlit as st
import random

from Embedding_Model.retrieval import Retrieval
from Bow_model.bow_retrieval import bow_retrieval
from utils.logger import log_to_sheet

st.set_page_config(
    page_title="ScentMatch",
    layout="wide"
)

st.title("ScentMatch")

if "results_generated" not in st.session_state:
    st.session_state.results_generated = False

query = st.text_input(
    "Describe the perfume you want",
    placeholder="Example: creamy vanilla for winter nights"
)

if st.button("Get Recommendations"):

    with st.spinner("Searching..."):

        semantic_results = Retrieval(query)
        bow_results = bow_retrieval(query)

        if random.random() < 0.5:

            st.session_state.left_results = semantic_results
            st.session_state.right_results = bow_results

            st.session_state.left_model = "semantic"
            st.session_state.right_model = "bow"

        else:

            st.session_state.left_results = bow_results
            st.session_state.right_results = semantic_results

            st.session_state.left_model = "bow"
            st.session_state.right_model = "semantic"

        st.session_state.query = query
        st.session_state.results_generated = True


if st.session_state.results_generated:

    st.divider()

    st.markdown(f"### Query: *{st.session_state.query}*")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Recommendation List A")

        for perfume in st.session_state.left_results:

            with st.container(border=True):
                st.markdown(f"### {perfume['name']}")
                st.markdown(f"**Notes:** {perfume['notes']}")

                if "description" in perfume:
                    st.write(perfume["description"])

    with col2:

        st.subheader("Recommendation List B")

        for perfume in st.session_state.right_results:

            with st.container(border=True):
                st.markdown(f"### {perfume['name']}")
                st.markdown(f"**Notes:** {perfume['notes']}")

                if "description" in perfume:
                    st.write(perfume["description"])

    st.divider()

    st.header("📊 Evaluation")

    participant_name = st.text_input(
        "Name (leave blank if you want to remain anonymous)"
    )

    preferred = st.radio(
        "Which recommendation list was better?",
        ["Recommendation List A", "Recommendation List B", "Tie"]
    )

    relevance_a = st.slider("Recommendation List A Relevance", 1, 5, 3)
    relevance_b = st.slider("Recommendation List B Relevance", 1, 5, 3)

    satisfaction = st.radio(
        "Did either list satisfy your need?",
        ["Yes", "Partially", "No"]
    )

    difficulty = st.slider(
        "How specific was your query?",
        1,
        5,
        3
    )

    comments = st.text_area("Additional comments")

    if st.button("Submit Evaluation"):

        perfumes_a = " | ".join([p["name"] for p in st.session_state.left_results])
        perfumes_b = " | ".join([p["name"] for p in st.session_state.right_results])

        feedback = {
            "name": participant_name,
            "query": st.session_state.query,
            "preferred": preferred,
            "relevance_a": relevance_a,
            "relevance_b": relevance_b,
            "satisfaction": satisfaction,
            "difficulty": difficulty,
            "comments": comments,
            "model_a": st.session_state.left_model,
            "model_b": st.session_state.right_model,
            "perfumes_a": perfumes_a,
            "perfumes_b": perfumes_b,
            "top_a": st.session_state.left_results[0]["name"],
            "top_b": st.session_state.right_results[0]["name"]
        }
        log_to_sheet(
            feedback,
            st.secrets["GOOGLE_SHEET_ID"],
            st.secrets["GOOGLE_SERVICE_ACCOUNT"])
        )

        st.success("Evaluation submitted successfully.")
