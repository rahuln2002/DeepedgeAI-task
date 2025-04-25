import streamlit as st
import requests

st.title("LLM-based RAG Search")

# Initialize session state for chat history
if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []

# Reset button to clear session state
if st.button("🔄 Reset Chat"):
    # Clear local Streamlit state
    st.session_state.qa_pairs = []

    # Notify backend to clear its cache
    try:
        requests.post("http://localhost:8501/query", json={"reset": True})
    except Exception as e:
        st.warning(f"Could not notify backend to reset: {e}")
        
    st.rerun()

# Input for user query
query = st.text_input("Enter your query:", key=f"query_{len(st.session_state.qa_pairs)}")

# Search button
if st.button("Search", key=f"search_{len(st.session_state.qa_pairs)}") and query:
    st.write("Searching...")

    try:
        # Call Flask API
        response = requests.post(
            "http://localhost:8501/query",
            json={"query": query}
        )

        if response.status_code == 200:
            answer = response.json().get('answer', "No answer received.")
        else:
            answer = f"Error: {response.status_code}"

    except Exception as e:
        answer = f"Request failed: {e}"

    # Store query-answer pair
    st.session_state.qa_pairs.insert(0, (query, answer))

# Display all past query-answer pairs
for i, (q, a) in enumerate(st.session_state.qa_pairs):
    st.markdown(f"Q: {q}")
    st.markdown(f"A: {a}")
    st.markdown("---")