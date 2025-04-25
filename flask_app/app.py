from flask import Flask, request, jsonify
from utils import search_articles, fetch_article_content, concatenate_content, generate_answer, reset_history_store

# Load environment variables from .env file

app = Flask(__name__)

session_cache = {}

@app.route('/query', methods=['POST'])
def query():
    """
    Handles the POST request to '/query'. Extracts the query from the request,
    processes it through the search, concatenate, and generate functions,
    and returns the generated answer.
    """
    # get the data/query from streamlit app
    data = request.get_json()
    query = data.get("query", "")
    reset = data.get("reset", False)

    print("Received query: ", query)

    if reset:
        session_cache.pop("rag_content", None)
        reset_history_store()
        return jsonify({"answer": "Session reset successful."})
    
    if "rag_content" not in session_cache:
        # Step 1: Search and scrape articles based on the query
        print("Step 1: searching articles")
        urls = search_articles(query = query)

        # Step 2: Concatenate content from the scraped articles
        print("Step 2: concatenating content")
        articles = fetch_article_content(urls = urls)
        content = concatenate_content(articles = articles)

        session_cache["rag_content"] = content
    else:
        print("Using cached content")
        content = session_cache["rag_content"]

    # Step 3: Generate an answer using the LLM
    print("Step 3: generating answer")
    response = generate_answer(
        content = content,
        query = query
    )

    # return the jsonified text back to streamlit
    return jsonify({"answer" : response})

if __name__ == '__main__':
    app.run(host='localhost', port=8501)