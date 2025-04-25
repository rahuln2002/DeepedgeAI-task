import os
from dotenv import load_dotenv
import http.client
import json
import certifi
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_together import Together
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load API keys from environment variables
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")


def search_articles(query):
    """
    Searches for articles related to the query using Serper API.
    Returns a list of dictionaries containing article URLs, headings, and text.
    """
    articles = []
    # implement the search logic - retrieves articles

    conn = http.client.HTTPSConnection("google.serper.dev")

    payload = json.dumps({
    "q": query
    })

    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    conn.request("POST", "/search", payload, headers)

    res = conn.getresponse()

    data = res.read().decode("utf-8")

    conn.close()

    results = json.loads(data).get("organic", [])
    for item in results[:5]:
        articles.append(item['link'])

    return articles


def fetch_article_content(urls):
    """
    Fetches the article content, extracting headings and text.
    """
    content = []
    # implementation of fetching headings and content from the articles
    for url in urls:
        response = requests.get(url, verify=certifi.where(), timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3']):
            text = h.get_text(strip=True)
            if text:
                headings.append(text)
        
        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if text:
                paragraphs.append(text)

        content.append([headings[:5], paragraphs[:5]])

    return content


def concatenate_content(articles):
    """
    Concatenates the content of the provided articles into a single string.
    """
    full_text = []
    # formatting + concatenation of the string is implemented here

    for article in articles:
        article_text = " ".join(article[0]) + " " + " ".join(article[1])
        full_text.append(article_text)

    return full_text

store = {}

def reset_history_store():
    store.clear()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def generate_answer(content, query):
    """
    Generates an answer from the concatenated content using GPT-4.
    The content and the user's query are used to generate a contextual answer.
    """
    # Create the prompt based on the content and the query
    documents = [Document(page_content=text) for text in content]
    db = FAISS.from_documents(documents, embedding=embedding_model)

    # implement together call logic and get back the response
    llm = Together(
        model="meta-llama/Llama-3-8b-chat-hf",
        temperature=0.3,
        max_tokens=256
    )

    retriever = db.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question "
        "If you don't know the answer, say that you don't know."
        "Do not use speaker tags in your response."
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    retriever_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history,"
        "formulate a standalone question which can be understood without the chat history."
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt  = ChatPromptTemplate.from_messages(
        [
            ("system", retriever_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    result = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": "abc123"}}
    )

    response = result.get("answer") if isinstance(result, dict) else str(result)
    response = response.strip()

    return response