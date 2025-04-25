# DeepEdgeAI Task - RAG-Based Search Application

This project is a full-stack RAG (Retrieval-Augmented Generation) search application that uses **Serper API** for web search, **BeautifulSoup** for web scraping, **FAISS** for vector storage, **HuggingFace embeddings**, and **Together API** for LLM inference. It features a **Flask backend** and a **Streamlit frontend**.

---

## 🔧 Technologies Used
- Python 3.12
- Flask
- Streamlit
- FAISS
- Together API (instead of OpenAI)
- HuggingFace Transformers
- Serper API
- BeautifulSoup

---

## 🚀 Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/rahuln2002/DeepedgeAI-task.git
   cd DeepedgeAI-task
   ```

2. **Create and activate a virtual environment**
   ```bash
   conda create -p venv python=3.12
   conda activate venv/
   ```

3. **Create a ```.env``` file and add your API keys**
   ```bash
   SERPER_API_KEY=your_serper_api_key
   TOGETHER_API_KEY=your_together_api_key
   ```

4. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask backend**
   ```bash
   python ./flask_app/app.py
   ```

4. **Run the Streamlit frontend**
   ```bash
   streamlit run ./streamlit_app/app.py --server.port 8502
   ```

---

## ✅ Working

- Takes user query as input from the Streamlit UI
- Uses Serper API to fetch search results
- Scrapes the contents of top results using BeautifulSoup
- Converts text into vector embeddings using HuggingFace
- Stores and queries vectors with FAISS
- Sends context + query to Together API for LLM-based response generation
- Displays the generated response in the UI

---

## 📌 Notes

- This app uses **Together API** instead of OpenAI for LLM inference
- Make sure your `.env` file contains valid API keys