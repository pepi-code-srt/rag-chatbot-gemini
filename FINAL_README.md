# 🚀 RAG Chatbot - Final Configuration (Google Native)

**Status:** ✅ Fully Operational  
**Stack:** LangChain • Streamlit • ChromaDB • **Google Gemini (Exclusive)**

This RAG Chatbot has been optimized to be **Cloud Native** using Google's Gemini 1.5/2.0 Flash for both generation and evaluation, and Gemini Text Embeddings for vector search. Dependencies on local LLMs (Ollama) have been completely removed for stability and performance.

---

## 🏗️ Architecture Changes

| Component | Previous (Local) | **Current (Google Native)** | Reason |
|-----------|------------------|-----------------------------|--------|
| **Generation** | Ollama (Llama 3, GPT-OSS) | **Gemini 1.5 Flash / 2.0 Flash** | Removed 12GB+ RAM requirement. Faster. |
| **Embeddings** | Nomic Local (Ollama) | **models/text-embedding-004** | Faster, higher quality, no local install. |
| **Evaluation** | N/A | **Ragas + Gemini 1.5 Flash** | Real-time hallucination detection enabled. |

---

## 🛠️ How to Run

### 1. Prerequisites
Ensure you have a Google API Key in your `.env` file:
```env
GOOGLE_API_KEY="AIzaSy..."
```

### 2. Install Dependencies
(If not already installed)
```bash
pip install -r requirements.txt
pip install langchain-google-genai
```

### 3. Run the App
```bash
streamlit run streamlit_app.py
```

---

## 🔍 Features

### 1. **Strictly Google Backend**
The app is hardcoded to use `models/gemini-3-flash-preview` (or configured model) via the Google API. It will NOT fall back to local models, preventing "Out of Memory" crashes on standard laptops.

### 2. **Real-time Evaluation (Ragas)**
Every answer is double-checked by a secondary AI agent ("The Evaluator"):
-   **Faithfulness:** Does the answer come strictly from the document?
-   **Relevance:** Does it answer the user's question?
-   **Hallucination Alert:** If Faithfulness is low (< 0.8), a warning ⚠️ is displayed.

### 3. **Semantic Chunking**
Documents are intelligently split based on meaning (similarity threshold) rather than arbitrary character counts, ensuring better context for the AI.

---

## 📝 Usage Tips
1.  **Upload First:** You MUST upload documents (PDF/TXT) before chatting.
2.  **Re-upload Validity:** If you restart the app after changing embedding models (which we just did), you must re-upload documents so they are re-indexed with the new Google Embeddings.

*Enjoy your fully optimized, cloud-powered RAG Agent!*
