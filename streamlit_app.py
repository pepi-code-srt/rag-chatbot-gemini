import streamlit as st
import os
import tempfile
import nest_asyncio
from pathlib import Path

# Apply nest_asyncio to fix Ragas/Streamlit loop issues
nest_asyncio.apply()

# Import RAG components
from config import (
    CHUNK_CONFIG, RETRIEVAL_CONFIG, GENERATION_CONFIG, 
    EVALUATION_CONFIG, CHROMADB_CONFIG, DOCUMENTS_DIR
)
from src.chunking import chunk_documents
from src.embedding import get_embeddings
from src.retrieval import HybridRetriever
from src.generation import GeminiGenerator
from src.evaluation import RagasEvaluator
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangChainDocument

# Page Config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 RAG Chatbot Assistant")

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key Management
    api_key = st.text_input("Google API Key", type="password", help="Enter your Gemini API Key")
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            st.success("API Key found in environment")
    
    st.divider()
    
    # Ragas Toggle
    st.toggle("Enable AI Evaluation (Ragas)", value=False, key="enable_ragas", help="Enable to check for hallucinations. SLOW! Adds 5-10s per message.")
    
    if not api_key:
        st.warning("Please enter your Google API Key to continue.")
        st.stop()
        
    # File Upload
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", 
        type=["txt", "pdf", "md"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if not uploaded_files:
            st.error("Please upload at least one file.")
        else:
            with st.spinner("Processing documents..."):
                # Save and chunk files
                processed_count = 0
                for uploaded_file in uploaded_files:
                    # Save temp file
                    file_path = DOCUMENTS_DIR / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Read content
                    try:
                        text_content = ""
                        if uploaded_file.name.lower().endswith(".pdf"):
                            import pypdf
                            pdf_reader = pypdf.PdfReader(uploaded_file)
                            for page in pdf_reader.pages:
                                text_content += page.extract_text() + "\n"
                        else:
                            # Text/MD files
                            text_content = uploaded_file.getvalue().decode("utf-8")
                        
                        doc = LangChainDocument(
                            page_content=text_content,
                            metadata={"source": uploaded_file.name}
                        )
                        
                        # Chunk
                        chunks = chunk_documents([doc], **CHUNK_CONFIG)
                        
                        # Add to Vectorstore
                        st.session_state.vectorstore.add_documents(chunks)
                        processed_count += 1
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                
                if processed_count > 0:
                    st.success(f"Successfully processed {processed_count} documents!")
                    # Force retriever refresh
                    st.session_state.retriever = None

# Initialize RAG System
@st.cache_resource
def get_chroma_client():
    import chromadb
    return chromadb.PersistentClient(path=CHROMADB_CONFIG["persist_directory"])

if "vectorstore" not in st.session_state:
    embeddings = get_embeddings(**CHUNK_CONFIG)
    client = get_chroma_client()
    st.session_state.vectorstore = Chroma(
        client=client,
        collection_name=CHROMADB_CONFIG["collection_name"],
        embedding_function=embeddings,
    )

from src.generation import GeminiGenerator

# Force re-initialization if backend switched or model changed
if "generator" in st.session_state:
    current_gen = st.session_state.generator
    backend = "google" # Hardcoded as we removed Ollama
    
    if not isinstance(current_gen, GeminiGenerator):
        del st.session_state.generator
        st.info(f"Switching to Google Generator...")
    elif hasattr(current_gen, 'model_name') and current_gen.model_name != GENERATION_CONFIG.get("model_name"):
        del st.session_state.generator
        st.info(f"Switching model to {GENERATION_CONFIG.get('model_name')}...")

if "generator" not in st.session_state:
    st.session_state.generator = GeminiGenerator(
        api_key=api_key,
        **{k: v for k, v in GENERATION_CONFIG.items() if k != "backend" and k != "fallback_backend"}
    )

# Dynamic Retriever (needs to be updated when docs change)
if "retriever" not in st.session_state or st.session_state.retriever is None:
    all_docs = st.session_state.vectorstore.get()
    if all_docs and all_docs.get("documents"):
        docs = [
            LangChainDocument(page_content=c, metadata=m or {})
            for c, m in zip(all_docs["documents"], all_docs["metadatas"])
        ]
        st.session_state.retriever = HybridRetriever(
            vectorstore=st.session_state.vectorstore,
            documents=docs,
            **RETRIEVAL_CONFIG
        )
    else:
        st.session_state.retriever = None

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if not st.session_state.retriever:
            response = "⚠️ Please upload and process documents first."
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            with st.spinner("Thinking..."):
                # Retrieve
                try:
                    results = st.session_state.retriever.retrieve(
                        prompt, 
                        top_k=RETRIEVAL_CONFIG["top_k"],
                        return_scores=True
                    )
                except Exception as e:
                    if "No connection could be made" in str(e) or "10061" in str(e):
                        st.error("🚨 Error: Could not connect to local Ollama server for embeddings.")
                        st.info("Please ensure Ollama is running (`ollama serve`). Retrieval depends on local embeddings.")
                        st.stop()
                    else:
                        raise e
                
                if not results:
                    response = "I couldn't find any relevant information in the documents."
                    message_placeholder.markdown(response)
                else:
                    # Generate with Fallback Logic
                    # Generate (Strictly Google API)
                    gen_result = st.session_state.generator.generate(
                        prompt, 
                        results, 
                        include_reasoning=False
                    )

                    if gen_result.get("status") == "error":
                        st.error(f"Generation failed: {gen_result.get('error')}")
                        # Don't throw, just show error
                        response = "⚠️ Error generating response."
                    else:
                        response = gen_result["response"]
                    

                    # Display Sources
                    source_text = "\n\n**Sources:**\n"
                    seen_sources = set()
                    
                    # Store context text for evaluation
                    context_texts = []
                    
                    for doc, score in results:
                        source_name = doc.metadata.get('source', 'Unknown')
                        if source_name not in seen_sources:
                            source_text += f"- {source_name} (Relevance: {score:.2f})\n"
                            seen_sources.add(source_name)
                        context_texts.append(doc.page_content)
                    
                    full_response = response + source_text
                    message_placeholder.markdown(full_response)
                    
                    # ---------------------------------------------------------
                    # RAGAS EVALUATION (Optional)
                    # ---------------------------------------------------------
                    if st.session_state.get("enable_ragas", False):
                        try:
                            with st.status("🕵️ Evaluator is validating response...", expanded=False) as status:
                                from langchain_google_genai import ChatGoogleGenerativeAI
                                from src.evaluation import RagasEvaluator
                                
                                # Init Eval LLM (Use the same API KEY)
                                # Use the same model as generation to ensure it works
                                eval_llm = ChatGoogleGenerativeAI(
                                    model=GENERATION_CONFIG["model_name"], 
                                    google_api_key=api_key,
                                    temperature=0
                                )
                                
                                # Init Evaluator
                                # Use the vectorstore's embedding function
                                evaluator = RagasEvaluator(
                                    llm=eval_llm, 
                                    embeddings=st.session_state.vectorstore._embedding_function,
                                    faithfulness_threshold=0.8,
                                    relevance_threshold=0.75
                                )
                                
                                st.write("Running Faithfulness and Relevance checks...")
                                # Evaluate
                                eval_result = evaluator.evaluate_single_response(
                                    query=prompt,
                                    context=context_texts,
                                    response=response
                                )
                                
                                scores = eval_result.get("scores", {})
                                faithfulness = scores.get("faithfulness", 0.0)
                                relevance = scores.get("answer_relevancy", 0.0)
                                
                                st.write(f"Faithfulness: {faithfulness:.2f}")
                                st.write(f"Relevance: {relevance:.2f}")
                                
                                if eval_result.get("hallucination_detected"):
                                    status.update(label="⚠️ Possible Hallucination Detected!", state="error")
                                    st.error(f"Low Faithfulness Score ({faithfulness:.2f}). The model may be hallucinating.")
                                    full_response += f"\n\n**⚠️ Evaluation Alert:** Possible Hallucination detected (Faithfulness: {faithfulness:.2f}). Verified context is low."
                                    message_placeholder.markdown(full_response)
                                else:
                                    status.update(label="✅ Response Verified", state="complete")
                                    full_response += f"\n\n*Verified (Faithfulness: {faithfulness:.2f}, Relevance: {relevance:.2f})*"
                                    message_placeholder.markdown(full_response)

                        except Exception as e:
                            print(f"Evaluation failed: {e}")
                            # Don't crash the chat if eval fails
                            st.warning(f"Evaluation skipped: {e}")

                    valid_response_msg = {"role": "assistant", "content": full_response}
                    if "model" in gen_result:
                        valid_response_msg["model"] = gen_result["model"]
                        
                    st.session_state.messages.append(valid_response_msg)
