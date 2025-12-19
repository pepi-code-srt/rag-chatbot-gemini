"""
FastAPI Application for RAG Chatbot
Main entry point for the web service
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json
from pathlib import Path

from config import (
    FASTAPI_CONFIG, CHROMADB_CONFIG, CHUNK_CONFIG, 
    RETRIEVAL_CONFIG, GENERATION_CONFIG, EVALUATION_CONFIG,
    GOOGLE_API_KEY, DOCUMENTS_DIR, CHROMADB_DIR
)
from src.chunking import chunk_documents, SemanticChunker
from src.embedding import get_embeddings
from src.retrieval import HybridRetriever
from src.generation import GeminiGenerator
from src.evaluation import RagasEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(**FASTAPI_CONFIG)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

rag_system = None  # Will be initialized on startup


class RAGSystem:
    """Main RAG system controller"""
    
    def __init__(self):
        self.vectorstore = None
        self.documents = None
        self.retriever = None
        self.generator = None
        self.evaluator = None
        self.embeddings = None
    
    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing RAG system...")
        
        # Initialize embeddings
        self.embeddings = get_embeddings(**CHUNK_CONFIG)
        logger.info("✅ Embeddings initialized")
        
        # Initialize Chroma
        import chromadb
        from langchain_chroma import Chroma
        
        chroma_client = chromadb.PersistentClient(
            path=CHROMADB_CONFIG["persist_directory"]
        )
        
        self.vectorstore = Chroma(
            client=chroma_client,
            collection_name=CHROMADB_CONFIG["collection_name"],
            embedding_function=self.embeddings,
        )
        logger.info("✅ ChromaDB initialized")
        
        # Initialize generator
        self.generator = GeminiGenerator(
            api_key=GOOGLE_API_KEY,
            **GENERATION_CONFIG
        )
        logger.info("✅ Gemini generator initialized")
        
        # Initialize evaluator
        self.evaluator = RagasEvaluator(
            faithfulness_threshold=EVALUATION_CONFIG["faithfulness_threshold"],
            relevance_threshold=EVALUATION_CONFIG["relevance_threshold"],
        )
        logger.info("✅ Ragas evaluator initialized")
        
        logger.info("✅ RAG system initialized successfully")


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class DocumentUpload(BaseModel):
    """Document upload request"""
    filename: str
    content: str
    source_type: str = "text"  # "text" or "pdf"


class ChatRequest(BaseModel):
    """Chat query request"""
    query: str
    top_k: Optional[int] = 5
    include_sources: bool = True
    include_scores: bool = True
    evaluate: bool = True


class ChatResponse(BaseModel):
    """Chat response"""
    query: str
    response: str
    sources: Optional[List[Dict[str, Any]]]
    scores: Optional[List[float]]
    evaluation: Optional[Dict[str, Any]]
    status: str


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    rag_system = RAGSystem()
    rag_system.initialize()
    logger.info("✅ RAG system started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down RAG system...")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document"""
    
    try:
        logger.info(f"Uploading document: {file.filename}")
        
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8', errors='ignore')
        
        # Save to disk
        doc_path = DOCUMENTS_DIR / file.filename
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(content_str)
        
        # Process document
        from langchain_core.documents import Document
        doc = Document(
            page_content=content_str,
            metadata={
                "source": file.filename,
                "file_size": len(content_str),
            }
        )
        
        # Chunk document
        chunked_docs = chunk_documents([doc], **CHUNK_CONFIG)
        logger.info(f"Created {len(chunked_docs)} chunks")
        
        # Add to vectorstore
        rag_system.vectorstore.add_documents(chunked_docs)
        
        return {
            "message": "Document uploaded successfully",
            "filename": file.filename,
            "chunks_created": len(chunked_docs),
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint"""
    
    try:
        logger.info(f"Processing query: {request.query[:50]}...")
        
        # Initialize retriever if not done
        if rag_system.retriever is None:
            all_docs = rag_system.vectorstore.get()
            if all_docs and all_docs.get("documents"):
                from langchain_core.documents import Document
                docs = [
                    Document(
                        page_content=content,
                        metadata=meta or {}
                    )
                    for content, meta in zip(
                        all_docs.get("documents", []),
                        all_docs.get("metadatas", [])
                    )
                ]
                
                rag_system.retriever = HybridRetriever(
                    vectorstore=rag_system.vectorstore,
                    documents=docs,
                    **RETRIEVAL_CONFIG
                )
        
        if not rag_system.retriever:
            raise HTTPException(
                status_code=400,
                detail="No documents indexed. Please upload documents first."
            )
        
        # Retrieve documents
        results = rag_system.retriever.retrieve(
            request.query,
            top_k=request.top_k,
            return_scores=True
        )
        
        if not results:
            return ChatResponse(
                query=request.query,
                response="No relevant documents found.",
                sources=None,
                scores=None,
                evaluation=None,
                status="no_results"
            )
        
        # Generate response
        gen_result = rag_system.generator.generate(
            request.query,
            results,
            include_reasoning=False
        )
        
        if gen_result["status"] == "error":
            raise HTTPException(status_code=500, detail=gen_result["error"])
        
        # Evaluate (optional)
        evaluation_result = None
        if request.evaluate:
            context = [doc.page_content for doc, _ in results]
            evaluation_result = rag_system.evaluator.evaluate_single_response(
                query=request.query,
                context=context,
                response=gen_result["response"]
            )
        
        # Prepare response
        sources = None
        scores = None
        
        if request.include_sources:
            sources = [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk_index": doc.metadata.get("chunk_index", 0)
                }
                for doc, _ in results
            ]
        
        if request.include_scores:
            scores = [score for _, score in results]
        
        return ChatResponse(
            query=request.query,
            response=gen_result["response"],
            sources=sources,
            scores=scores,
            evaluation=evaluation_result,
            status="success"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_system_initialized": rag_system is not None,
        "vectorstore_ready": rag_system and rag_system.vectorstore is not None
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    
    if not rag_system or not rag_system.vectorstore:
        raise HTTPException(status_code=400, detail="RAG system not initialized")
    
    try:
        collection = rag_system.vectorstore._collection
        stats = collection.count()
        
        return {
            "total_documents": stats,
            "vectorstore": "ChromaDB",
            "retrieval_method": "Hybrid (Vector + BM25)",
            "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
            "generation_model": "gemini-pro"
        }
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Advanced RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload a document",
            "chat": "POST /chat - Send a query",
            "health": "GET /health - Health check",
            "stats": "GET /stats - System statistics"
        },
        "docs": "/docs"
    }


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=FASTAPI_CONFIG["host"],
        port=FASTAPI_CONFIG["port"],
        reload=FASTAPI_CONFIG["reload"],
        workers=FASTAPI_CONFIG["workers"]
    )
