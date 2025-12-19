
import os
import logging
import nest_asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from src.evaluation import RagasEvaluator

# Apply patch just in case
nest_asyncio.apply()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ragas():
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ No API Key found")
        return

    print("🚀 Initializing Google Gemini for Evaluation...")
    
    # 1. Setup LLM
    eval_llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0
    )
    
    # 2. Setup Embeddings
    eval_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=api_key
    )
    
    # 3. Initialize Evaluator
    evaluator = RagasEvaluator(
        llm=eval_llm, 
        embeddings=eval_embeddings
    )
    
    print("🧪 Running Test Evaluation...")
    
    # 4. Test Data
    query = "What is the capital of France?"
    context = ["Paris is the capital and most populous city of France."]
    response = "The capital of France is Paris."
    
    # 5. Run
    result = evaluator.evaluate_single_response(
        query=query,
        context=context,
        response=response
    )
    
    print("\n✅ Evaluation Result:")
    print(result)

if __name__ == "__main__":
    test_ragas()
