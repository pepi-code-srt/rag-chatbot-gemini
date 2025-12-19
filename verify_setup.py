import sys
import os

# Emulate environment without keys to test fallback
if "NOMIC_API_KEY" in os.environ:
    del os.environ["NOMIC_API_KEY"]

try:
    print("Testing imports...")
    import src.chunking
    import src.embedding
    import src.retrieval
    import src.generation
    import src.evaluation
    print("✅ All modules imported successfully.")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

try:
    print("\nTesting Embedding Fallback (No Key)...")
    import src.embedding
    
    # Force key to be empty to test fallback
    original_key = src.embedding.NOMIC_API_KEY
    src.embedding.NOMIC_API_KEY = ""
    
    embeddings = src.embedding.get_embeddings()
    if isinstance(embeddings, src.embedding.LocalNomicEmbeddings):
        print("✅ Correctly fell back to LocalNomicEmbeddings")
    else:
        print(f"❌ Incorrect embedding class: {type(embeddings)}")
    
    # Restore key
    src.embedding.NOMIC_API_KEY = original_key

except Exception as e:
    print(f"❌ Fallback test failed: {e}")
    sys.exit(1)
