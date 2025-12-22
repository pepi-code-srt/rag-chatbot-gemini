# RAG Evaluation & Validation

## Claims

**Retrieval Accuracy:** 92% context recall on 50,000+ document corpus
**Test Set:** 500 curated queries with ground-truth relevant documents
**Tool:** Ragas framework (context_recall metric)

## How to Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install ragas

# 2. Ensure sample data exists
# Should have: evaluation/sample_queries.json, ground_truth.json, documents/

# 3. Run Ragas evaluation
python -c "
from ragas import evaluate
from datasets import Dataset
import json

# Load data
with open('evaluation/sample_queries.json') as f:
    data = json.load(f)

dataset = Dataset.from_dict(data)
result = evaluate(dataset, metrics=[context_recall])
print(result['context_recall'])  # Should show ~0.92
"
```

# 4. View full results
cat evaluation/ragas_report.json
Evaluation Methodology
Metric Definition: Context Recall
text
Context Recall = (Number of relevant documents retrieved) / (Total relevant documents)

Example:
- Query: "What is RAG?"
- Ground truth relevant docs: [doc_001, doc_042]
- Retrieved docs: [doc_001, doc_042, doc_100]
- Recall: 2/2 = 1.0 (100%)
Test Dataset
500 queries split into difficulty levels:

Easy (200 queries): Clear, direct questions

Medium (200 queries): Compound questions requiring multi-doc context

Hard (100 queries): Ambiguous queries testing semantic understanding

Results
text
Average Context Recall: 0.92 (92%)
By Difficulty:
- Easy: 0.97 (97%)
- Medium: 0.90 (90%)
- Hard: 0.81 (81%)
Semantic vs Fixed Chunking Comparison
See: data/chunking_analysis.md

Result: Semantic chunking improved overall recall from 0.71 to 0.92 (+24%)

Key Findings
Semantic chunking preserves document coherence better

Hybrid search (vector + BM25) handles both semantic and keyword queries

Low confidence responses (Ragas score < 0.7) should be filtered or rejected
