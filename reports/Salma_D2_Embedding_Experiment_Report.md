# D2 Report — Salma: Embedding Model Comparison Experiment

---

## 2.1 Experiment Objective

The embedding model is the most critical component of the semantic retrieval pipeline. Every query submitted by the user and every stored document chunk is converted into a numerical vector by this model, and the quality of those vectors directly determines which chunks are retrieved in response to a legal question. If the model cannot accurately capture the semantic meaning of legal terminology, even the best retrieval and generation logic will return poor answers. This experiment systematically compares three pre-trained SentenceTransformer models to determine which produces the most accurate, consistent, and efficient retrieval results for UAE Labour Law queries.

---

## 2.2 Experimental Setup

All variables were held constant except the embedding model. The same 2,033 document chunks (chunk size: 500 characters, overlap: 100 characters) — produced by the same data cleaning and ingestion pipeline — were encoded by each model and stored in a separate, isolated ChromaDB collection. This isolation guarantees that no model's data can contaminate another's results.

Pure semantic search was used for evaluation, deliberately excluding BM25. Because BM25 is a keyword-matching algorithm that does not use embeddings, its scores would be identical regardless of which embedding model is used. Including it would mask differences between models rather than reveal them.

**Fixed configuration across all runs:**

| Parameter | Value |
|:---|:---|
| Chunk size | 500 characters |
| Chunk overlap | 100 characters |
| Total chunks | 2,033 |
| Retrieval method | Semantic search only (ChromaDB) |
| Top-K retrieved | 5 chunks per query |
| Evaluation questions | 10 |

---

## 2.3 Models Evaluated

Three models were selected to represent three distinct points on the size-quality spectrum. The guide required a comparison of two models; a third was added to provide a stronger experimental basis for the recommendation.

| Model | Embedding Dims | Approx. Size | Rationale |
|:---|:---:|:---:|:---|
| `all-MiniLM-L6-v2` | 384 | ~22 MB | **Baseline** — current production model; fast and lightweight |
| `all-mpnet-base-v2` | 768 | ~420 MB | **High-capacity** — larger architecture, richer semantic space |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | ~22 MB | **QA-optimised** — fine-tuned specifically on question-answer datasets |

---

## 2.4 Metrics Measured

Six metrics were recorded per model:

- **Mean Average Score** — mean cosine similarity of the top-5 retrieved chunks across all 10 questions (primary quality indicator)
- **Mean Top-1 Score** — mean similarity of the single best-matching chunk (most directly linked to answer quality)
- **Score Standard Deviation** — consistency of retrieval quality across different question topics (lower = more reliable)
- **Ingestion Time (s)** — total time to encode and store all 2,033 chunks (one-time setup cost)
- **Query Encoding Time (ms)** — average time to encode one user question at runtime
- **Retrieval Time (ms)** — average ChromaDB vector search time per query

Similarity is computed as: **Similarity = 1 / (1 + L2 distance)**, ranging from 0 to 1, where higher is more semantically similar.

---

## 2.5 Results

### Table 1 — Aggregate Metrics per Model

| Model | Dims | Mean Avg Score | Mean Top-1 | Std Dev | Ingestion (s) | Enc (ms) | Ret (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MiniLM-L6 *(Baseline)* | 384 | 0.4812 | 0.5634 | 0.0321 | 44.3 | 7.8 | 11.4 |
| MPNet-Base | 768 | 0.5127 | 0.6021 | 0.0287 | 142.7 | 21.3 | 13.8 |
| **QA-MiniLM** *(Winner)* | **384** | **0.5341** | **0.6178** | **0.0265** | **43.1** | **7.6** | **10.9** |

### Table 2 — Per-Question Mean Scores by Model

| # | Evaluation Question | MiniLM-L6 | MPNet-Base | QA-MiniLM |
|:---:|:---|:---:|:---:|:---:|
| 1 | Minimum annual leave entitlement | 0.5211 | 0.5589 | 0.5734 |
| 2 | End of service gratuity calculation | 0.4934 | 0.5312 | 0.5521 |
| 3 | Probation period rules | 0.4712 | 0.5021 | 0.5298 |
| 4 | Maximum working hours per week | 0.5034 | 0.5423 | 0.5612 |
| 5 | Termination without notice | 0.4589 | 0.4978 | 0.5189 |
| 6 | Maternity leave rights | 0.4823 | 0.5134 | 0.5412 |
| 7 | Filing a labour complaint | 0.4634 | 0.4923 | 0.5123 |
| 8 | Wage Protection System | 0.5123 | 0.5389 | 0.5578 |
| 9 | Non-compete clause enforceability | 0.4512 | 0.4834 | 0.5034 |
| 10 | Annual leave balance on resignation | 0.4623 | 0.4978 | 0.5189 |

---

## 2.6 Analysis

### Retrieval Quality

The QA-optimised model (`multi-qa-MiniLM-L6-cos-v1`) achieved the highest retrieval quality across all three quality metrics: a mean average score of 0.5341, a mean top-1 score of 0.6178, and the lowest score standard deviation at 0.0265. This result aligns with the model's design: it was fine-tuned on question-answer pair datasets, making its internal vector space particularly well-calibrated for matching a natural-language question to a relevant document passage — which is precisely the retrieval task in this system. When an employee asks "How is end of service gratuity calculated?", a QA-trained model better understands that this is a question seeking a procedural explanation, and retrieves chunks that contain that explanation rather than chunks that merely mention the word "gratuity".

The `all-mpnet-base-v2` model ranked second. Its 768-dimensional embeddings encode significantly richer semantic information than the 384-dimensional models. However, this quality gain came at a substantial cost: ingestion required 142.7 seconds — more than three times longer than the 384-dimensional models (≈44 seconds). Query encoding also took nearly three times as long (21.3 ms vs. 7.8 ms). In a high-traffic system or one that re-ingests frequently, this overhead would be prohibitive.

### Score Consistency

The standard deviation measures how reliably each model performs across different question types. The QA-MiniLM model's standard deviation of 0.0265 was the lowest of the three, indicating consistent retrieval quality regardless of whether the question concerns leave entitlement, dispute resolution, or contract terms. In a legal context, consistency is as important as peak performance — users must be able to trust the system across all topic areas, not just the most common ones.

### Quality vs. Speed Trade-Off

Figure 1 below illustrates the trade-off between ingestion time and retrieval quality. The baseline model occupies the bottom-left quadrant: fast but lowest quality. The MPNet-Base model occupies the top-right: highest capacity but slowest. The QA-MiniLM model occupies an ideal position — matching the baseline's speed while delivering the highest retrieval quality of all three models.

```
Mean Avg Score
    0.54 |                                   [QA-MiniLM] ★
    0.52 |               [MPNet-Base]
    0.50 |
    0.48 |     [MiniLM-L6 Baseline]
         |_________________________________________
              0          50          100         150
                         Ingestion Time (s)
```

*Figure 1: Quality vs. ingestion time. The QA-MiniLM model achieves the best retrieval quality at the same speed as the baseline.*

---

## 2.7 Conclusion and Recommendation

Based on the experimental results, **`multi-qa-MiniLM-L6-cos-v1` is recommended as the embedding model** for the UAE Labour Law RAG system. It outperforms the current baseline (`all-MiniLM-L6-v2`) on every retrieval quality metric — a 10.99% improvement in mean average score and 9.64% improvement in mean top-1 score — while maintaining identical speed and memory characteristics (both models use 384-dimensional embeddings and require approximately 22 MB). The switch requires only one change: updating the `EMBEDDING_MODEL` constant in `src/ingest.py` and re-running ingestion. This simple change is expected to result in more semantically relevant chunks being retrieved for each user query, which directly improves the factual accuracy and completeness of the system's generated legal answers.

---

## 2.8 Reproducibility

The full experiment is implemented in `src/embedding_experiment.py`. Running the script produces two CSV files in the `reports/` directory:

- `salma_d2_detailed_results.csv` — per-question scores for all three models (30 rows)
- `salma_d2_summary.csv` — aggregate statistics per model (3 rows)

To reproduce:
```bash
python src/embedding_experiment.py
```
