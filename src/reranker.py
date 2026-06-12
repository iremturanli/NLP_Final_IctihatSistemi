"""Cross-encoder reranker: hibrit retrieval'ın getirdiği top-N adayı yeniden sırala.

Model: BAAI/bge-reranker-v2-m3 (multilingual, Türkçe destekli, state-of-the-art reranker)

Mimari:
  query --> [hybrid retrieval top-N] --> cross-encoder(query, passage) --> sorted top-K
"""
import torch
from sentence_transformers import CrossEncoder


_RERANKER_CACHE = {}


def get_reranker(model_name="BAAI/bge-reranker-v2-m3", device=None):
    """Reranker modelini cache'le yükle."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (model_name, device)
    if key not in _RERANKER_CACHE:
        print(f"[reranker] Loading {model_name} on {device}")
        _RERANKER_CACHE[key] = CrossEncoder(model_name, device=device, max_length=512)
    return _RERANKER_CACHE[key]


def rerank(query, candidates, reranker=None, top_k=5):
    """Adayları cross-encoder skorlarıyla yeniden sırala.

    candidates: list of dict (each must have 'text' key)
    Döndürür: yeniden sıralanmış list (top_k kadar), her elemana 'rerank_score' eklenir.
    """
    if reranker is None:
        reranker = get_reranker()
    if not candidates:
        return []
    pairs = [(query, c["text"][:1500]) for c in candidates]  # uzun pasajları kısalt
    scores = reranker.predict(pairs, show_progress_bar=False)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


class RerankedRetriever:
    """HybridRetriever üzerine wrap edilen reranker."""

    def __init__(self, hybrid_retriever, reranker_model_name="BAAI/bge-reranker-v2-m3",
                 fetch_k=20, device=None):
        self.hybrid = hybrid_retriever
        self.reranker = get_reranker(reranker_model_name, device)
        self.fetch_k = fetch_k

    def search(self, query, top_k=5):
        candidates = self.hybrid.search(query, top_k=self.fetch_k)
        return rerank(query, candidates, reranker=self.reranker, top_k=top_k)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from embedder import load_model, load_index
    from retriever import HybridRetriever

    index, embeddings, records = load_index("artifacts/index_v1")
    model = load_model("intfloat/multilingual-e5-base")
    hybrid = HybridRetriever(index, embeddings, records, model)
    rr = RerankedRetriever(hybrid, fetch_k=15)

    q = "İhtiyati haciz icra takibi sayılır mı?"
    res = rr.search(q, top_k=3)
    print(f"\nQuery: {q}\n")
    for i, r in enumerate(res, 1):
        print(f"[{i}] rerank={r['rerank_score']:.3f} hybrid={r['score']:.4f} "
              f"| {r['kurul']} E.{r['esas_no']}/K.{r['karar_no']}")
        print(f"    {r['text'][:200]}...\n")
