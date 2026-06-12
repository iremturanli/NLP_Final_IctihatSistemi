"""Hybrid retrieval: BM25 (lexical) + dense (semantic) + RRF füzyonu."""
import re
import numpy as np
from rank_bm25 import BM25Okapi


TURKISH_STOPWORDS = {
    "ve", "ile", "bir", "bu", "için", "olarak", "olan", "olup", "ancak",
    "ki", "ya", "veya", "de", "da", "mi", "mı", "mu", "mü", "ise",
    "her", "şu", "o", "biz", "siz", "ben", "sen", "onlar", "ne",
    "nasıl", "neden", "kim", "hangi", "nerede", "ne zaman", "çok",
    "az", "gibi", "kadar", "göre", "dolayı", "rağmen", "sonra", "önce",
    "şey", "şeyi", "şeyin", "şeyler",
}


def tokenize_tr(text):
    """Basit Türkçe tokenizasyon: küçük harfe çevir, kelime karakterleri."""
    text = text.lower()
    text = text.replace("ı", "i")  # BM25 için normalize
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    return [t for t in tokens if len(t) > 2 and t not in TURKISH_STOPWORDS]


class HybridRetriever:
    def __init__(self, faiss_index, embeddings, records, embedding_model,
                 alpha_dense=0.6, rrf_k=60):
        """
        faiss_index: FAISS index (inner product, normalized)
        embeddings:  numpy array (N, D)
        records:     list of dict (her chunk için metadata + text)
        embedding_model: sentence_transformers model
        alpha_dense: dense skorun ağırlığı (RRF kullanılırsa devre dışı)
        rrf_k: RRF formülündeki k sabiti
        """
        self.index = faiss_index
        self.embeddings = embeddings
        self.records = records
        self.model = embedding_model
        self.alpha_dense = alpha_dense
        self.rrf_k = rrf_k

        # BM25 indeksi
        print("[retriever] Building BM25 index...")
        self.bm25_corpus = [tokenize_tr(r["text"]) for r in records]
        self.bm25 = BM25Okapi(self.bm25_corpus)

    def _dense_search(self, query, top_k):
        from sentence_transformers import util as st_util
        formatted = f"query: {query}"
        q_emb = self.model.encode(
            [formatted], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        scores, idx = self.index.search(q_emb, top_k)
        return list(zip(idx[0].tolist(), scores[0].tolist()))

    def _bm25_search(self, query, top_k):
        q_tokens = tokenize_tr(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        top = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def search(self, query, top_k=10, fetch_k=30):
        """RRF füzyonu ile hybrid arama. Aynı kararın birden çok chunk'ı varsa en iyiyi tutar.

        Her sonuca eklenir:
          - score      : RRF füzyon skoru (sıralama için)
          - dense_score: ham cosine benzerliği (abstention/güven için — mutlak anlam taşır)
        """
        dense = self._dense_search(query, fetch_k)
        sparse = self._bm25_search(query, fetch_k)

        # ham dense cosine skorlarını idx -> cosine olarak sakla (abstention sinyali)
        dense_cos = {idx: sc for idx, sc in dense}

        rrf_scores = {}
        for rank, (idx, _) in enumerate(dense):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, (idx, _) in enumerate(sparse):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        seen_karar = set()
        for idx, score in ranked:
            rec = self.records[idx]
            kid = rec.get("karar_id")
            if kid in seen_karar:
                continue
            seen_karar.add(kid)
            results.append({
                **rec,
                "score": score,
                "dense_score": float(dense_cos.get(idx, 0.0)),
                "dense_rank": next((r for r, (i, _) in enumerate(dense) if i == idx), None),
                "bm25_rank": next((r for r, (i, _) in enumerate(sparse) if i == idx), None),
            })
            if len(results) >= top_k:
                break
        return results
