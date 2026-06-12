"""Hard negative mining — fine-tune kalitesini artıran standart teknik.

In-batch negatives (basit) yerine, BM25/dense ile "yakın ama yanlış" örnekleri
açıkça seçer. Literatürde retrieval fine-tuning'de +5-10 puan getiren yöntem
(DPR, ANCE, RocketQA gelenekleri).

Strateji:
  Her (query, positive) çifti için:
    - BM25 veya dense ile top-N benzer dökümanı çek
    - positive olmayan, ama yüksek skorlu olanları "hard negative" seç
    - false-negative riskini azaltmak için pozitife çok benzer (skor eşiği üstü)
      olanları ele
"""
import random
import numpy as np


def mine_bm25_hard_negatives(pairs, corpus_texts, corpus_ids, bm25,
                             tokenize_fn, n_neg=4, top_k=30, skip_top=2, seed=42):
    """BM25 ile hard negative seç.

    pairs: list of (query, positive_doc_id)
    corpus_texts/corpus_ids: paralel listeler
    bm25: önceden kurulmuş BM25Okapi
    Döndürür: list of dict {query, positive_id, negative_ids: [...]}
    """
    random.seed(seed)
    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    out = []
    for query, pos_id in pairs:
        q_tokens = tokenize_fn(query)
        if not q_tokens:
            continue
        scores = bm25.get_scores(q_tokens)
        ranked = np.argsort(scores)[::-1]
        negs = []
        # skip_top: en üstteki birkaçı atla (büyük olasılıkla pozitifin kendisi/çok yakını)
        for idx in ranked[skip_top:top_k]:
            cid = corpus_ids[idx]
            if cid == pos_id:
                continue
            negs.append(cid)
            if len(negs) >= n_neg:
                break
        if negs:
            out.append({"query": query, "positive_id": pos_id, "negative_ids": negs})
    return out


def mine_dense_hard_negatives(pairs, embeddings, corpus_ids, model,
                              n_neg=4, top_k=30, skip_top=2, sim_ceiling=0.95):
    """Dense embedding ile hard negative seç.

    embeddings: (N, D) normalize edilmiş corpus embeddings
    pairs: list of (query, positive_id)
    sim_ceiling: pozitife çok benzer (false negative riski) olanları ele
    """
    import faiss
    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    out = []
    queries = [f"query: {q}" for q, _ in pairs]
    q_emb = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float32)
    scores, idxs = index.search(q_emb, top_k)

    for (query, pos_id), row_idx, row_score in zip(pairs, idxs, scores):
        pos_idx = id_to_idx.get(pos_id)
        negs = []
        for rank, (ci, sc) in enumerate(zip(row_idx, row_score)):
            if rank < skip_top:
                continue
            cid = corpus_ids[ci]
            if cid == pos_id:
                continue
            if sc >= sim_ceiling:    # pozitife aşırı benzer → muhtemelen gerçek positive
                continue
            negs.append(cid)
            if len(negs) >= n_neg:
                break
        if negs:
            out.append({"query": query, "positive_id": pos_id, "negative_ids": negs})
    return out


def to_triplet_examples(mined, corpus_map):
    """Mined hard negatives'i (anchor, positive, negative) üçlülerine çevir.

    corpus_map: dict doc_id -> text
    Döndürür: list of dict {anchor, positive, negative} (her negatif için bir satır)
    """
    triplets = []
    for m in mined:
        pos_text = corpus_map.get(m["positive_id"], "")
        if not pos_text:
            continue
        for neg_id in m["negative_ids"]:
            neg_text = corpus_map.get(neg_id, "")
            if not neg_text:
                continue
            triplets.append({
                "anchor": f"query: {m['query']}",
                "positive": f"passage: {pos_text[:1200]}",
                "negative": f"passage: {neg_text[:1200]}",
            })
    return triplets


if __name__ == "__main__":
    print("hard_negatives.py — fonksiyon modülü. Colab notebook'undan çağrılır.")
    print("  - mine_bm25_hard_negatives()")
    print("  - mine_dense_hard_negatives()")
    print("  - to_triplet_examples()")
