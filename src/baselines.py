"""Çoklu-model retrieval değerlendirmesi — TurkLegalBench üzerinde baseline matrisi.

BEIR-format benchmark yükler, her model için corpus+query encode eder, retrieval
yapar ve standart metrikleri (Recall@k, MRR@10, nDCG@10, MAP) hesaplar.

Desteklenen model türleri:
  - 'bm25'                 : sparse lexical
  - 'sentence-transformer' : herhangi bir ST modeli (e5, bge, jina, fine-tuned)
  - 'mean-pool'            : retrieval-FT olmayan BERT (BERTurk) için mean pooling

Sonuç: her model için metrik dict + karşılaştırma tablosu.
"""
import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict


# ---------- Benchmark yükleme ----------

def load_beir(bench_dir):
    bench_dir = Path(bench_dir)
    corpus, queries = {}, {}
    with open(bench_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            corpus[r["_id"]] = r
    with open(bench_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            queries[r["_id"]] = r
    qrels = defaultdict(dict)
    with open(bench_dir / "qrels.tsv", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            qid, did, score = line.strip().split("\t")
            qrels[qid][did] = int(score)
    return corpus, queries, dict(qrels)


# ---------- Metrikler ----------

def _dcg(rels):
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def compute_metrics(ranked_ids, relevant_map, k_values=(1, 5, 10, 100)):
    """ranked_ids: sıralı doc_id listesi; relevant_map: {doc_id: rel}"""
    rels = [relevant_map.get(did, 0) for did in ranked_ids]
    out = {}
    for k in k_values:
        topk = rels[:k]
        out[f"recall@{k}"] = int(any(r > 0 for r in topk))
    # MRR@10
    rr = 0.0
    for i, r in enumerate(rels[:10], 1):
        if r > 0:
            rr = 1.0 / i
            break
    out["mrr@10"] = rr
    # nDCG@10
    topk = rels[:10]
    ideal = sorted([r for r in relevant_map.values()], reverse=True)[:10]
    idcg = _dcg(ideal) or 1.0
    out["ndcg@10"] = _dcg(topk) / idcg
    # MAP
    n_rel = sum(1 for r in relevant_map.values() if r > 0)
    hits, ap = 0, 0.0
    for i, r in enumerate(rels, 1):
        if r > 0:
            hits += 1
            ap += hits / i
    out["map"] = ap / n_rel if n_rel else 0.0
    return out


def aggregate_metrics(per_query):
    agg = defaultdict(float)
    for m in per_query:
        for k, v in m.items():
            agg[k] += v
    n = len(per_query) or 1
    return {k: v / n for k, v in agg.items()}


# ---------- Encoder'lar ----------

def encode_bm25(corpus_ids, corpus_texts, queries, tokenize_fn, top_k=100):
    """BM25 retrieval; her sorgu için sıralı doc_id listesi döndür."""
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([tokenize_fn(t) for t in corpus_texts])
    results = {}
    for qid, q in queries.items():
        toks = tokenize_fn(q["text"])
        if not toks:
            results[qid] = []
            continue
        scores = bm25.get_scores(toks)
        order = np.argsort(scores)[::-1][:top_k]
        results[qid] = [corpus_ids[i] for i in order]
    return results


def encode_dense(model, corpus_ids, corpus_texts, queries, top_k=100,
                 query_prefix="query: ", passage_prefix="passage: ",
                 batch_size=128, mean_pool=False):
    """Dense retrieval; SentenceTransformer veya mean-pool BERT."""
    import faiss
    if mean_pool:
        emb_corpus = _mean_pool_encode(model, [passage_prefix + t for t in corpus_texts], batch_size)
        emb_query = _mean_pool_encode(model, [query_prefix + queries[q]["text"] for q in queries], batch_size)
    else:
        emb_corpus = model.encode([passage_prefix + t for t in corpus_texts],
                                  batch_size=batch_size, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        qlist = list(queries.keys())
        emb_query = model.encode([query_prefix + queries[q]["text"] for q in qlist],
                                 batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    index = faiss.IndexFlatIP(emb_corpus.shape[1])
    index.add(emb_corpus)
    qlist = list(queries.keys())
    _, idxs = index.search(emb_query, top_k)
    return {qid: [corpus_ids[i] for i in row] for qid, row in zip(qlist, idxs)}


def _mean_pool_encode(hf_model_tuple, texts, batch_size=64):
    """BERTurk gibi retrieval-FT olmayan modeller için mean pooling."""
    import torch
    tokenizer, model = hf_model_tuple
    device = next(model.parameters()).device
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256,
                        return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(1)
        counts = mask.sum(1).clamp(min=1e-9)
        mean = summed / counts
        mean = torch.nn.functional.normalize(mean, p=2, dim=1)
        embs.append(mean.cpu().numpy())
    return np.vstack(embs).astype(np.float32)


# ---------- Üst seviye değerlendirme ----------

def evaluate_model(model_spec, corpus, queries, qrels, tokenize_fn, top_k=100):
    """Tek bir modeli değerlendir.

    model_spec: {"name", "type", "model"(opsiyonel), "kwargs"(opsiyonel)}
    """
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[c]["text"] for c in corpus_ids]
    mtype = model_spec["type"]

    if mtype == "bm25":
        results = encode_bm25(corpus_ids, corpus_texts, queries, tokenize_fn, top_k)
    elif mtype == "sentence-transformer":
        results = encode_dense(model_spec["model"], corpus_ids, corpus_texts, queries,
                               top_k=top_k, **model_spec.get("kwargs", {}))
    elif mtype == "mean-pool":
        results = encode_dense(model_spec["model"], corpus_ids, corpus_texts, queries,
                               top_k=top_k, mean_pool=True, **model_spec.get("kwargs", {}))
    else:
        raise ValueError(f"Bilinmeyen model tipi: {mtype}")

    per_query = []
    for qid, ranked in results.items():
        if qid not in qrels:
            continue
        per_query.append(compute_metrics(ranked, qrels[qid]))
    return aggregate_metrics(per_query), per_query, results


def run_suite(model_configs, corpus, queries, qrels, tokenize_fn,
              top_k=100, device="cuda", batch_size=64):
    """Modelleri TEK TEK yükle → değerlendir → GPU'dan boşalt (OOM önler).

    model_configs: list of dict, her biri:
      {"name", "type", "model_id"(ST/mean-pool için), "kwargs"(opsiyonel)}
    Döndürür: (all_results: {name: agg}, per_query_store: {name: per_query})

    Kritik: her modelden sonra del + torch.cuda.empty_cache() ile bellek serbest.
    Büyük modeller (e5-large, BGE-m3) için bu zorunludur — aksi halde hepsi aynı
    anda GPU'da kalır ve 80GB bile yetmez.
    """
    import torch
    import gc

    all_results, per_query_store = {}, {}
    for cfg in model_configs:
        name, mtype = cfg["name"], cfg["type"]
        print(f"\n>>> {name}")
        model_obj = None
        try:
            if mtype == "bm25":
                spec = {"name": name, "type": "bm25"}
            elif mtype == "sentence-transformer":
                from sentence_transformers import SentenceTransformer
                model_obj = SentenceTransformer(cfg["model_id"], device=device)
                if cfg.get("max_seq_length"):
                    model_obj.max_seq_length = cfg["max_seq_length"]
                spec = {"name": name, "type": mtype, "model": model_obj,
                        "kwargs": cfg.get("kwargs", {})}
            elif mtype == "mean-pool":
                from transformers import AutoTokenizer, AutoModel
                tok = AutoTokenizer.from_pretrained(cfg["model_id"])
                mdl = AutoModel.from_pretrained(cfg["model_id"]).to(device).eval()
                model_obj = (tok, mdl)
                spec = {"name": name, "type": mtype, "model": model_obj,
                        "kwargs": cfg.get("kwargs", {})}
            else:
                print(f"  (atlandı: bilinmeyen tip {mtype})")
                continue

            agg, per_q, _ = evaluate_model(spec, corpus, queries, qrels, tokenize_fn,
                                           top_k=top_k)
            all_results[name] = agg
            per_query_store[name] = per_q
            print(f"  R@1={agg['recall@1']:.3f} R@5={agg['recall@5']:.3f} "
                  f"MRR@10={agg['mrr@10']:.3f} nDCG@10={agg['ndcg@10']:.3f}")
        except Exception as e:
            print(f"  HATA ({name}): {type(e).__name__}: {str(e)[:120]}")
        finally:
            # GPU belleğini serbest bırak — bir sonraki model için kritik
            if model_obj is not None:
                if isinstance(model_obj, tuple):
                    for o in model_obj:
                        del o
                del model_obj
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return all_results, per_query_store


def comparison_table(all_results):
    """all_results: {model_name: agg_dict} -> yazdırılabilir tablo string."""
    keys = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10", "map"]
    lines = []
    header = f"{'Model':<32}" + "".join(f"{k:>11}" for k in keys)
    lines.append(header)
    lines.append("-" * len(header))
    for name, agg in all_results.items():
        row = f"{name:<32}" + "".join(f"{agg.get(k,0):>11.3f}" for k in keys)
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    print("baselines.py — Colab notebook'undan çağrılır.")
    print("Akış: load_beir() -> evaluate_model(her model) -> comparison_table()")
