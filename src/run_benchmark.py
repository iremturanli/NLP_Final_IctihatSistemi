"""5 konfigürasyon üzerinde benchmark koş ve karşılaştırmalı tablo çıkar.

Konfig matrisi:
  1) e5-small (no FT)       — small baseline
  2) e5-small + FT          — fine-tuned
  3) e5-base                — large baseline
  4) e5-small + FT + reranker
  5) e5-base + reranker

Metrikler: Recall@{1,3,5,10}, MRR, nDCG@{5,10}
"""
import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from embedder import load_model, load_index
from retriever import HybridRetriever
from reranker import RerankedRetriever
from benchmark import BENCHMARK_QUERIES, evaluate_results, aggregate


CONFIGS = [
    {
        "name": "e5-small (baseline)",
        "index_dir": "artifacts/index_small_base",
        "model_name": "intfloat/multilingual-e5-small",
        "use_reranker": False,
    },
    {
        "name": "e5-small + FT",
        "index_dir": "artifacts/index_small_ft",
        "model_name": "artifacts/e5-small-tr-legal-ft",
        "use_reranker": False,
    },
    {
        "name": "e5-base (baseline)",
        "index_dir": "artifacts/index_v1",
        "model_name": "intfloat/multilingual-e5-base",
        "use_reranker": False,
    },
    {
        "name": "e5-small + FT + reranker",
        "index_dir": "artifacts/index_small_ft",
        "model_name": "artifacts/e5-small-tr-legal-ft",
        "use_reranker": True,
    },
    {
        "name": "e5-base + reranker",
        "index_dir": "artifacts/index_v1",
        "model_name": "intfloat/multilingual-e5-base",
        "use_reranker": True,
    },
]


def run_config(cfg, fetch_k=15):
    print(f"\n{'='*70}\n>>> {cfg['name']}\n{'='*70}")
    index, embeddings, records = load_index(cfg["index_dir"])
    model = load_model(cfg["model_name"])
    hybrid = HybridRetriever(index, embeddings, records, model)
    retriever = (RerankedRetriever(hybrid, fetch_k=fetch_k)
                 if cfg["use_reranker"] else hybrid)

    t0 = time.time()
    per_query = []
    for q_spec in BENCHMARK_QUERIES:
        results = retriever.search(q_spec["query"], top_k=10)
        metrics = evaluate_results(q_spec, results, k_values=(1, 3, 5, 10))
        per_query.append(metrics)
    elapsed = time.time() - t0

    agg = aggregate(per_query)
    agg["avg_latency_ms"] = (elapsed / len(BENCHMARK_QUERIES)) * 1000
    return agg, per_query


def main(out_path="artifacts/benchmark_results.json"):
    results = {}
    for cfg in CONFIGS:
        agg, per_query = run_config(cfg)
        results[cfg["name"]] = {"config": cfg, "agg": agg, "per_query": per_query}
        print(f"  Recall@5={agg['recall@5']:.3f}  MRR={agg['mrr']:.3f}  "
              f"nDCG@10={agg['ndcg@10']:.3f}  latency={agg['avg_latency_ms']:.0f}ms")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Pretty table
    print(f"\n{'='*90}\nCOMPARISON TABLE ({len(BENCHMARK_QUERIES)} queries)\n{'='*90}")
    metric_keys = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10", "avg_latency_ms"]
    header = f"{'System':<32}" + "".join(f"{k:>11}" for k in metric_keys)
    print(header)
    print("-" * len(header))
    for name, data in results.items():
        agg = data["agg"]
        row = f"{name:<32}"
        for k in metric_keys:
            v = agg.get(k, 0)
            if k == "avg_latency_ms":
                row += f"{v:>10.0f}ms"
            else:
                row += f"{v:>11.3f}"
        print(row)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
