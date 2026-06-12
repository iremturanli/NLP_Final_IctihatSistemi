"""İndeks üzerinde örnek sorgularla retrieval kalitesini gözle değerlendir.

Kullanım:
  python src/eval_retrieval.py --index artifacts/index_v1
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from embedder import load_model, load_index
from retriever import HybridRetriever


TEST_QUERIES = [
    ("ihtiyati haciz, icra takibi sayılır mı", "icra/iflas"),
    ("karşılıksız yararlanma suçunda beraat", "ceza"),
    ("idari yargıda yürütmenin durdurulması", "idari"),
    ("4733 sayılı yasa tütün muhalefet", "ceza/leksik"),
    ("boşanma davasında nafaka miktarı", "aile"),
    ("iş akdinin haksız feshi tazminatı", "iş hukuku"),
    ("konkordato mühleti süresince alacaklı haklarına etkisi", "icra/iflas"),
    ("2004 sayılı İcra İflas Kanunu 264. madde", "leksik test"),
    ("kasten öldürme suçunda tasarlama", "ceza"),
    ("kira sözleşmesinde tahliye taahhüdü", "borçlar"),
]


def evaluate(retriever, top_k=5):
    for q, kategori in TEST_QUERIES:
        print(f"\n{'='*70}\n[{kategori}] SORGU: {q}\n{'='*70}")
        res = retriever.search(q, top_k=top_k)
        if not res:
            print("  (Sonuç yok)")
            continue
        for i, r in enumerate(res, 1):
            kurul = r.get("kurul") or "?"
            esas = r.get("esas_no") or "?"
            karar = r.get("karar_no") or "?"
            tarih = r.get("tarih") or "?"
            score = r.get("score", 0)
            snippet = r["text"][:200].replace("\n", " ")
            print(f"  [{i}] score={score:.4f} | {kurul} | E.{esas}/K.{karar} ({tarih})")
            print(f"      {snippet}...")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="artifacts/index_v1")
    p.add_argument("--model", default="intfloat/multilingual-e5-base")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    print(f"Loading index from {args.index}...")
    index, embeddings, records = load_index(args.index)
    print(f"Loaded {len(records)} chunks ({len(set(r['karar_id'] for r in records))} kararlar)")

    model = load_model(args.model)
    retriever = HybridRetriever(index, embeddings, records, model)
    evaluate(retriever, top_k=args.top_k)


if __name__ == "__main__":
    main()
