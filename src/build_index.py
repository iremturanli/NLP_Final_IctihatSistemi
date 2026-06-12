"""Tüm pipeline'ı koştur: veri yükle → chunk → embed → indeks kaydet."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_decisions
from hf_loader import load_hf_decisions
from chunker import build_chunk_records
from embedder import load_model, embed_passages, build_faiss_index, save_index


def main(source, json_path, n_decisions=3000, out_dir="../artifacts/index_v1",
         model_name="intfloat/multilingual-e5-base",
         max_chars=1200, overlap=150, batch_size=32):
    t0 = time.time()
    print(f"[1/4] Loading {n_decisions} decisions from source={source}...")
    if source == "hf":
        decisions = load_hf_decisions(n=n_decisions)
    else:
        decisions = load_decisions(json_path, limit=n_decisions)
    print(f"  -> {len(decisions)} decisions loaded ({time.time()-t0:.1f}s)")

    t1 = time.time()
    print(f"[2/4] Chunking (max_chars={max_chars}, overlap={overlap})...")
    chunk_records = build_chunk_records(decisions, max_chars=max_chars, overlap=overlap)
    print(f"  -> {len(chunk_records)} chunks ({time.time()-t1:.1f}s)")

    t2 = time.time()
    print(f"[3/4] Embedding with {model_name}...")
    model = load_model(model_name)
    texts = [r["text"] for r in chunk_records]
    embeddings = embed_passages(model, texts, batch_size=batch_size)
    print(f"  -> embeddings shape {embeddings.shape} ({time.time()-t2:.1f}s)")

    t3 = time.time()
    print(f"[4/4] Building FAISS index + saving to {out_dir}...")
    index = build_faiss_index(embeddings)
    save_index(index, embeddings, chunk_records, out_dir)
    print(f"  -> done ({time.time()-t3:.1f}s)")
    print(f"TOTAL: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["hf", "json"], default="hf",
                   help="hf: HuggingFace dataset (default); json: legacy local JSON")
    p.add_argument("--json", default="../KARARLAR_202601061508.json")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--out", default="../artifacts/index_v1")
    p.add_argument("--model", default="intfloat/multilingual-e5-base")
    p.add_argument("--max-chars", type=int, default=1200)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    main(args.source, args.json, args.n, args.out, args.model, args.max_chars, args.overlap, args.batch)
