"""Dense embedding + FAISS indeks."""
import os
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch


def get_device():
    """CUDA varsa onu, yoksa CPU'yu döndür."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name="intfloat/multilingual-e5-base", device=None):
    """Embedding modelini yükle."""
    if device is None:
        device = get_device()
    print(f"[embedder] Loading {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    return model


def _format_passage(text):
    """e5 ailesi için passage prefix'i ekle."""
    return f"passage: {text}"


def _format_query(text):
    """e5 ailesi için query prefix'i ekle."""
    return f"query: {text}"


def embed_passages(model, texts, batch_size=32):
    """Chunk metinleri için vektör üret. L2-normalize edilir (cosine için)."""
    formatted = [_format_passage(t) for t in texts]
    embs = model.encode(
        formatted,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs.astype(np.float32)


def embed_query(model, q):
    formatted = [_format_query(q)]
    e = model.encode(formatted, convert_to_numpy=True, normalize_embeddings=True)
    return e.astype(np.float32)


def build_faiss_index(embeddings):
    """Inner product (cosine, normalize edilmiş vektörler için) indeks kur."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index, embeddings, records, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "faiss.index"))
    np.save(out / "embeddings.npy", embeddings)
    with open(out / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[embedder] Saved index + embeddings + {len(records)} records to {out_dir}")


def load_index(out_dir):
    out = Path(out_dir)
    index = faiss.read_index(str(out / "faiss.index"))
    embeddings = np.load(out / "embeddings.npy")
    records = []
    with open(out / "records.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return index, embeddings, records
