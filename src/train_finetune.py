"""multilingual-e5-base modelini Türkçe hukuk metinleri üzerinde fine-tune et.

Strateji: Multiple Negatives Ranking Loss (MNRL)
  - Her örnekte (query, positive) çifti
  - Batch içindeki diğer pozitifler in-batch negative olarak kullanılır
  - Manuel negative seçimi gerekmez

Zayıf-denetim:
  - Query: karar başlığı (içtihat metnindeki büyük harfli anahtar kavram)
  - Positive: karar metninin gövdesinden uzun bir chunk

Çıktı: `artifacts/e5-tr-legal-ft/` altında fine-tune edilmiş model
"""
import os
import sys
import re
import random
import time
from pathlib import Path

# CUDA bellek parçalanmasını azalt
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).parent))


STOP_QUERIES = {
    "türk milleti adına", "i̇çtihat metni", "içtihat metni",
    "yargıtay kararı", "karar verilmiştir", "türk milleti",
    "yargitay kararı", "tasarlayarak",
}


def _is_metadata_query(s):
    """Esas/karar no, daire adı + metadata gibi anlamsız sorguları filtrele."""
    # Sadece metadata: "X. Ceza Dairesi - E. 1234, K. 5678" pattern
    if re.search(r'\b[ek]\.\s*\d+', s):
        return True
    if re.fullmatch(r'[\d/.\s-]+', s):
        return True
    return False


def extract_query_from_title(title, text):
    """Karar başlığı veya metnin ilk kısımlarından sorgu üret."""
    cands = []
    head = text[:600]
    # Büyük harf yoğunluğu yüksek başlık satırlarını yakala (içtihat anahtar kavramları)
    big_caps = re.findall(r'(?:^|\n)\s*([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ \'\-]{8,60})\s*(?:\n|$)', head)
    for bc in big_caps:
        s = bc.strip().lower()
        if 8 < len(s) < 100 and len(s.split()) >= 2:
            if s in STOP_QUERIES or _is_metadata_query(s):
                continue
            cands.append(s)
    # SUÇ veya KONU
    m = re.search(r'(?:SUÇ|KONU|DAVA)\s*:\s*([^\n]+)', head)
    if m:
        s = m.group(1).strip().lower()[:80]
        if s not in STOP_QUERIES and not _is_metadata_query(s):
            cands.append(s)
    return cands


def build_training_pairs(decisions, max_pairs_per_decision=3, min_chunk_chars=400):
    """Karar listesinden (query, passage) çiftleri üret.

    Birden fazla strateji ile sorgu çıkarır; her bir kararın farklı chunk'larına eşler
    böylece hem leksik hem anlamsal sinyaller train edilir.
    """
    from chunker import chunk_text
    import random as _r
    pairs = []
    for d in decisions:
        text = d.get("text", "")
        if len(text) < min_chunk_chars * 2:
            continue
        queries = extract_query_from_title(d.get("title", ""), text)
        # Yedek: konu/suç (anlamsal)
        if d.get("konu"):
            s = d["konu"].lower()
            if s not in STOP_QUERIES and not _is_metadata_query(s):
                queries.append(s)
        # Yedek: kanun atfı (BM25 odaklı senkronlama için)
        if d.get("kanunlar"):
            for k in d["kanunlar"][:2]:
                queries.append(k.lower())
        # Title'dan kurul + konuyu de query yapabiliriz
        if d.get("title"):
            t = d["title"].lower()
            # "X. Daire - E. ..., K. ..." kısmından sonra varsa konuyu al
            if ' - ' in t:
                konu_part = t.split(' - ', 1)[1]
                if konu_part not in STOP_QUERIES and not _is_metadata_query(konu_part):
                    queries.append(konu_part)

        queries = [q for q in queries if 8 < len(q) < 200]
        queries = list(dict.fromkeys(queries))  # benzersizleştir

        # Fallback: hiç anlamlı query yoksa, kararın ilk anlamlı cümlesini query olarak kullan
        if not queries:
            head = text[:1500]
            # Metadata sonrası ilk gerçek cümleyi bul
            sents = re.split(r'(?<=[.!?])\s+', head)
            for s in sents:
                s = s.strip()
                s_low = s.lower()
                if 40 < len(s) < 200 and not _is_metadata_query(s_low) \
                   and "i̇çtihat" not in s_low and "mahkemesi" not in s_low \
                   and "esas" not in s_low and "karar" not in s_low.split(' ')[0]:
                    queries.append(s_low[:180])
                    break
        if not queries:
            continue

        chunks = chunk_text(text, max_chars=1200, overlap=100)
        chunks = [c for c in chunks if len(c) >= min_chunk_chars]
        if not chunks:
            continue

        for q in queries[:max_pairs_per_decision]:
            passage = random.choice(chunks)
            pairs.append((q, passage))
    return pairs


def main(n_decisions=2000, epochs=1, batch_size=16, out_dir="artifacts/e5-tr-legal-ft",
         base_model="intfloat/multilingual-e5-base", max_pairs_per_decision=2,
         warmup_ratio=0.1):
    from hf_loader import load_hf_decisions
    from sentence_transformers import (
        SentenceTransformer, losses,
        SentenceTransformerTrainer, SentenceTransformerTrainingArguments,
    )
    from datasets import Dataset
    import torch

    print(f"[finetune] Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"[finetune] Loading {n_decisions} decisions from HF...")
    decisions = load_hf_decisions(n=n_decisions)

    print(f"[finetune] Building (query, passage) pairs...")
    pairs = build_training_pairs(decisions, max_pairs_per_decision=max_pairs_per_decision)
    random.shuffle(pairs)
    print(f"  -> {len(pairs)} training pairs")
    if not pairs:
        raise RuntimeError("No training pairs generated. Check query extraction logic.")

    # Sanity preview
    for q, p in pairs[:3]:
        print(f"  Q: {q[:80]}")
        print(f"  P: {p[:80]}")
        print()

    print(f"[finetune] Loading base model {base_model}...")
    model = SentenceTransformer(base_model)
    # Bellek tasarrufu: 256 token'a indir (chunk'lar zaten ~200-300 token civarı)
    model.max_seq_length = int(os.environ.get("LEGALRAG_MAX_SEQ_LEN", "256"))
    print(f"[finetune] max_seq_length={model.max_seq_length}")

    # HF Dataset formatına dönüştür (sentence-transformers v3+ Trainer için)
    ds = Dataset.from_dict({
        "query": [f"query: {q}" for q, _ in pairs],
        "passage": [f"passage: {p[:1200]}" for _, p in pairs],
    })

    loss = losses.MultipleNegativesRankingLoss(model)

    total_steps = (len(ds) // batch_size) * epochs
    print(f"[finetune] Training: {len(ds)} examples, batch={batch_size}, epochs={epochs}, "
          f"total_steps≈{total_steps}, fp16=True")

    args = SentenceTransformerTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_ratio=warmup_ratio,
        fp16=True,                          # FP16 mixed precision (yarı bellek)
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        logging_steps=20,
        save_strategy="no",                 # epoch sonu kaydetme; sonra manuel save
        report_to=[],
        dataloader_drop_last=True,
        dataloader_num_workers=0,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        loss=loss,
    )

    t0 = time.time()
    trainer.train()
    print(f"[finetune] Training done in {time.time()-t0:.1f}s.")

    model.save_pretrained(out_dir)
    print(f"[finetune] Saved to {out_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000, help="Karar sayısı")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", default="../artifacts/e5-tr-legal-ft")
    p.add_argument("--base", default="intfloat/multilingual-e5-base")
    p.add_argument("--pairs-per", type=int, default=2)
    args = p.parse_args()
    random.seed(42)
    main(args.n, args.epochs, args.batch, args.out, args.base, args.pairs_per)
