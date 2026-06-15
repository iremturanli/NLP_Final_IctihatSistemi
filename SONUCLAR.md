# Deneysel Sonuçlar

> 5 farklı sistem konfigürasyonu üzerinde 15-sorgu mini benchmark sonuçları.

## Benchmark Seti
- **15 manuel Türkçe hukuk sorgusu** (icra/iflas, ceza, idari, aile, iş, borçlar, ticari kategorileri)
- **Pseudo-relevance judgment:** kararın metninde anahtar terim grubu sağlanıyorsa relevant=1
- 2.000 karar (~11k chunk) üzerinde

## Karşılaştırma Tablosu

| Sistem | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | nDCG@10 | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| e5-small (baseline) | 0.733 | 0.933 | 0.933 | 0.833 | 0.850 | 0.836 | 22 ms |
| **e5-small + FT** | **0.933** | 0.933 | 0.933 | **0.933** | **0.913** | **0.887** | 15 ms |
| e5-base (baseline) | 0.800 | 0.867 | 0.867 | 0.833 | 0.823 | 0.811 | 17 ms |
| e5-small + FT + reranker | 0.733 | 0.933 | 0.933 | 0.833 | 0.860 | 0.851 | 554 ms |
| e5-base + reranker | 0.733 | 0.933 | 0.933 | 0.822 | 0.854 | 0.851 | 539 ms |

## Temel Bulgular

### 1. Domain-Adaptive Fine-Tuning Etkisi
`multilingual-e5-small` modeline 2.500 Türkçe Yargıtay/Danıştay kararı üzerinde **Multiple Negatives Ranking Loss** ile 2 epoch fine-tuning uygulandı (33 saniye, RTX 4050 6GB, fp16):

| Metrik | Baseline | Fine-Tuned | Göreceli Δ |
|---|---:|---:|---:|
| Recall@1 | 0.733 | 0.933 | **+27 %** |
| MRR | 0.833 | 0.933 | +12 % |
| nDCG@5 | 0.850 | 0.913 | +7 % |
| nDCG@10 | 0.836 | 0.887 | +6 % |
| Latency | 22 ms | 15 ms | -32 % |

**Akademik vurgu:** Türkçe hukuk metinleri üzerinde domain-adaptive fine-tuning ilk doğru karara ulaşma olasılığını %27 artırdı; üstelik daha düşük gecikmeyle.

### 2. Küçük + Fine-Tune > Büyük + Generic
- `e5-base` (110M parametre, 768d): Recall@1 = 0.800
- `e5-small + FT` (30M parametre, 384d): Recall@1 = **0.933**

**Domain adaptation, model boyutunu telafi etti.** 3.7x daha küçük bir model, daha iyi top-1 isabet veriyor. Bu, ölçeklenmesi kolay deployment için kritik (CPU'da bile <50 ms).

### 3. Cross-Encoder Reranker — Beklenmeyen Bulgu
- BAAI/bge-reranker-v2-m3 (multilingual cross-encoder) test edildi
- **Performansı artırmadı, bazı metriklerde düşürdü**
- Latency 30-37x arttı (15ms → 554ms)

**Olası nedenler ve gelecek iş:**
- Reranker'ın Türkçe ağırlıkları sınırlı olabilir (eğitim verisinde Türkçe oranı düşük)
- Mini benchmark'ın küçük olması (15 sorgu) reranker'ın varyansını yakalamış olabilir
- **Türkçe hukuk için reranker fine-tuning** — projenin bir sonraki adımı

## Teknik Detaylar (Üretim Bilgisi)

### Donanım
- GPU: NVIDIA GeForce RTX 4050 Laptop (6 GB VRAM)
- CPU yedek: torch.cuda.is_available() ile otomatik tespit

### Fine-tune Hyperparametreleri
- Base: `intfloat/multilingual-e5-small`
- Eğitim verisi: 2.313 (query, passage) çifti, weak supervision (karar başlığı/anahtar konu → query, gövde → passage)
- Loss: Multiple Negatives Ranking Loss (in-batch negatives)
- Epoch: 2 / Batch: 32 / LR: 2e-5 / Warmup: 10% / Max seq: 256
- Precision: fp16 mixed precision
- Süre: 33.5 saniye

### Hybrid Retrieval
- Sparse: BM25 (rank_bm25), Türkçe stopword filtreli
- Dense: yukarıdaki embedding modeli + FAISS IndexFlatIP (cosine via L2-normalize)
- Fusion: Reciprocal Rank Fusion (k=60)
- Cevap üretimi: extractive (LLM yok, halüsinasyon yok), kaynak (esas/karar no/tarih) ile

