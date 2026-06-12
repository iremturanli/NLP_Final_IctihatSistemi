# YargıRAG

**Türkçe Hukuki Yapay Zeka için Açık Benchmark, Domain-Adaptive Retrieval ve Halüsinasyon-Güvenli Soru-Cevap Sistemi**

Genel amaçlı yapay zeka hukuki sorulara kaynak göstermeden ve bazen var olmayan kararlar uydurarak cevap verir — hukukta bu tehlikelidir. YargıRAG, açık kaynak Türkçe Yargıtay/Danıştay kararları üzerinde **kaynak gösteren, "bilmiyorum" diyebilen (abstention) ve halüsinasyonu ölçülmüş** bir sistemdir.

## Üç Somut Çıktı (hepsi açık)

1. **TurkLegalBench** — Türkçe hukuk retrieval için **ilk açık benchmark** (BEIR formatı, HuggingFace)
2. **legal-e5-tr** — Türkçe hukuka **domain-adaptive fine-tune** edilmiş açık embedding modeli (HuggingFace)
3. **YargıRAG** — Hibrit retrieval + reranking + grounded generation + abstention + faithfulness ölçümü

## Sonuçlar (TurkLegalBench, sızıntısız test bölmesi — 573 sorgu)

| Model | Recall@1 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| BERTurk-Legal (mevcut hukuk modeli) | 0.031 | 0.069 | 0.095 |
| mE5-base (temel model) | 0.112 | 0.169 | 0.206 |
| mE5-large (560M) | 0.152 | 0.221 | 0.263 |
| BGE-m3 (SOTA, 560M) | 0.162 | 0.239 | 0.284 |
| BM25 | 0.185 | 0.273 | 0.322 |
| **legal-e5-tr (bizim, 110M)** | **0.353** | **0.446** | **0.495** |

- 🎯 Temel modeli nDCG@10'da **%140 göreli** artırdı (0.206→0.495), **p<0.001** (eşlemeli bootstrap)
- 🎯 5× büyük SOTA modeli (BGE-m3) ve BM25'i istatistiksel anlamlı geçti
- 🎯 Karar-bazlı sızıntısız (leakage-free) bölme

### Gerçekçi Gold Set (30 doğal hukuki soru)

| | Silver (otomatik) | **Gold (gerçek sorular)** |
|---|---:|---:|
| Recall@1 | 0.353 | **0.800** |
| Recall@5 | 0.576 | **0.900** |
| nDCG@10 | 0.495 | **0.825** |

Gerçek sorgularda sistem ilk denemede %80, ilk 5'te %90 doğru emsali bulur.

> Detay: [`artifacts/colab_results_FINAL.md`](artifacts/colab_results_FINAL.md), makale: [`paper/yargirag.tex`](paper/yargirag.tex)

## Farklılaşma (Aynı Konuyu Yapan Diğer Öğrenciden)

| Eksen | Tipik hukuk-chatbot ödevi | YargıRAG |
|---|---|---|
| **Ölçüm** | Benchmark yok | **TurkLegalBench** — standart koyar |
| **Güvenlik** | Her soruya cevap (uydurabilir) | **Halüsinasyon ölçümü + abstention** |
| **Model** | Hazır model tüketir | **legal-e5-tr** — kendi modeli, HF'te yayımlı |

> "O bir hukuk soru-cevap ürünü yapıyor; ben Türkçe hukuk YZ için ölçüm altyapısı + güvenlik katmanı + açık model üretiyorum."

## Proje Yapısı

```
project/
├── src/
│   ├── hf_loader.py          # açık kaynak HF veri yükleme
│   ├── chunker.py            # paragraf+cümle bazlı chunk
│   ├── embedder.py           # embedding + FAISS (GPU/CPU)
│   ├── retriever.py          # BM25 + dense + RRF hibrit
│   ├── reranker.py           # cross-encoder yeniden sıralama
│   ├── answer.py             # kaynaklı extractive cevap
│   ├── train_finetune.py     # MNRL fine-tune (fp16)
│   ├── build_benchmark.py    # TurkLegalBench üretimi (BEIR)
│   ├── hard_negatives.py     # zor olumsuz örnek madenciliği
│   ├── baselines.py          # 8+ model değerlendirme
│   ├── stats.py              # bootstrap CI + paired test
│   ├── generative_rag.py     # LLM + faithfulness + abstention
│   ├── hf_push.py            # HuggingFace yayını
│   ├── build_index.py        # uçtan uca indeks
│   └── app.py                # Gradio demo
├── colab/
│   └── YargiRAG_full_pipeline.ipynb   # A100 uçtan uca
├── paper/
│   └── yargirag.tex          # IEEE Türkçe bildiri (Overleaf)
├── artifacts/                # indeksler, model, benchmark, sonuçlar
├── demo.ipynb                # canlı sunum notebook'u
├── MASTER_PLAN.md            # vizyon + takvim + farklılaşma
├── SONUCLAR.md / SUNUM_KONUSMA.md
└── README.md
```

## Hızlı Başlangıç

### Local demo (mevcut, çalışıyor)
```bash
conda activate new_iwish
cd project
PYTHONNOUSERSITE=1 python src/app.py        # Gradio → http://localhost:7860
```

### Colab A100 (büyük ölçek — tez/makale için)
1. `YargiRAG_src.zip`'i veya `src/`'i Google Drive `MyDrive/YargiRAG/` altına yükle
2. `colab/YargiRAG_full_pipeline.ipynb`'i Colab'de aç → Runtime: A100
3. "Run All" — benchmark üretir, baseline koşar, fine-tune eder, HF'e yükler

## Akademik Katkılar

- **C1:** TurkLegalBench — ilk açık Türkçe hukuk retrieval benchmark'ı
- **C2:** legal-e5-tr — domain-adaptive embedding + hard negative mining
- **C3:** 8+ modelli kapsamlı baseline karşılaştırması
- **C4:** Halüsinasyon-güvenli RAG (faithfulness + abstention ölçümü)
- **C5:** Genel LLM vs grounded-RAG güvenlik karşılaştırması
- **C6:** Ablation + istatistiksel anlamlılık

## Veri & Lisans
- Veri: [`erdem-erdem/Turkish-Law-Documents-700k-clustered`](https://huggingface.co/datasets/erdem-erdem/Turkish-Law-Documents-700k-clustered) (açık, Yargıtay+Danıştay)
- Kod: MIT · Benchmark: CC-BY-4.0
- ⚠ Sistem hukuki tavsiye sağlamaz; kaynak gösteren bir araştırma aracıdır.
