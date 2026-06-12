# YargıRAG — Master Plan

**Türkçe Hukuki Yapay Zeka için Açık Benchmark, Domain-Adaptive Retrieval ve Halüsinasyon-Güvenli Soru-Cevap Sistemi**

> Hedef: Cuma gece teslim. Colab Pro (A100) ile eğitim. Türkçe IEEE bildiri. HuggingFace açık yayın.

---

## 1. Konumlandırma (Neden Gündem Olur?)

### Problem (basın-dostu, net)
Genel amaçlı yapay zeka (ChatGPT, Gemini) hukuki bir soruya **kaynak göstermeden, bazen var olmayan kararlar uydurarak** cevap veriyor. ABD'de avukatlar ChatGPT'nin uydurduğu sahte kararları mahkemeye sundukları için ceza aldı (Mata v. Avianca, 2023). Türkçe'de ise ne bu riski ölçen bir araç, ne de güvenli bir alternatif var.

### Çözüm (üç somut çıktı)
1. **TurkLegalBench** — Türkçe hukuk bilgi erişimi için **ilk açık benchmark** (HuggingFace'te yayımlanacak)
2. **legal-e5-tr** — Türkçe hukuk metinlerine **domain-adaptive fine-tune** edilmiş açık embedding modeli (HuggingFace'te)
3. **YargıRAG** — Kaynak gösteren, "bilmiyorum" diyebilen (abstention), **halüsinasyonu ölçülmüş** uçtan uca sistem

### Ayırt edici "şok" karşılaştırması
Genel LLM (kaynak yok, halüsinasyon var) **vs** YargıRAG (kaynak var, halüsinasyon ölçülü) — tek tabloda.

---

## 1.5 Farklılaşma (Aynı Konuyu Yapan Diğer Öğrenciden Ayrışma)

**Diğer kişi büyük olasılıkla:** hazır embedding + hazır LLM ile standart bir hukuk soru-cevap chatbotu (ölçüm yok, güvenlik yok, yeni model yok).

**Bizim imza farkımız — üç katman:**

| Eksen | Diğer kişi (tipik) | YargıRAG (bu çalışma) |
|---|---|---|
| **Ölçüm** | Benchmark yok, "çalışıyor" der | **TurkLegalBench**: ilk açık Türkçe hukuk benchmark'ı — standart koyar |
| **Güvenlik** | Her soruya cevap (uydurabilir) | **Halüsinasyon ölçümü + abstention** ("bilmiyorum" diyebilir) |
| **Model** | Hazır model tüketir | **legal-e5-tr**: kendi fine-tuned modeli, HuggingFace'te yayımlı |

**Tek cümlelik konumlandırma:**
> "O bir hukuk soru-cevap *ürünü* yapıyor; ben Türkçe hukuk yapay zekası için *ölçüm altyapısı + güvenlik katmanı + açık model* üretiyorum — bilim ve altyapı."

Sonuç: diğer kişinin sistemi bile bu çalışmanın benchmark'ı (TurkLegalBench) ile değerlendirilmek zorunda — bu, çalışmayı "standart koyan" konuma getirir.

---

## 2. Akademik Katkı Listesi (Reviewer Diliyle)

| # | Katkı | Yenilik düzeyi |
|---|---|---|
| C1 | Türkçe hukuk retrieval için ilk açık benchmark (TurkLegalBench) | **Yüksek** — yok |
| C2 | Domain-adaptive Türkçe hukuk embedding (legal-e5-tr) + hard negative mining | Orta-Yüksek |
| C3 | 8+ modelli kapsamlı baseline karşılaştırması (BM25→SOTA) | Orta |
| C4 | Halüsinasyon-güvenli RAG + faithfulness/abstention ölçümü | Yüksek (Türkçe'de yok) |
| C5 | Genel LLM vs grounded-RAG güvenlik karşılaştırması | Yüksek (basın-dostu) |
| C6 | Ablation + istatistiksel anlamlılık | Metodolojik ciddiyet |

---

## 3. Sistem Mimarisi

```
                         ┌──────────────────────────────────────┐
   Kullanıcı sorusu ───► │  YargıRAG                            │
                         │                                       │
                         │  1. Hybrid Retrieval                  │
                         │     BM25  ─┐                          │
                         │     dense ─┼─► RRF ─► top-50          │
                         │     (legal-e5-tr)                     │
                         │                                       │
                         │  2. Cross-encoder Rerank ─► top-5     │
                         │                                       │
                         │  3. Grounded Generation               │
                         │     Türkçe LLM + retrieved bağlam      │
                         │     + faithfulness kontrolü           │
                         │     + abstention (düşük güven)        │
                         │                                       │
                         │  4. Kaynak gösterimi                  │
                         │     [Karar: 17.CD E.2016/12248]       │
                         └──────────────────────────────────────┘
```

---

## 4. Deney Matrisi

### 4.1 Retrieval Baseline'ları (TurkLegalBench üzerinde)
| Model | Tür | Parametre |
|---|---|---|
| BM25 | sparse/lexical | — |
| BERTurk (mean-pool) | dense, retrieval-FT değil | 110M |
| multilingual-e5-small | dense | 30M |
| multilingual-e5-base | dense | 110M |
| multilingual-e5-large | dense | 560M |
| BGE-m3 | dense+sparse+colbert | 560M |
| jina-embeddings-v3 | dense | 570M |
| KocLab BERTurk-Legal | Türkçe legal | 110M |
| **legal-e5-tr (BİZİM)** | domain-adaptive | 110-560M |
| **+ hard negatives** | | |
| **+ reranker** | | |

### 4.2 Metrikler
- **Retrieval:** Recall@{1,5,10,100}, MRR@10, nDCG@10, MAP
- **Generative:** Faithfulness (NLI-based), Citation Accuracy, Abstention Precision/Recall, Answer Relevance
- **İstatistik:** 3 seed, bootstrap %95 GA, baseline'a karşı paired t-test

### 4.3 Ablation
- Chunk boyutu: 256 / 512 / 1024
- Retrieval: dense-only / sparse-only / hybrid
- Fine-tune: epoch ve veri miktarı learning curve
- Hard negatives: var / yok
- Reranker: var / yok

---

## 5. TurkLegalBench Oluşturma Metodolojisi

Manuel 500 sorgu etiketlemek 4 günde olmaz; **hibrit otomatik+manuel** yöntem (BEIR/NFCorpus geleneği):

1. **Otomatik query-doc çiftleri (silver):**
   - Kararın anahtar konusu/hukmü → query, karar gövdesi → positive
   - Atıf yapılan kanun maddesi → query, o maddeyi yorumlayan kararlar → positive
   - ~2000 otomatik çift
2. **Manuel doğrulanmış altküme (gold):**
   - 100-150 sorguyu elle doğrula (relevance 0/1/2)
   - Bunlar "gold test set", gerisi "silver train/dev"
3. **Zorluk katmanları:** kolay (leksik eşleşme) / zor (parafraz/semantik)

Yayım: HuggingFace `datasets` formatında, BEIR-uyumlu (corpus.jsonl, queries.jsonl, qrels.tsv).

---

## 6. Takvim (Cuma gece teslim)

| Gün | İş |
|---|---|
| **Gün 1 (bugün)** | Master plan ✓, Colab notebook iskeleti, repo'yu Colab-ready yap, LaTeX iskeleti |
| **Gün 2** | TurkLegalBench oluştur (silver+gold), baseline matrisini Colab'de koştur |
| **Gün 3** | legal-e5-tr fine-tune (e5-base/large + hard negatives), reranker, ablation |
| **Gün 4** | Generative RAG + faithfulness, LLM vs RAG karşılaştırması, HF push |
| **Gün 5 (Cuma)** | Makaleyi sonuçlarla doldur, grafikler, sunum, son kontrol, teslim |

---

## 7. Risk Yönetimi

| Risk | Azaltma |
|---|---|
| Colab A100 kotası biter | Checkpoint'leri Drive'a kaydet, kaldığı yerden devam |
| Fine-tune iyileşme göstermez | Hard negatives + daha çok veri; en kötü e5-small sonuçları zaten elde |
| Generative LLM Türkçe zayıf | Birden çok dene (Trendyol, Llama-TR, Gemma-TR); olmazsa extractive'e düş |
| Benchmark gold etiketleme yavaş | 100 sorguyla başla, kalitesi yüksek olsun |
| Teslim yetişmez | Her gün sonunda "teslim edilebilir" durumda ol (incremental) |

---

## 8. Çıktı Dosyaları

```
project/
├── colab/
│   └── YargiRAG_full_pipeline.ipynb   # A100'de uçtan uca
├── src/                                # modüler kod (Colab'den import)
│   ├── ... (mevcut)
│   ├── build_benchmark.py              # TurkLegalBench üretimi
│   ├── hard_negatives.py               # hard negative mining
│   ├── baselines.py                    # çoklu model değerlendirme
│   ├── generative_rag.py               # LLM + faithfulness
│   ├── stats.py                        # bootstrap, t-test
│   └── hf_push.py                      # HuggingFace yayını
├── paper/
│   ├── yargirag.tex                    # IEEE Türkçe
│   └── figures/
├── MASTER_PLAN.md                      # bu dosya
├── SONUCLAR.md
└── README.md
```
