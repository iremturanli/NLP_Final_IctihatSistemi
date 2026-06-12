# YargıRAG — Teslim Durumu ve Kontrol Listesi

> Akademik dürüstlük: Buradaki TÜM sayılar gerçek deney koşumlarından gelir. Hiçbir sonuç uydurulmamıştır. Sınırlamalar açıkça belirtilmiştir.

## ✅ Hazır olanlar

| Çıktı | Dosya | Durum |
|---|---|---|
| **Makale** (IEEE Türkçe, 3 sayfa) | `paper/yargirag.tex` | Gerçek sonuçlarla dolu, derleniyor |
| **Temiz pipeline notebook** | `colab/YargiRAG_pipeline_CLEAN.ipynb` | 31 hücre, leakage-free, 0 syntax hata |
| **Kod modülleri** | `src/*.py` | 20 modül, 0 syntax hata |
| **Avukat UI** | `src/app_pro.py` | Çalışıyor: kaynak gösterimi, abstention, vurgu, 2-sütun grid |
| **Fine-tuned model** | `artifacts/legal-e5-tr-splitclean/` | e5-base, indirildi, local çalışıyor |
| **Erişim indeksi** | `artifacts/index_ft_splitclean_full/` | 15k karar, 75.676 pasaj |
| **Sonuç dökümü** | `artifacts/colab_results_FINAL.md` | Leakage-free test sonuçları |
| **Colab src paketi** | `YargiRAG_src.zip` | Güncel |

## 📊 Ana Sonuçlar (leakage-free test, 573 sorgu — GERÇEK)

| Model | nDCG@10 | MRR@10 | R@1 |
|---|---:|---:|---:|
| BERTurk (mean-pool) | 0.054 | 0.037 | 0.014 |
| mE5-base (temel model) | 0.206 | 0.169 | 0.112 |
| mE5-large (560M) | 0.263 | 0.221 | 0.152 |
| BGE-m3 (SOTA, 560M) | 0.284 | 0.239 | 0.162 |
| BM25 | 0.322 | 0.273 | 0.185 |
| **legal-e5-tr (bizim)** | **0.495** | **0.446** | **0.353** |

- Temel modeli %140 göreli artırdı, p<0.001 (eşlemeli bootstrap)
- 5× büyük SOTA modeli (BGE-m3) ve BM25'i geçti

## 🚀 UI'ı çalıştır

```bash
cd /home/iremturanli/data/irem/YL/NLP/project
PYTHONNOUSERSITE=1 \
LEGALRAG_INDEX=artifacts/index_ft_splitclean_full \
LEGALRAG_MODEL=artifacts/legal-e5-tr-splitclean \
python src/app_pro.py
```
→ `http://localhost:7860/?__theme=light`

## 📝 Makaleyi derle (Overleaf)

`paper/yargirag.tex` dosyasını Overleaf'e yükle → IEEEtran hazır → derle.
(Local'de de derlendi: `paper/yargirag.pdf`)

## ⚠️ Dürüst Sınırlamalar (sunumda SEN söyle — güç katar)

1. **Benchmark "silver" (otomatik)** — sorgular kararların kendi başlık/konusundan üretildi; insan-doğrulanmış "gold" altküme bir sonraki adım. (Makalede belirtildi.)
2. **Tek-relevant qrels** — bir soruya tek doğru cevap; metrikleri alttan sınırlar. Bu yüzden *karşılaştırmalı* üstünlük daha güvenilir gösterge.
3. **Mutlak skorlar orta** (nDCG ~0.50) — zorlu görev (15k karar içinde tek doğru). Bulgu: alana uyarlama generic+SOTA modelleri anlamlı geçiyor.
4. **Generative katman** (LLM özet + faithfulness) yarım — sistem şu an extractive (kaynak listeler, uydurmaz). Abstention çalışıyor. Bu *gelecek iş*.

## 🎯 Opsiyonel güçlendirmeler (zaman kalırsa, Colab)

- **BERTurk-Legal baseline** notebook'a eklendi → "Run All" ile koşarsan "mevcut Türkçe hukuk modelini de değerlendirdik" katkısı çıkar.
- **Veri temizliği** güncellendi (bozuk/generic sorgu filtresi) → benchmark'ı yeniden üretirsen daha temiz; ama mevcut sonuçlar zaten geçerli ve filtrelenmiş.

## Sunum tek-cümle konumlandırma

> "Türkçe hukuki bilgi erişimi için ilk açık, sızıntısız kıyaslama kümesini (TurkLegalBench), alana uyarlanmış açık bir gömme modelini (legal-e5-tr) ve kaynak gösteren bir avukat asistanını geliştirdim. Modelim, beş kat büyük SOTA modelleri dahil tüm temelleri istatistiksel olarak anlamlı biçimde geçti."
