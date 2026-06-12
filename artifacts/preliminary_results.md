# Ön Sonuçlar (Local, TurkLegalBench v2 — düzeltilmiş)

> Local RTX 4050, 1500 karar corpus / 499 sorgu. Colab'de e5-base/large + 50k veri ile artar.

## Düzeltme Öncesi vs Sonrası (nDCG@10)
| Model | Bozuk (v1) | Düzeltilmiş (v2) |
|---|---:|---:|
| BM25 | 0.095 | 0.322 |
| mE5-small | 0.048 | 0.245 |

Düzeltmeler: (1) corpus pasajı boilerplate atlanıp 2000 char'a sınırlandı (e5 512-token
truncation adaleti), (2) generic tek-kelime sorgular filtrelendi, (3) ayırt edici sorgular.

## Baseline Matrisi (TurkLegalBench v2)
| Model | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | MAP |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.186 | 0.389 | 0.485 | 0.271 | 0.322 | 0.282 |
| mE5-small (generic) | 0.140 | 0.299 | 0.371 | 0.206 | 0.245 | 0.214 |
| **legal-e5-tr (FT)** | **0.246** | **0.453** | **0.559** | **0.335** | **0.388** | **0.345** |

## İstatistiksel Anlamlılık (legal-e5-tr vs mE5-small, paired bootstrap)
| Metrik | legal-e5-tr [%95 GA] | mE5-small [%95 GA] | Δ | p |
|---|---|---|---:|---|
| Recall@1 | 0.246 [0.208, 0.285] | 0.140 [0.110, 0.170] | +0.106 | <0.0001 *** |
| MRR@10 | 0.335 [0.299, 0.372] | 0.206 [0.176, 0.237] | +0.129 | <0.0001 *** |
| nDCG@10 | 0.388 [0.352, 0.424] | 0.245 [0.214, 0.278] | +0.143 | <0.0001 *** |

**Bulgular:**
- Fine-tune, generic e5-small'u nDCG@10'da %58 göreceli artırdı
- legal-e5-tr hem generic dense'i hem BM25'i geçti
- Güven aralıkları ayrık → fark istatistiksel olarak güçlü
