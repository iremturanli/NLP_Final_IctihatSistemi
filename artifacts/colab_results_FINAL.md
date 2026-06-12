# YargıRAG — Colab A100 Sonuçları (LEAKAGE-FREE, makale için)

> 15.000 karar, TurkLegalBench. Karar bazında train/dev/test split (sızıntısız).
> Test: 1.500 doküman / 573 sorgu. Tüm değerlendirmeler TEST split'te.

## Ana Sonuç: Erişim Başarımı (leakage-free test)

| Model | Param | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | MAP |
|---|---|---:|---:|---:|---:|---:|---:|
| BERTurk (mean-pool) | 110M | 0.014 | 0.070 | 0.110 | 0.037 | 0.054 | 0.045 |
| BERTurk-Legal (mean-pool) | 110M | 0.031 | 0.122 | 0.182 | 0.069 | 0.095 | 0.080 |
| mE5-base (temel model) | 110M | 0.112 | 0.244 | 0.328 | 0.169 | 0.206 | 0.177 |
| mE5-small | 30M | 0.140 | 0.295 | 0.379 | 0.206 | 0.247 | 0.216 |
| mE5-large | 560M | 0.152 | 0.309 | — | 0.221 | 0.263 | — |
| BGE-m3 (SOTA) | 560M | 0.162 | 0.346 | — | 0.239 | 0.284 | — |
| BM25 | — | 0.185 | 0.394 | 0.478 | 0.273 | 0.322 | 0.283 |
| **legal-e5-tr (FT+hardneg)** | 110M | **0.353** | **0.576** | **0.649** | **0.446** | **0.495** | **0.453** |
| BM25 + legal-e5-tr (RRF) | — | 0.333 | 0.546 | 0.651 | 0.428 | 0.481 | 0.436 |

## Temel Bulgular

**BERTurk-Legal notu:** Mevcut Türkçe hukuk modeli BERTurk-Legal (nDCG@10=0.095), generic BERTurk'ü (0.054) geçse de, retrieval'a uyarlanmadığından generic e5 modellerinin bile gerisinde kalmıştır. Bu, alan ön-eğitiminin tek başına yetersiz olduğunu, asimetrik retrieval ince ayarının kritik olduğunu gösterir.

1. **legal-e5-tr tüm modelleri açık ara geçiyor**: nDCG@10 = 0.495
   - Temel modeli (mE5-base 0.206) → 0.495: **%140 göreceli artış**
   - 5x büyük SOTA modeli BGE-m3'ü (0.284) geçti — domain adaptation boyutu yendi
   - En güçlü klasik baseline BM25'i (0.322) %54 geçti

2. **Sızıntısız (leakage-free)**: karar bazında train/dev/test → fine-tune test kararlarını görmedi

3. **İstatistiksel anlamlılık** (paired bootstrap, test split):
   - legal-e5-tr vs mE5-base: ΔnDCG@10 = +0.244, **p<0.0001 ***
   - hybrid (RRF) vs BM25: ΔnDCG@10 = +0.158, **p<0.0001 ***

4. **Hybrid (RRF) bulgusu**: BM25+legal-e5-tr ile tek başına legal-e5-tr arasında anlamlı fark YOK (p>0.17). Fine-tuned dense, hibrit kadar iyi → fine-tune'un gücünü gösterir.

5. **Chunk boyutu ablation** (dense, chunked): chunk=900 en iyi (nDCG 0.516); 600-1500 arası robust (0.512-0.516).

6. **Error analysis**: Leksik sorgularda fine-tuned dense artık BM25'i geçiyor (66 dense-win vs 0 bm25-win) → FT leksik eşleşmeyi de öğrendi.

## Kalıcı Çıktılar (Drive)
- legal-e5-tr-splitclean (model)
- turklegalbench_{train,dev,test} (benchmark)
- index_ft_splitclean_full: 15k karar, 75.676 chunk FAISS indeks

## Gold Değerlendirme Seti (30 gerçek hukuki soru, içerik-bazlı)

> Dairesellik sınırlamasını kapatmak için: sorgular GERÇEK avukat sorularıdır
> (karar başlıklarından türetilmez), ilgililik kararların İÇERİĞİNE göre çoklu-ilgili işaretlenmiştir.

| Metrik | Silver (otomatik) | Gold (gerçek sorular) |
|---|---:|---:|
| Recall@1 | 0.353 | **0.800** |
| Recall@5 | 0.576 | **0.900** |
| MRR@10 | 0.446 | **0.842** |
| nDCG@10 | 0.495 | **0.825** |

Gerçek doğal sorularla sistem ilk denemede %80, ilk 5'te %90 doğru emsali getirmektedir.
Mutlak değerlerin silver'dan yüksek olması: (1) gerçek sorguların daha zengin sinyali,
(2) çoklu-ilgili etiketlemenin tek-ilgili kısıtını kaldırması.
