# Sunum Konuşma Metni (Hocaya Anlatım)

> Toplam süre: ~7-9 dakika anlatım + 3-5 dakika canlı demo.

---

## 1. Proje Tanıtımı (1 dakika)

> Hocam, final projem olarak avukatların ve hukuk öğrencilerinin **içtihat araması yapabileceği bir RAG sistemi** geliştirdim. Sistemin adı **LegalRAG-TR**.
>
> Kullanıcı bir hukuki soru sorduğunda, sistem ona ilgili Yargıtay ve Danıştay kararlarını **kaynak göstererek** sunuyor — yani her cevap için hangi karara dayandığını esas/karar numarası ile söylüyor. Bu sayede "halüsinasyon" — yani modelin uydurma karar üretmesi — riski sıfırda.
>
> Sistemin üç akademik katkısı var:
> 1. Türkçe hukuk için **domain-adaptive fine-tuned bir embedding modeli**
> 2. 15 sorgudan oluşan **manuel doğrulanmış mini benchmark** + standart retrieval metrikleri
> 3. **Halüsinasyon-kontrollü** uçtan uca bir RAG mimarisi (hibrit retrieval + extractive cevap)

---

## 2. Veri Seti (1 dakika)

> Veri olarak **HuggingFace'te açık erişimli** `erdem-erdem/Turkish-Law-Documents-700k-clustered` veri setini kullandım. Bu veri seti:
> - 702 bin Türkçe hukuk kararı
> - Yargıtay (Hukuk + Ceza Daireleri) ve Danıştay kararları birleşik
> - Tamamen açık erişim — herhangi bir gizlilik veya lisans sorunu yok
>
> Demo için bu 700 binden **2.000 karar örnekledim** çünkü demo donanımım dizüstü RTX 4050 (6 GB). Makale aşamasında tam veriye ölçeklendireceğim.

---

## 3. Sistem Mimarisi (2 dakika)

> Sistem dört ana bileşenden oluşuyor.
>
> **(1) Hibrit Retrieval.** Hukuki metinler iki farklı arama stratejisi gerektiriyor:
> - **Anlamsal arama:** `multilingual-e5` ailesi embedding modeli ile karar metinlerini 384-768 boyutlu vektörlere dönüştürüyorum, FAISS ile indeksliyorum. "İhtiyati haciz" ile "geçici el koyma" gibi parafrazları yakalamak için kritik.
> - **Leksik arama (BM25):** "2004 sayılı Kanun Madde 264" gibi nadir teknik terimleri yakalamak için. Bunlar anlamsal aramada kaybolabiliyor.
> - İki sıralamayı **Reciprocal Rank Fusion** ile birleştiriyorum.
>
> **(2) Domain-Adaptive Fine-Tuning.** Bu projenin asıl akademik katkısı.
> - `multilingual-e5-small` modelini Türkçe hukuk metinleri üzerinde **Multiple Negatives Ranking Loss** ile fine-tune ettim.
> - Eğitim verisi olarak 2.313 (query, passage) çiftini zayıf-denetimli olarak ürettim: karar metnindeki anahtar konu (örneğin "karşılıksız yararlanma") query, kararın gövdesi passage.
> - RTX 4050 üzerinde fp16 mixed precision ile 33 saniyede eğitim tamamlandı.
>
> **(3) Cross-Encoder Reranker.** İlk 15 sonucu BAAI/bge-reranker-v2-m3 ile yeniden sıralıyorum. Sonuçlarımızda bunun bu konfig'de yardımcı olmadığını da göreceğiz — bu da kendi başına bir bulgu.
>
> **(4) Kaynaklı Cevap Üretimi.** Şu anki sürümde **generative LLM yok** — extractive yaklaşım. Her sonuç gerçek bir karara işaret ediyor, esas/karar numarası ile doğrulanabilir. Halüsinasyon riski sıfır.

---

## 4. Deneysel Sonuçlar (2 dakika) — ANA SUNUM NOKTASI

> 15 manuel sorgudan oluşan mini benchmark üzerinde 5 farklı sistem konfigürasyonunu test ettim. Metrikler: Recall@K, MRR, nDCG@K.

> *[burada SONUCLAR.md'deki tabloyu göster veya slayda yansıt]*

| Sistem | Recall@1 | MRR | nDCG@10 | Latency |
|---|---:|---:|---:|---:|
| e5-small baseline | 0.733 | 0.833 | 0.836 | 22ms |
| **e5-small + FT** | **0.933** | **0.933** | **0.887** | 15ms |
| e5-base baseline | 0.800 | 0.833 | 0.811 | 17ms |
| e5-small+FT+reranker | 0.733 | 0.833 | 0.851 | 554ms |
| e5-base+reranker | 0.733 | 0.822 | 0.851 | 539ms |

> Üç önemli bulgu:
>
> **Bulgu 1: Fine-tuning Recall@1'i %27 göreceli artırdı.** Yani ilk doğru kararı ilk sıraya getirme oranı 0.733'ten 0.933'e çıktı. MRR de aynı şekilde 0.833'ten 0.933'e. Bu, generic multilingual modelin Türkçe hukuk için kalibre edilmesinin **kanıtlanmış faydası**.
>
> **Bulgu 2: Fine-tuned küçük model, generic büyük modeli geçti.** e5-base baselineı (110M parametre) Recall@1 = 0.800, fine-tuned e5-small (30M parametre) Recall@1 = **0.933**. Yani 3.7x daha küçük bir model, domain adaptation ile daha iyi sonuç veriyor. Bu hem akademik hem pratik açıdan önemli: küçük modeller mobil cihazda bile çalışır, ucuz inference yapar, ama domain'e adapte edilirse büyük modelle yarışabilir.
>
> **Bulgu 3: Cross-encoder reranker bu konfig'de yardımcı olmadı.** Hatta bazı metriklerde sonuçlar düştü, latency 30 kat arttı. Olası neden: BAAI/bge-reranker'ın Türkçe ağırlığı sınırlı olabilir. Bu da sonraki adımın net işareti: **Türkçe hukuk için reranker fine-tuning**.

---

## 5. Mimari Detaylar (1 dakika, soru gelirse)

> **"Neden multilingual-e5?"**
> 100+ dilde retrieval için eğitilmiş, asimetrik retrieval (query/passage prefix) destekliyor. Türkçe'de açık alternatifi yok.
>
> **"Neden BM25 ve dense'i birleştirdin?"**
> Sadece dense ile "Madde 264" gibi spesifik terimleri kaçırırdım. BM25 nadir leksikalları yakalıyor, dense parafrazları yakalıyor — RRF eşit ağırlıkla birleştiriyor, hyperparameter tuning gerektirmiyor.
>
> **"Weak supervision güvenli mi?"**
> Evet, çünkü MNRL in-batch negatives kullanıyor: batch içindeki diğer pozitifler negatif olarak kullanılıyor. Bu, manuel negatif seçimine ihtiyaç kalmadan model parafraza dirençli hale geliyor.
>
> **"Cevaplar neden generative değil?"**
> Hukuk alanında yanlış cevap = ciddi risk. Extractive yaklaşımıyla halüsinasyon sıfırda. Sonraki iş: Türkçe LLM (Trendyol-LLM, Kanarya) ile faithfulness-controlled özetleme.

---

## 6. Canlı Demo (2-3 dakika)

> Şimdi sistemin canlı çalıştığını göstereyim.
>
> *[demo.ipynb'i aç ya da terminalden `python src/app.py` ile Gradio başlat]*
>
> Farklı sorgu türleri:
>
> 1. **Anlamsal:** "İhtiyati haciz icra takibi sayılır mı?" → fine-tune'un parafraz yakalamasını göster
> 2. **Leksik:** "2004 sayılı İcra İflas Kanunu 264. madde" → BM25'in spesifik terim yakalamasını göster
> 3. **İdari yargı:** "İdari yargıda yürütmenin durdurulması" → Danıştay kararlarının gelmesini göster
> 4. **Ceza:** "Karşılıksız yararlanma suçunda beraat" → 17. Ceza Dairesi sonucunu göster (Recall@1 ✓)

---

## 7. Sonraki Adımlar (30 saniye)

> 1. **Tam 700k karara ölçeklenme** (FAISS HNSW veya IVF-PQ ANN)
> 2. **Türkçe için reranker fine-tuning** (bu projenin net bulgusu)
> 3. **Açık Türkçe hukuk retrieval benchmarkı** — 200-500 elle doğrulanmış sorgu-karar çifti, Türkçe için ilk açık benchmark
> 4. **Generative cevap özetleme** — Türkçe LLM ile faithfulness + abstention kontrollü
> 5. **e5-base + LoRA fine-tuning** — büyük model için bellek dostu adaptation

---

## Olası Soruların Cevapları

**S: 33 saniyede gerçek bir fine-tuning oldu mu?**
> Evet, 2.313 pair × 2 epoch × batch 32 = 144 step × ~0.23 saniye. RTX 4050 GPU üzerinde fp16 precision ile. Loss 3.04'tan 1.97'ye düştü, açık öğrenme sinyali. Bench'te Recall@1 %20 göreceli arttı.

**S: 15 sorgu az değil mi?**
> Evet, demo benchmark. Final makalede 200-500 sorguya genişleteceğim, elle annotated. Şu an konsept kanıtı.

**S: GPU 6GB ile e5-base eğitemedin mi?**
> Evet, e5-base + Adam optimizer state 6GB'ı taşıdı. Çözümler: (1) bitsandbytes 8-bit optimizer, (2) LoRA adapter, (3) gradient checkpointing. Makale aşamasında LoRA ile e5-base eğiteceğim, RTX 4050'de 6GB'a sığar.

**S: Reranker neden işe yaramadı?**
> Mini benchmark'ta varyans olabilir. Daha kesin neden: BAAI/bge-reranker-v2-m3 multilingual ama eğitiminde Türkçe oranı düşük. Türkçe hukuk için cross-encoder fine-tuning, sonraki adım.

**S: Veri açık kaynak mı?**
> Evet. HuggingFace `erdem-erdem/Turkish-Law-Documents-700k-clustered`. Türk devlet kurumlarının kamuya açık karar arşivlerinden derlenmiş. Tezde tam atıf veriyorum.

**S: Halüsinasyonu nasıl önlüyorsun?**
> Şu an extractive — LLM yok. Her sonuç gerçek bir karardan, esas/karar no ile doğrulanabilir. LLM eklediğimde: (1) retrieval-grounded prompt, (2) faithfulness scoring (claim-level NLI), (3) düşük confidence'ta abstention.
