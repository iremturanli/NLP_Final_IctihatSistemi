# YargıRAG — Sunum Metni

> Yüksek lisans sunumu. Her slayt için: **[Slaytta ne yazacak]** + 🗣️ **[Ne söyleyeceksin]**
> Toplam ~12 slayt, ~10-12 dakika anlatım + 3-5 dakika demo/soru.

---

## Slayt 1 — Başlık

**Slaytta:**
- YargıRAG: Türkçe Hukuki Bilgi Erişimi için Açık Kıyaslama Kümesi, Alana Uyarlanmış Model ve Avukat Asistanı
- İrem Turanlı · [Üniversite] · Yüksek Lisans NLP Projesi

🗣️ "Merhaba, bugün Türkçe hukuk alanında geliştirdiğim bir bilgi erişim sistemini anlatacağım: YargıRAG."

---

## Slayt 2 — Problem

**Slaytta:**
- ⚖️ Avukatlar emsal karar ararken binlerce karar arasında kayboluyor
- 🤖 Genel yapay zeka (ChatGPT) kaynak göstermeden, bazen **uydurarak** cevap veriyor
- 📰 ABD'de avukat, ChatGPT'nin uydurduğu sahte kararları mahkemeye sundu → ceza aldı (Mata v. Avianca, 2023)

🗣️ "Hukuk araştırmasında en büyük sorun emsal karara hızlı ulaşmak. Genel yapay zeka burada riskli — kaynak göstermeden, hatta var olmayan kararlar uydurarak cevap veriyor. ABD'de bir avukat ChatGPT'nin uydurduğu sahte kararları mahkemeye sunduğu için ceza aldı. Hukukta bu kabul edilemez."

---

## Slayt 3 — Boşluk (neden Türkçe?)

**Slaytta:**
- İngilizce için var: Legal-BERT, LegalBench, LeCaRD
- Türkçe için YOK:
  - ❌ Hukuka uyarlanmış açık arama (retrieval) modeli
  - ❌ Standart değerlendirme (benchmark) seti
- → İlerlemeyi ölçmek bile mümkün değil

🗣️ "İngilizce hukuk NLP'sinde olgun modeller ve benchmark'lar var. Ama Türkçe için ne arama modeli ne de bir ölçüm seti var. Yani bir sistem yapsanız bile 'ne kadar iyi?' sorusunu cevaplayamıyorsunuz. Ben bu iki boşluğu da doldurdum."

---

## Slayt 4 — Katkılarım (3 madde)

**Slaytta:**
1. **TurkLegalBench** — Türkçe hukuk için ilk açık benchmark
2. **legal-e5-tr** — hukuka uyarlanmış açık arama modeli
3. **YargıRAG** — kaynak gösteren, "bilmiyorum" diyebilen avukat asistanı

🗣️ "Üç katkım var: Birincisi, bir ölçüm seti. İkincisi, hukuka özel bir arama modeli. Üçüncüsü, bunları birleştiren, kaynak gösteren bir asistan. Sırayla anlatayım."

---

## Slayt 5 — Veri

**Slaytta:**
- 📚 Kaynak: `Turkish-Law-Documents-700k-clustered` (HuggingFace, **açık erişim**)
- 702.000 Yargıtay + Danıştay kararı
- Bu çalışmada: **15.000 karar** (hesaplama kısıtı; temsili örneklem)
- Ön işleme: HTML/markdown temizleme → anlamlı pasajlara bölme → gizli/bozuk sorgu filtreleme

🗣️ "Veri olarak HuggingFace'te açık erişimli 702 bin Yargıtay ve Danıştay kararını kullandım — gizlilik sorunu yok. Tüm veriyi işlemek yüksek hesaplama gerektirdiği için temsili bir örneklem olarak 15 binini aldım; tam veriye ölçekleme bir sonraki adım. Kararları temizleyip anlamlı parçalara böldüm."

---

## Slayt 6 — Katkı 1: TurkLegalBench

**Slaytta:**
- BEIR-uyumlu format (uluslararası standart)
- **Sızıntısız (leakage-free)** bölme: karar bazında train/dev/test
- → Model test kararlarını eğitimde GÖRMEZ (adil değerlendirme)
- 15.000 belge, 5.369 sorgu

🗣️ "İlk katkım, bir benchmark. En kritik nokta: bölmeyi karar bazında yaptım. Yani bir kararın hem eğitimde hem testte olması imkânsız. Bu, modelin 'kopya çekmesini' engelliyor — sonuçların adil olmasını garanti ediyor. Çoğu çalışmanın atladığı bir titizlik."

---

## Slayt 7 — Katkı 2: legal-e5-tr (model)

**Slaytta:**
- Temel: `multilingual-e5-base` (genel amaçlı)
- Yöntem: Türkçe hukuk metinleriyle ince ayar (fine-tuning)
  - Çoklu Olumsuz Sıralama Kaybı (MNRL)
  - Zor olumsuz örnek madenciliği (sadece eğitim verisinden)
- → Hukuk diline uyarlanmış arama modeli

🗣️ "İkinci katkım model. Genel bir çok dilli modeli aldım ve Türkçe hukuk metinleriyle yeniden eğittim. Modelin 'kıdem tazminatı' ile 'iş akdi feshi' gibi hukuki kavramları yakınlaştırmasını öğrettim. Özellikle 'zor örnekler' tekniğiyle, birbirine benzeyen ama farklı kararları ayırt etmesini sağladım."

---

## Slayt 8 — Katkı 3: YargıRAG sistemi

**Slaytta:**
- 🔍 Hibrit arama: anlamsal (model) + kelime (BM25) birleşik
- 📄 Kaynak gösterimi: daire, esas/karar no, tarih, ilgili pasaj
- 🛑 Abstention: emin değilse "yeterli emsal bulamadım" der
- ⚠️ LLM ile cevap **üretmez** → uydurma yok (güvenli)

🗣️ "Üçüncüsü, sistem. İki arama yöntemini birleştiriyorum: anlam bazlı ve kelime bazlı. Sonuçları kaynak göstererek sunuyorum — hangi daire, hangi karar numarası. Emin olmadığında 'bulamadım' diyebiliyor. Önemli: sistem kendi cümlesini yazıp uydurmuyor, gerçek kararları getiriyor. Bir kütüphaneci gibi — doğru sayfaları açıp veriyor, yorum katmıyor."

---

## Slayt 9 — Sonuçlar: Modelim herkesi geçti

**Slaytta (tablo):**
| Model | nDCG@10 |
|---|---|
| BERTurk-Legal (mevcut hukuk modeli) | 0.095 |
| mE5-base (temel model) | 0.206 |
| BGE-m3 (dünyanın en iyisi, 5× büyük) | 0.284 |
| BM25 | 0.322 |
| **legal-e5-tr (benim, 110M)** | **0.495** |

- Temel modeli **%140** artırdı · **p<0.001** (istatistiksel anlamlı)

🗣️ "Sonuçlara gelince: modelim, 8 farklı modeli geçti. En çarpıcısı: 5 kat daha büyük, dünyanın en iyi açık modeli BGE-m3'ü bile geçtim. Yani 'küçük ama alana özel model, büyük ama genel modeli yener'. Bu fark istatistiksel olarak da kanıtlı."

**[Düşük sayı sorusu gelirse]:** "Mutlak değerler düşük görünebilir ama bu görevin zorluğundan — dünyanın en iyi modeli bile 0.28 alıyor. Önemli olan aynı zorlukta herkesi geçmem."

---

## Slayt 10 — Sonuçlar: Gerçek sorularda %80

**Slaytta:**
- Otomatik test zorlu/yapay → düşük görünüyor
- **30 gerçek hukuki soruyla** test ettim:

| | Otomatik | **Gerçek sorular** |
|---|---|---|
| İlk denemede doğru (R@1) | 0.35 | **0.80** |
| İlk 5'te doğru (R@5) | 0.58 | **0.90** |

🗣️ "Otomatik test biraz yapay olduğu için düşük çıkıyordu. Bunu kapatmak için 30 gerçek hukuki soruyla test ettim — 'kıdem tazminatı nasıl hesaplanır' gibi. Gerçek sorularda sistem ilk denemede %80, ilk 5 sonuçta %90 doğru emsali buluyor. Yani gerçek kullanımda çok başarılı."

---

## Slayt 11 — Canlı Demo

**Slaytta:**
- 🖥️ YargıRAG arayüzü (ekran görüntüsü veya canlı)
- Örnek: "Boşanmada velayet çocuğun üstün yararı"

🗣️ "Şimdi canlı göstereyim." [Demo'da 2-3 soru sor: velayet, kıdem tazminatı, bir de alakasız soru — abstention'ı göster]
- "Görüyorsunuz: doğru daire geliyor, kaynak gösteriliyor, ilgili kısımlar vurgulanıyor."
- "Alakasız soru sorduğumda 'emsal bulamadım' diyor — uydurmuyor."

---

## Slayt 12 — Sınırlamalar (dürüst) + Gelecek

**Slaytta:**
- Sınırlamalar:
  - Benchmark yarı-otomatik (gold seti insan-hukukçu doğrulaması gelecek iş)
  - Sistem extractive — LLM ile cevap üretme henüz entegre değil
- Gelecek iş:
  - Tam 702k karara ölçekleme
  - Üretici (generative) cevap katmanı + faithfulness ölçümü
  - Modeli ve benchmark'ı HuggingFace'te yayımlama

🗣️ "Dürüst olmak gerekirse iki sınırlamam var: benchmark'ın bir kısmı otomatik üretildi, tam hukukçu doğrulaması gelecek iş. Ve sistem şu an kaynak gösteriyor ama LLM ile cevap yazmıyor — onu da bilinçli olarak güvenlik için yapmadım. Sonraki adımlar: tam veriye ölçekleme ve güvenli bir cevap üretme katmanı."

---

## Slayt 13 — Sonuç

**Slaytta:**
- ✅ TurkLegalBench: ilk açık Türkçe hukuk benchmark
- ✅ legal-e5-tr: SOTA'yı geçen, alana uyarlanmış model
- ✅ YargıRAG: kaynak gösteren, güvenli avukat asistanı
- 📂 Kod açık: github.com/iremturanli/NLP_Final_IctihatSistemi

🗣️ "Özetle: Türkçe hukuk için ilk açık benchmark'ı, SOTA'yı geçen bir model, ve güvenli bir avukat asistanı geliştirdim. Hepsi açık kaynak. Teşekkürler, sorularınızı alabilirim."

---

## 🎤 OLASI SORULAR — hazır cevaplar

**S: Sonuçlar neden bu kadar düşük?**
> Hukuki erişim zorlu bir görev — dünyanın en iyi modeli BGE-m3 bile bu kurulumda 0.28 alıyor. Mutlak sayı kurulumun zorluğundan; önemli olan aynı zorlukta herkesi geçmem. Gerçek sorularla test ettiğimde %80 başarı aldım.

**S: Neden sadece 15 bin karar?**
> 702 binin tamamını işlemek yüksek hesaplama ve bellek gerektiriyor. Temsili bir örneklem olarak 15 bin seçtim — fine-tune farkını ve karşılaştırmayı göstermek için yeterli. Tam veriye ölçekleme gelecek iş.

**S: Bu gerçek bir RAG mi? Cevap üretiyor mu?**
> Sistem retrieval (getirme) + kaynak gösterme yapıyor; LLM ile cevap üretme kısmını bilinçli olarak yapmadım çünkü hukukta uydurma riski var. Yani "güvenli kütüphaneci" gibi — kaynakları getiriyor, yorum katmıyor. Üretici katman gelecek iş.

**S: Leakage (veri sızıntısı) yok mu, emin misin?**
> Evet, eminim. Bölmeyi karar bazında yaptım ve assert ile doğruladım — bir kararın hem eğitimde hem testte olması imkânsız. Fine-tune sadece eğitim bölmesinde, değerlendirme sadece test bölmesinde.

**S: Benchmark'ı nasıl oluşturdun, güvenilir mi?**
> Otomatik (silver) üretim + 30 soruluk gerçek (gold) doğrulama. Silver'ın bir sınırlaması var — sorgular kararların başlığından geliyor. Bunu fark ettim ve gerçek sorularla bir gold set ekleyerek kapattım. Tam hukukçu doğrulaması bir sonraki adım.

**S: Mevcut Türkçe hukuk modeli (BERTurk-Legal) ile farkın ne?**
> BERTurk-Legal sınıflandırma için eğitilmiş, arama için değil. Benim modelim arama için özel eğitildi. Sonuç: BERTurk-Legal 0.095, benimki 0.495 — beş kattan fazla fark.

**S: Pratik faydası ne?**
> Bir avukat doğal dille soru soruyor, sistem ilgili emsal kararları kaynak göstererek getiriyor — uydurma riski olmadan. Mevcut anahtar-kelime aramalarından çok daha akıllı, ChatGPT'den çok daha güvenli.
