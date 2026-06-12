"""Gold (altın) değerlendirme seti — gerçek, doğal hukuki sorularla içerik-bazlı doğrulama.

Silver benchmark'ın dairesellik sınırlamasını kapatır:
  - Sorgular GERÇEK avukat sorularıdır (kararların başlığından türetilmez)
  - İlgililik, kararın İÇERİĞİNE bakılarak otomatik-destekli işaretlenir
    (anahtar kavram örtüşmesi + daire uygunluğu)

Yöntem: Her gerçek soru için sistem top-K karar getirir. Bir karar "ilgili" sayılır
eğer sorunun temsil ettiği hukuki kavram/terimler kararın gövdesinde yer alıyorsa.
Bu, silver'daki "başlık=sorgu" dairesinden farklı olarak gerçek soru-içerik eşleşmesidir.
"""

# 30 gerçek, doğal hukuki soru + ilgililik için içerik anahtarları (uzman tanımı)
# rel_terms: kararın gövdesinde bu terim GRUPLARINDAN biri tam geçiyorsa karar ilgilidir.
GOLD_QUERIES = [
    {"q": "İş sözleşmesinin haklı nedenle feshinde kıdem tazminatı ödenir mi?",
     "rel": [["kıdem tazminatı"], ["kıdem", "fesih"]]},
    {"q": "İşçinin haklı nedenle iş sözleşmesini feshi hangi koşullarda mümkündür?",
     "rel": [["haklı neden", "fesih"], ["iş sözleşmesi", "fesih"]]},
    {"q": "İşe iade davasında işverenin ispat yükü nedir?",
     "rel": [["işe iade"], ["feshin geçersizliği"]]},
    {"q": "Boşanmada velayet belirlenirken hangi ölçütler dikkate alınır?",
     "rel": [["velayet"], ["çocuğun üstün yarar"]]},
    {"q": "Boşanma davasında yoksulluk nafakası şartları nelerdir?",
     "rel": [["yoksulluk nafakası"], ["nafaka"]]},
    {"q": "Anlaşmalı boşanmada mahkemenin denetim yetkisi var mıdır?",
     "rel": [["anlaşmalı boşanma"], ["boşanma", "protokol"]]},
    {"q": "Mal rejiminin tasfiyesinde katılma alacağı nasıl hesaplanır?",
     "rel": [["katılma alacağı"], ["mal rejimi", "tasfiye"]]},
    {"q": "İhtiyati haciz kararına itiraz edilebilir mi?",
     "rel": [["ihtiyati haciz"], ["ihtiyati", "haciz"]]},
    {"q": "İcra takibinde istihkak iddiası ne zaman ileri sürülür?",
     "rel": [["istihkak"], ["haciz", "istihkak"]]},
    {"q": "Konkordato mühletinin alacaklılar üzerindeki etkisi nedir?",
     "rel": [["konkordato"]]},
    {"q": "Karşılıksız çek keşide etme suçunun unsurları nelerdir?",
     "rel": [["karşılıksız", "çek"], ["çek", "keşide"]]},
    {"q": "Kasten öldürme suçunda tasarlama unsuru nasıl değerlendirilir?",
     "rel": [["tasarla", "öldür"], ["kasten öldürme"]]},
    {"q": "Hırsızlık suçunda etkin pişmanlık hükümleri nasıl uygulanır?",
     "rel": [["hırsızlık", "etkin pişmanlık"], ["etkin pişmanlık"]]},
    {"q": "Dolandırıcılık suçunda nitelikli hal hangi durumlarda oluşur?",
     "rel": [["dolandırıcılık"]]},
    {"q": "Uyuşturucu ticareti suçunda cezayı artıran nedenler nelerdir?",
     "rel": [["uyuşturucu"], ["uyuşturucu madde ticareti"]]},
    {"q": "Görevi kötüye kullanma suçu hangi koşullarda oluşur?",
     "rel": [["görevi kötüye kullanma"], ["görevi kötüye"]]},
    {"q": "Trafik kazasından kaynaklanan destekten yoksun kalma tazminatı nasıl belirlenir?",
     "rel": [["destekten yoksun"], ["trafik", "tazminat"]]},
    {"q": "Haksız fiilde manevi tazminatın takdirinde hangi kriterler esastır?",
     "rel": [["manevi tazminat"]]},
    {"q": "Kira sözleşmesinde tahliye taahhüdü hangi şartlarda geçerlidir?",
     "rel": [["tahliye taahhüd"], ["kira", "tahliye"]]},
    {"q": "Kira bedelinin tespiti davası hangi durumlarda açılır?",
     "rel": [["kira", "tespit"], ["kira bedeli"]]},
    {"q": "Eser sözleşmesinde ayıplı ifa halinde iş sahibinin hakları nelerdir?",
     "rel": [["eser sözleşmesi"], ["ayıp", "eser"]]},
    {"q": "Tüketici hakem heyeti kararlarına itiraz nasıl yapılır?",
     "rel": [["tüketici"], ["hakem heyeti"]]},
    {"q": "Faturanın ticari defterlerde delil değeri nedir?",
     "rel": [["fatura", "delil"], ["fatura", "defter"]]},
    {"q": "Ticari işletmenin devrinde sorumluluğun kapsamı nedir?",
     "rel": [["işletme devri"], ["ticari işletme", "devir"]]},
    {"q": "Tapu iptali ve tescil davasında husumet kime yöneltilir?",
     "rel": [["tapu iptali"], ["tescil", "tapu"]]},
    {"q": "Kadastro tespitine itiraz davasında zilyetlik nasıl ispatlanır?",
     "rel": [["kadastro"], ["zilyetlik"]]},
    {"q": "Miras taksim sözleşmesinin şekil şartı nedir?",
     "rel": [["miras", "taksim"], ["taksim sözleşmesi"]]},
    {"q": "Vasiyetnamenin iptali hangi sebeplerle istenebilir?",
     "rel": [["vasiyetname"], ["vasiyet"]]},
    {"q": "İdari işlemin iptali için menfaat ihlali şartı nedir?",
     "rel": [["idari işlem", "iptal"], ["menfaat ihlal"]]},
    {"q": "Memur disiplin cezalarına karşı yargı yolu nasıl işler?",
     "rel": [["disiplin ceza"], ["memur", "disiplin"]]},
]


def _norm(s):
    s = s.lower().replace("ı", "i").replace("İ", "i").replace("â", "a")
    return s


def judge_relevance(query_spec, doc_text):
    """Kararın içeriğine bakarak ilgililik (0/1/2) ver.

    İçerik-bazlı: sorunun temsil ettiği kavram grubu kararın gövdesinde geçiyorsa ilgili.
    2 = ana kavram grubu (ilk grup) tam karşılanıyor; 1 = yedek grup; 0 = yok.
    """
    dn = _norm(doc_text)
    for gi, group in enumerate(query_spec["rel"]):
        if all(_norm(t) in dn for t in group):
            return 2 if gi == 0 else 1
    return 0


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import math
    from embedder import load_model, load_index
    from retriever import HybridRetriever

    INDEX = "artifacts/index_ft_splitclean_full"
    MODEL = "artifacts/legal-e5-tr-splitclean"
    index, emb, recs = load_index(INDEX)
    model = load_model(MODEL)
    retr = HybridRetriever(index, emb, recs, model)

    def dcg(rels):
        return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

    per_q = []
    print(f"{'Soru':<60} R@1 R@5 nDCG")
    print("-" * 80)
    for spec in GOLD_QUERIES:
        results = retr.search(spec["q"], top_k=10)
        rels = [judge_relevance(spec, r.get("text", "")) for r in results]
        recall1 = int(any(r > 0 for r in rels[:1]))
        recall5 = int(any(r > 0 for r in rels[:5]))
        ideal = sorted(rels, reverse=True)
        ndcg = dcg(rels) / (dcg(ideal) or 1.0)
        mrr = 0.0
        for i, r in enumerate(rels, 1):
            if r > 0:
                mrr = 1.0 / i
                break
        per_q.append({"recall@1": recall1, "recall@5": recall5, "ndcg@10": ndcg, "mrr@10": mrr})
        print(f"{spec['q'][:58]:<60} {recall1}   {recall5}   {ndcg:.2f}")

    n = len(per_q)
    print("-" * 80)
    for k in ["recall@1", "recall@5", "mrr@10", "ndcg@10"]:
        print(f"  Ortalama {k}: {sum(p[k] for p in per_q)/n:.3f}")
    print(f"\n  Toplam {n} gerçek hukuki soru (gold set)")
