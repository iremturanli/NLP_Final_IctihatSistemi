"""Manuel doğrulanmış mini benchmark + retrieval metrik hesaplama.

Strateji:
  - 15 manuel sorgu × her birinde ground truth karar ID/anahtar listesi
  - Ground truth: kararın "doğru olduğunu" anlamlı anahtar terimle eşleştir
    (örn. sorgu "ihtiyati haciz icra takibi" → karar metninde bu terim geçmeli)
  - Pseudo-judgment olarak relevance: karar metni bu anahtar terimleri içeriyorsa relevant=1

Metrikler:
  - Recall@K: ilk K'da en az 1 ilgili karar var mı (binary)
  - MRR (Mean Reciprocal Rank): ilk ilgili kararın ranskinin tersi, ortalama
  - nDCG@K: kademeli skorlama
"""
import math
import re
from collections import defaultdict


# 15 manuel sorgu — her biri için (relevance_keywords) ve must_have anahtarlar
# Bir karar relevant sayılır eğer karar metninde:
#   - tüm "must_all" terimler var, ya da
#   - en az bir "must_any" terim grubu sağlanıyor
BENCHMARK_QUERIES = [
    {
        "id": "q01", "kategori": "icra_iflas",
        "query": "İhtiyati haciz icra takibi sayılır mı?",
        "must_any": [["ihtiyati haciz", "icra"], ["ihtiyati haciz", "takip"]],
    },
    {
        "id": "q02", "kategori": "ceza",
        "query": "Karşılıksız yararlanma suçunda beraat hangi koşullarda mümkündür?",
        "must_any": [["karşılıksız yararlanma"], ["karşilıksız yararlanma"]],
    },
    {
        "id": "q03", "kategori": "idari",
        "query": "İdari yargıda yürütmenin durdurulması koşulları",
        "must_any": [["idari yargı"], ["yürütmenin durdurulması"], ["idari yargı", "görev"]],
    },
    {
        "id": "q04", "kategori": "ceza_leksik",
        "query": "4733 sayılı yasaya muhalefet suçunun unsurları",
        "must_any": [["4733 sayılı"], ["4733", "tütün"]],
    },
    {
        "id": "q05", "kategori": "aile",
        "query": "Boşanma davasında nafaka miktarı neye göre belirlenir?",
        "must_any": [["nafaka"], ["boşanma", "tazminat"]],
    },
    {
        "id": "q06", "kategori": "is_hukuku",
        "query": "İş akdinin haksız feshi tazminatı",
        "must_any": [["iş akdi", "fesih"], ["haksız fesih"], ["işe iade"], ["kıdem tazminatı"]],
    },
    {
        "id": "q07", "kategori": "icra_iflas",
        "query": "Konkordato mühleti süresince alacaklı haklarına etkisi",
        "must_any": [["konkordato"]],
    },
    {
        "id": "q08", "kategori": "leksik_madde",
        "query": "2004 sayılı İcra İflas Kanunu 264. madde",
        "must_any": [["2004 sayılı", "icra"], ["icra ve iflas kanunu"]],
    },
    {
        "id": "q09", "kategori": "ceza",
        "query": "Kasten öldürme suçunda tasarlama unsuru",
        "must_any": [["kasten öldürme", "tasarla"], ["kasten öldürme"], ["tasarlayarak öldürme"]],
    },
    {
        "id": "q10", "kategori": "borclar",
        "query": "Kira sözleşmesinde tahliye taahhüdü",
        "must_any": [["kira", "tahliye"], ["tahliye taahhüdü"]],
    },
    {
        "id": "q11", "kategori": "ceza_leksik",
        "query": "5607 sayılı kaçakçılıkla mücadele kanununa muhalefet",
        "must_any": [["5607"], ["kaçakçılık"]],
    },
    {
        "id": "q12", "kategori": "borclar",
        "query": "Trafik kazasından kaynaklı maddi ve manevi tazminat",
        "must_any": [["trafik kaza", "tazminat"], ["maddi tazminat", "kaza"], ["destekten yoksun kalma"]],
    },
    {
        "id": "q13", "kategori": "aile",
        "query": "Velayet ve çocuğun üstün yararı",
        "must_any": [["velayet"], ["çocuğun üstün"]],
    },
    {
        "id": "q14", "kategori": "ceza",
        "query": "Hırsızlık suçunda etkin pişmanlık",
        "must_any": [["hırsızlık", "etkin pişmanlık"], ["etkin pişmanlık"]],
    },
    {
        "id": "q15", "kategori": "ticari",
        "query": "Faturanın delil niteliği ve ispat yükü",
        "must_any": [["fatura", "delil"], ["fatura", "ispat"]],
    },
]


def normalize(text):
    """Türkçe metni karşılaştırma için normalize et."""
    t = text.lower()
    t = re.sub(r'[\.,;:!\?"\'\(\)\[\]/—–\-]', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    return t


def is_relevant(query_spec, doc_text):
    """Bir kararın belirli sorguya 'relevant' olup olmadığını döndür."""
    doc_norm = normalize(doc_text)
    for group in query_spec["must_any"]:
        if all(normalize(term) in doc_norm for term in group):
            return True
    return False


def evaluate_results(query_spec, ranked_results, k_values=(1, 3, 5, 10)):
    """Bir sorgunun sonuçlarını değerlendir.

    ranked_results: list of dict (her birinde 'text' var)
    Döndürür: dict (recall@k, mrr, ndcg@k)
    """
    relevance = [is_relevant(query_spec, r["text"]) for r in ranked_results]
    out = {}
    for k in k_values:
        topk = relevance[:k]
        out[f"recall@{k}"] = int(any(topk))
    # MRR
    rr = 0.0
    for i, rel in enumerate(relevance, 1):
        if rel:
            rr = 1.0 / i
            break
    out["mrr"] = rr
    # nDCG@K (binary relevance)
    def dcg(rels):
        return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))
    for k in k_values:
        topk = [int(r) for r in relevance[:k]]
        ideal = sorted(topk, reverse=True)
        idcg = dcg(ideal) or 1.0
        out[f"ndcg@{k}"] = dcg(topk) / idcg
    return out


def aggregate(per_query_metrics):
    """Sorgu başına metrikleri ortalamasını al."""
    agg = defaultdict(float)
    for m in per_query_metrics:
        for k, v in m.items():
            agg[k] += v
    n = len(per_query_metrics) or 1
    return {k: v / n for k, v in agg.items()}


if __name__ == "__main__":
    print(f"Benchmark sorgu sayısı: {len(BENCHMARK_QUERIES)}")
    for q in BENCHMARK_QUERIES:
        print(f"  [{q['id']}] ({q['kategori']}) {q['query'][:60]}")
