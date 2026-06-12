"""TurkLegalBench — Türkçe hukuk retrieval için açık benchmark üretimi.

BEIR-uyumlu format üretir:
  corpus.jsonl   : {"_id", "title", "text"}        (aranabilir kararlar)
  queries.jsonl  : {"_id", "text"}                  (sorgular)
  qrels.tsv      : query-id  corpus-id  relevance   (ilgililik etiketleri)

Üretim stratejisi (hibrit, BEIR/NFCorpus geleneği):
  1. SILVER (otomatik, ölçek): kararın anahtar konusu/suçu → query,
     o kararın gövdesi → positive (relevance=1)
  2. GOLD (manuel doğrulama, kalite): bir altküme insan tarafından
     relevance 0/1/2 ile etiketlenir (ayrı dosyada)
  3. Zorluk katmanı: leksik (kolay) vs parafraz (zor) ayrımı

Bu modül SILVER seti otomatik üretir; GOLD için doğrulama şablonu çıkarır.
"""
import os
import re
import json
import random
from pathlib import Path
from collections import defaultdict


# Anlamsız/metadata sorguları ele
STOP_QUERY_TERMS = {
    "türk milleti adına", "içtihat metni", "yargıtay kararı",
    "karar verilmiştir", "türk milleti", "i̇çtihat metni",
}

# Tek başına ayırt edici OLMAYAN generic hukuk terimleri (corpus'ta yüzlerce eşleşir)
GENERIC_TERMS = {
    "beraat", "temyiz", "karar", "mahkeme", "mahkemesi", "dava", "hüküm", "bozma", "onama",
    "mahkumiyet", "ceza", "tazminat", "red", "kabul", "itiraz", "istinaf",
    "yargılama", "sanık", "davacı", "davalı", "katılan", "hukuk", "suç",
    "türk milleti adına", "gereği düşünüldü", "incelenen", "kararın", "yargıtay",
    "yargıtay kararı", "incelenen kararın",
}

# Karar metninin asıl içeriğe yaklaşık başlangıç işaretleri (boilerplate sonrası)
CONTENT_MARKERS = ["İçtihat Metni", "GEREĞİ DÜŞÜNÜLDÜ", "gereği düşünüldü",
                   "KARAR", "DAVA", "HUKUKÎ SÜREÇ"]


def _norm(s):
    # Türkçe ı/i ayrımını ve nokta-üstü i (i̇) artefaktını birleştir → tek 'i'
    s = s.replace("i̇", "i").replace("İ", "i").replace("I", "i").replace("ı", "i")
    s = s.replace("â", "a").replace("î", "i").replace("û", "u")
    return re.sub(r"\s+", " ", s.lower()).strip()


# Normalize edilmiş stop/generic kümeleri (ı/i, â vb. farklarından bağımsız eşleşme)
_STOP_NORM = None
_GENERIC_NORM = None


def _is_broken_spaced(s):
    """'y a r g i t a y' gibi harf-arası boşluklu bozuk metinleri yakala."""
    toks = s.split()
    if len(toks) >= 4:
        single = sum(1 for t in toks if len(t) == 1)
        if single / len(toks) >= 0.5:   # tokenların yarısı tek harf → bozuk
            return True
    return False


def informative_passage(text, max_chars=2000):
    """Karar metninin bilgilendirici kısmını çıkar (boilerplate'i atla, e5 512-token'a sığsın).

    e5 modelleri ilk ~512 token'ı görür; karar başı genelde mahkeme/taraf boilerplate'idir.
    Asıl hukuki içerikten başlatarak hem BM25 hem dense için ADİL ve bilgilendirici
    bir pasaj sağlar.
    """
    start = 0
    for marker in CONTENT_MARKERS:
        idx = text.find(marker)
        if 0 <= idx < 1500:        # makul aralıkta bir içerik işareti bulunduysa oradan başla
            start = idx
            break
    return text[start:start + max_chars].strip()


def _is_bad_query(q):
    global _STOP_NORM, _GENERIC_NORM
    if _STOP_NORM is None:
        _STOP_NORM = {_norm(s) for s in STOP_QUERY_TERMS}
        _GENERIC_NORM = {_norm(s) for s in GENERIC_TERMS}
    qn = _norm(q)
    if qn in _STOP_NORM:
        return True
    if _is_broken_spaced(qn):             # "y a r g i t a y" gibi bozuk metin
        return True
    if re.search(r"\b[ek]\.\s*\d+", qn):  # esas/karar no
        return True
    if re.fullmatch(r"[\d/.\s\-]+", qn):
        return True
    # "madde N" / "md N" tek başına (kanun adı yok) → ayırt edici değil
    if re.fullmatch(r"(madde|md|m)\.?\s*\d+", qn):
        return True
    if len(qn.split()) < 2:               # tek kelime sorgular ayırt edici değil
        return True
    if qn in _GENERIC_NORM:               # generic tek-kavram
        return True
    # tüm kelimeleri generic olan kombinasyonları ele
    if set(qn.split()).issubset(_GENERIC_NORM):
        return True
    return False


def extract_queries_for_decision(decision):
    """Bir karardan aday sorgular üret (anlamsal + leksik)."""
    text = decision.get("text", "")
    head = text[:800]
    queries = []  # (query_text, type)  type: 'semantic' | 'lexical'

    # 1) SUÇ / KONU / DAVA alanı (anlamsal)
    m = re.search(r"(?:SUÇ|KONU|DAVA)\s*:\s*([^\n]+)", head)
    if m:
        q = m.group(1).strip()
        if not _is_bad_query(q):
            queries.append((q[:120], "semantic"))

    # 2) Büyük harfli anahtar kavram başlıkları (anlamsal)
    for bc in re.findall(r"(?:^|\n)\s*([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ '\-]{10,70})\s*(?:\n|$)", head):
        q = bc.strip()
        if not _is_bad_query(q):
            queries.append((q.lower()[:120], "semantic"))

    # 3) Atıf yapılan kanun maddesi (leksik)
    for k in (decision.get("kanunlar") or [])[:2]:
        if not _is_bad_query(k):
            queries.append((k.strip()[:120], "lexical"))

    # benzersizleştir
    seen = set()
    out = []
    for q, t in queries:
        key = _norm(q)
        if key not in seen:
            seen.add(key)
            out.append((q, t))
    return out


def build_silver(decisions, max_queries_per_decision=2, seed=42):
    """Otomatik silver benchmark üret.

    Döndürür: corpus(dict id->rec), queries(dict id->{text,type}), qrels(list of (qid,did,rel))
    """
    random.seed(seed)
    corpus = {}
    queries = {}
    qrels = []

    for d in decisions:
        did = f"d{d['karar_id']}"
        full_text = d.get("text", "")
        corpus[did] = {
            "title": d.get("title", "") or "",
            # Bilgilendirici pasaj: boilerplate atlanır, ~2000 char (e5 512-token sınırına uygun).
            # Hem BM25 hem dense aynı pasajı görür → adil karşılaştırma, truncation kaybı yok.
            "text": informative_passage(full_text, max_chars=2000),
            "kurul": d.get("kurul", ""),
            "esas_no": d.get("esas_no", ""),
            "karar_no": d.get("karar_no", ""),
            "tarih": d.get("tarih", ""),
        }
        cand = extract_queries_for_decision(d)
        random.shuffle(cand)
        for j, (qtext, qtype) in enumerate(cand[:max_queries_per_decision]):
            qid = f"q{d['karar_id']}_{j}"
            queries[qid] = {"text": qtext, "type": qtype}
            qrels.append((qid, did, 1))   # silver: bu kararın kendisi positive

    return corpus, queries, qrels


def save_beir(corpus, queries, qrels, out_dir):
    """BEIR formatında kaydet."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "corpus.jsonl", "w", encoding="utf-8") as f:
        for did, rec in corpus.items():
            f.write(json.dumps({"_id": did, "title": rec["title"], "text": rec["text"],
                                "metadata": {k: rec[k] for k in ("kurul", "esas_no", "karar_no", "tarih")}},
                               ensure_ascii=False) + "\n")

    with open(out / "queries.jsonl", "w", encoding="utf-8") as f:
        for qid, q in queries.items():
            f.write(json.dumps({"_id": qid, "text": q["text"], "type": q["type"]},
                               ensure_ascii=False) + "\n")

    with open(out / "qrels.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, did, rel in qrels:
            f.write(f"{qid}\t{did}\t{rel}\n")

    # İstatistik
    stats = {
        "n_corpus": len(corpus),
        "n_queries": len(queries),
        "n_qrels": len(qrels),
        "query_types": dict(_count_types(queries)),
    }
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[benchmark] Saved BEIR-format to {out_dir}: "
          f"{stats['n_corpus']} docs, {stats['n_queries']} queries, {stats['n_qrels']} qrels")
    return stats


def _count_types(queries):
    c = defaultdict(int)
    for q in queries.values():
        c[q.get("type", "unknown")] += 1
    return c


def export_gold_template(queries, corpus, qrels, out_path, n_sample=150, seed=42):
    """Manuel GOLD etiketleme için CSV şablonu çıkar.

    Annotator her satır için relevance girer: 0 (ilgisiz) / 1 (ilgili) / 2 (çok ilgili)
    """
    random.seed(seed)
    qrel_map = defaultdict(list)
    for qid, did, rel in qrels:
        qrel_map[qid].append(did)

    qids = list(queries.keys())
    random.shuffle(qids)
    sample_qids = qids[:n_sample]

    import csv
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query_text", "query_type", "candidate_doc_id",
                    "candidate_title", "candidate_excerpt", "relevance_0_1_2"])
        for qid in sample_qids:
            q = queries[qid]
            for did in qrel_map.get(qid, []):
                rec = corpus.get(did, {})
                excerpt = (rec.get("text", "")[:300]).replace("\n", " ")
                w.writerow([qid, q["text"], q.get("type", ""), did,
                            rec.get("title", ""), excerpt, ""])
    print(f"[benchmark] Gold annotation template ({len(sample_qids)} queries) -> {out_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from hf_loader import load_hf_decisions

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "../artifacts/turklegalbench"

    print(f"Loading {n} decisions...")
    decisions = load_hf_decisions(n=n)
    corpus, queries, qrels = build_silver(decisions, max_queries_per_decision=2)
    save_beir(corpus, queries, qrels, out_dir)
    export_gold_template(queries, corpus, qrels, f"{out_dir}/gold_template.csv", n_sample=150)
