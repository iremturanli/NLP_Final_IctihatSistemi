"""HuggingFace açık kaynak Türkçe hukuk dataset'inden veri yükle.

Dataset: erdem-erdem/Turkish-Law-Documents-700k-clustered
Lisans: Türk devlet kurumlarının halka açık karar metinlerinden derlenmiş (Yargıtay + Danıştay).
"""
import re
from datasets import load_dataset


SOURCE_MAP = {
    "ygty1": "Yargıtay (Hukuk Daireleri)",
    "ygty2": "Yargıtay (Ceza Daireleri)",
    "dnsy1": "Danıştay",
    "dnsy2": "Danıştay",
}


def clean_markdown(text):
    """Markdown işaretlerini temizle ama içerik yapısını koru."""
    if not text:
        return ""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def extract_kurul(text, source):
    """Karar metninin ilk satırlarından kurul adını çıkar."""
    head = text[:500]
    m = re.search(r'(\d+\s*\.?\s*(?:Hukuk|Ceza|İdari|Vergi)\s*(?:Dairesi|Genel Kurulu))', head)
    if m:
        return m.group(1).strip()
    m2 = re.search(r'(Hukuk Genel Kurulu|Ceza Genel Kurulu|İçtihatları Birleştirme)', head)
    if m2:
        return m2.group(1)
    return SOURCE_MAP.get(source, source or "Bilinmeyen")


def extract_kanunlar(text):
    """Metinden atıfta bulunulan kanun maddelerini bul."""
    patterns = [
        r'\d{3,4}\s*[Ss]ayılı\s+[\wÇĞİÖŞÜçğıöşü\s]{3,80}?Kanun[\wçğıöşü]*(?:[\’\']n[unun]+)?\s+(?:m\.?|[Mm]adde|Md\.?)\s*\d+',
        r'[Mm]adde\s+\d+',
    ]
    out = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            s = m.group(0).strip()
            if s not in out:
                out.append(s)
            if len(out) >= 20:
                break
        if len(out) >= 20:
            break
    return out


def extract_suc_or_konu(text):
    """SUÇ veya KONU alanını çıkar (varsa)."""
    m = re.search(r'(?:SUÇ|KONU|DAVA)\s*:\s*([^\n]+)', text)
    if m:
        return m.group(1).strip()[:200]
    return ""


def load_hf_decisions(n=3000, min_chars=500, max_chars=15000, dataset_name="erdem-erdem/Turkish-Law-Documents-700k-clustered"):
    """HF dataset'ten N karar yükle, parse et."""
    ds = load_dataset(dataset_name, split="train", streaming=True)
    out = []
    skipped = 0
    for item in ds:
        if len(out) >= n:
            break
        raw_text = item.get("text") or ""
        text = clean_markdown(raw_text)
        if len(text) < min_chars:
            skipped += 1
            continue
        if len(text) > max_chars:
            text = text[:max_chars]

        source = item.get("source", "")
        kurul = extract_kurul(text, source)
        kanunlar = extract_kanunlar(text)
        konu = extract_suc_or_konu(text)

        out.append({
            "karar_id": item.get("id"),
            "esas_no": item.get("esasNo"),
            "karar_no": item.get("kararNo"),
            "tarih": item.get("kararTarihi"),
            "kurul": kurul,
            "source": source,
            "source_label": SOURCE_MAP.get(source, source),
            "title": f"{kurul} - E. {item.get('esasNo')}, K. {item.get('kararNo')}",
            "kanunlar": kanunlar,
            "konu": konu,
            "hukum": "",
            "text": text,
            "sonuc": "",
        })
    print(f"[hf_loader] Loaded {len(out)} decisions, skipped {skipped} (too short)")
    return out


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    recs = load_hf_decisions(n=n)
    for r in recs[:3]:
        print("---")
        print(f"  source: {r['source']} ({r['source_label']})")
        print(f"  kurul: {r['kurul']}")
        print(f"  esas/karar: {r['esas_no']} / {r['karar_no']} ({r['tarih']})")
        print(f"  konu: {r['konu']}")
        print(f"  kanunlar: {r['kanunlar'][:3]}")
        print(f"  text len: {len(r['text'])}")
        print(f"  text excerpt: {r['text'][:200]}...")
