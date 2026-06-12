"""Yargıtay kararları veri yükleme ve HTML temizleme."""
import json
import re
import ijson
from bs4 import BeautifulSoup
from pathlib import Path


def stream_records(json_path, limit=None):
    """Büyük JSON dosyasını stream et."""
    with open(json_path, 'rb') as f:
        items = ijson.items(f, 'KARARLAR.item')
        for i, rec in enumerate(items):
            if limit and i >= limit:
                break
            yield rec


def clean_html(html_text):
    """HTML'i düz metne çevir, fazla boşlukları temizle."""
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text(separator="\n")
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def extract_sections(html_text):
    """HTML'den başlık / atıf yapılan kanunlar / içtihat metni ve sonuç bölümlerini ayır."""
    if not html_text:
        return {"title": "", "kanunlar": [], "icithat": "", "sonuc": ""}

    soup = BeautifulSoup(html_text, "html.parser")
    title = (soup.title.get_text(strip=True) if soup.title else "")

    kanunlar = []
    for li in soup.find_all('li'):
        t = li.get_text(" ", strip=True)
        if re.search(r'\bS\.\s*\w|Madde\s+\d', t):
            kanunlar.append(t)

    full = clean_html(html_text)
    icithat = ""
    sonuc = ""
    if 'İçtihat Metni' in full:
        parts = full.split('İçtihat Metni', 1)
        icithat = parts[1] if len(parts) > 1 else ""
    else:
        icithat = full

    for marker in ['SONUÇ', 'HGK KARARI', 'KARAR:']:
        if marker in icithat:
            idx = icithat.rfind(marker)
            sonuc = icithat[idx:idx+2000]
            break

    return {
        "title": title,
        "kanunlar": kanunlar[:20],
        "icithat": icithat,
        "sonuc": sonuc,
    }


def load_decisions(json_path, limit=3000, min_chars=500):
    """Kararları yükle, temizle, yapılandır."""
    out = []
    for rec in stream_records(json_path, limit=limit * 2):
        if len(out) >= limit:
            break
        meta = rec.get("metadata", {})
        html = rec.get("karar_html_excerpt") or rec.get("KARAR") or meta.get("KARAR", "")
        if isinstance(html, dict):
            html = html.get("KARAR", "")
        text = clean_html(html)
        if len(text) < min_chars:
            continue
        sections = extract_sections(html)

        # metadata alanları kayıt seviyesinde de bulunabilir
        karar_id = meta.get("KARAR_ID") or rec.get("KARAR_ID")
        esas_no = meta.get("ESAS_NO") or rec.get("ESAS_NO")
        karar_no = meta.get("KARAR_NO") or rec.get("KARAR_NO")
        tarih = meta.get("KARAR_TARIHI") or rec.get("KARAR_TARIHI")
        kurul = meta.get("KURUL_ADI") or rec.get("KURUL_ADI")
        hukum = meta.get("HUKUM") or rec.get("HUKUM", "")

        out.append({
            "karar_id": karar_id,
            "esas_no": esas_no,
            "karar_no": karar_no,
            "tarih": tarih,
            "kurul": kurul,
            "hukum": hukum,
            "title": sections["title"],
            "kanunlar": sections["kanunlar"],
            "text": text,
            "sonuc": sections["sonuc"],
        })
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../KARARLAR_202601061508.json"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    recs = load_decisions(path, limit=n)
    print(f"Loaded {len(recs)} decisions")
    print("Sample:")
    r = recs[0]
    print(f"  karar_id: {r['karar_id']}")
    print(f"  esas/karar: {r['esas_no']} / {r['karar_no']}")
    print(f"  kurul: {r['kurul']}")
    print(f"  title: {r['title'][:80]}")
    print(f"  kanunlar: {r['kanunlar'][:3]}")
    print(f"  text len: {len(r['text'])}")
    print(f"  text excerpt: {r['text'][:300]}")
