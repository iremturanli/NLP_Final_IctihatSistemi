"""Karar metinlerini retrieval için anlamlı chunk'lara böl."""
import re


def split_paragraphs(text):
    """Boş satırlarla paragraflara böl, çok kısa olanları birleştir."""
    parts = re.split(r'\n\s*\n+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def split_sentences(text):
    """Cümle sınırlarında böl. Yargıtay metinlerinde nokta ve nokta-virgül yaygın."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]


def split_long_paragraph(p, max_chars):
    """Uzun bir paragrafı cümle bazında parçalara böl."""
    sents = split_sentences(p)
    parts = []
    cur = ""
    for s in sents:
        if not cur:
            cur = s
        elif len(cur) + len(s) + 1 <= max_chars:
            cur = cur + " " + s
        else:
            parts.append(cur)
            cur = s
    if cur:
        parts.append(cur)
    if not parts:
        # Tek cümle bile max_chars'dan büyükse — kelime ile zorla böl.
        parts = [p[i:i+max_chars] for i in range(0, len(p), max_chars)]
    return parts


def chunk_text(text, max_chars=1200, overlap=150, min_chars=200):
    """Karar metnini chunk'lara böl. Önce paragraf, gerekirse cümle sınırlarında."""
    paragraphs = split_paragraphs(text)
    # Uzun paragrafları cümle bazında alt paragraflara böl
    flat = []
    for p in paragraphs:
        if len(p) <= max_chars:
            flat.append(p)
        else:
            flat.extend(split_long_paragraph(p, max_chars))

    chunks = []
    current = ""
    for p in flat:
        if not current:
            current = p
        elif len(current) + len(p) + 2 <= max_chars:
            current = current + "\n\n" + p
        else:
            if len(current) >= min_chars:
                chunks.append(current)
            tail = current[-overlap:] if overlap and len(current) > overlap else ""
            current = (tail + "\n\n" + p).strip() if tail else p
    if current and len(current) >= min_chars:
        chunks.append(current)
    return chunks


def build_chunk_records(decisions, max_chars=1200, overlap=150):
    """Karar listesini chunk seviyesinde kayıt listesine dönüştür."""
    out = []
    for d in decisions:
        chunks = chunk_text(d["text"], max_chars=max_chars, overlap=overlap)
        for i, c in enumerate(chunks):
            out.append({
                "karar_id": d["karar_id"],
                "esas_no": d["esas_no"],
                "karar_no": d["karar_no"],
                "tarih": d["tarih"],
                "kurul": d["kurul"],
                "title": d["title"],
                "kanunlar": d["kanunlar"],
                "chunk_idx": i,
                "n_chunks": len(chunks),
                "text": c,
            })
    return out


if __name__ == "__main__":
    sample = "A" * 300 + "\n\n" + "B" * 800 + "\n\n" + "C" * 400 + "\n\n" + "D" * 600
    chunks = chunk_text(sample, max_chars=1000)
    print(f"Got {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"--- chunk {i} ({len(c)} chars) ---\n{c[:100]}...\n")
