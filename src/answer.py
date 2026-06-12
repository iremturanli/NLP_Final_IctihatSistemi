"""Retrieved kararlardan kaynaklı cevap üret (extractive, hallucination-free)."""
import re
import textwrap


def _highlight_query_terms(text, query_tokens, max_chars=600):
    """Sorgu kelimelerinin geçtiği ilk bölümü kırp."""
    text_lower = text.lower()
    best_pos = -1
    for t in query_tokens:
        pos = text_lower.find(t)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos
    if best_pos == -1:
        return text[:max_chars] + ("..." if len(text) > max_chars else "")
    start = max(0, best_pos - 100)
    end = min(len(text), start + max_chars)
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def extract_summary(text, max_sentences=3):
    """Karardan ilk anlamlı cümleleri çıkar."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 30]
    return " ".join(sents[:max_sentences])


def format_answer(query, results, max_results=5):
    """Markdown formatında kaynaklı cevap üret."""
    from retriever import tokenize_tr
    q_tokens = tokenize_tr(query)

    if not results:
        return "**Üzgünüm, bu soruyla ilgili veri tabanımda eşleşen bir Yargıtay kararı bulamadım.**"

    out = []
    out.append(f"### Sorgu\n> {query}\n")
    out.append(f"**Bulunan {min(len(results), max_results)} ilgili Yargıtay kararı:**\n")

    for i, r in enumerate(results[:max_results], 1):
        kurul = r.get("kurul") or "Bilinmeyen Kurul"
        esas = r.get("esas_no") or "?"
        karar = r.get("karar_no") or "?"
        tarih = r.get("tarih") or "?"
        score = r.get("score", 0)

        snippet = _highlight_query_terms(r["text"], q_tokens, max_chars=500)
        snippet = re.sub(r'\s+', ' ', snippet).strip()

        kanunlar = r.get("kanunlar") or []
        kanun_str = "; ".join(kanunlar[:3]) if kanunlar else "—"

        out.append(
            f"---\n"
            f"**[{i}] {kurul} — E. {esas}, K. {karar} ({tarih})**  \n"
            f"İlgili mevzuat: {kanun_str}  \n"
            f"Skor: {score:.3f}  \n\n"
            f"> {snippet}\n"
        )

    out.append(
        "---\n"
        "_⚠ Not: Yukarıdaki sonuçlar Yargıtay kararları üzerinden retrieval ile getirilmiştir. "
        "Hukuki tavsiye değildir; bağlayıcı yorum için ilgili kararın tam metnine başvurunuz._"
    )
    return "\n".join(out)
