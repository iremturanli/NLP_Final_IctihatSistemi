"""YargıRAG — Avukat Asistanı (profesyonel UI).

Güvenilirlik öncelikli tasarım:
  - Her cevap gerçek karara dayanır (esas/karar no ile doğrulanabilir)
  - Yetersiz emsal bulunursa "cevap veremiyorum" (abstention)
  - Görsel güven göstergesi (yüksek/orta/düşük)
  - Uydurma yok (extractive) — hukuki güvenlik için
  - "Hukuki tavsiye değildir" uyarısı

Çalıştırma:
  PYTHONNOUSERSITE=1 LEGALRAG_INDEX=artifacts/index_ft_splitclean_full \\
    LEGALRAG_MODEL=artifacts/legal-e5-tr-splitclean python src/app_pro.py
"""
import sys
import os
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import gradio as gr
from embedder import load_model, load_index
from retriever import HybridRetriever, tokenize_tr

_ROOT = Path(__file__).parent.parent
INDEX_DIR = os.environ.get("LEGALRAG_INDEX", str(_ROOT / "artifacts" / "index_small_ft"))
MODEL_NAME = os.environ.get("LEGALRAG_MODEL", str(_ROOT / "artifacts" / "e5-small-tr-legal-ft"))
USE_RERANKER = os.environ.get("LEGALRAG_RERANKER", "0") == "1"

# Abstention eşiği: ham dense COSINE benzerliği bunun altındaysa "yeterli emsal yok".
# Kalibrasyon:
#   - legal-e5-tr-splitclean (e5-base): alakalı ~0.35-0.64, alakasız ~0.24-0.26 → eşik 0.30
#   - e5-small-tr-legal-ft           : alakalı ~0.53-0.72, alakasız ~0.40-0.45 → eşik 0.48
# Varsayılan split-clean modele göre 0.30; başka model için LEGALRAG_ABSTAIN ile geç.
ABSTAIN_THRESHOLD = float(os.environ.get("LEGALRAG_ABSTAIN", "0.30"))

print(f"[app] İndeks yükleniyor: {INDEX_DIR}")
index, embeddings, records = load_index(INDEX_DIR)
N_KARAR = len(set(r.get("karar_id") for r in records))
print(f"[app] {N_KARAR} karar, {len(records)} pasaj")

print(f"[app] Model yükleniyor: {MODEL_NAME}")
model = load_model(MODEL_NAME)
hybrid = HybridRetriever(index, embeddings, records, model)

if USE_RERANKER:
    from reranker import RerankedRetriever
    retriever = RerankedRetriever(hybrid, fetch_k=20)
    print("[app] Hazır (hibrit + reranker).")
else:
    retriever = hybrid
    print("[app] Hazır (hibrit retrieval).")


EXAMPLES = [
    "İhtiyati haciz icra takibi sayılır mı?",
    "Karşılıksız yararlanma suçunda beraat hangi koşullarda mümkündür?",
    "İş akdinin haksız feshinde kıdem tazminatı nasıl hesaplanır?",
    "Kira sözleşmesinde tahliye taahhüdünün geçerlilik şartları",
    "Kasten öldürme suçunda tasarlama unsurunun değerlendirilmesi",
    "Boşanmada velayet belirlenirken çocuğun üstün yararı",
]


def _confidence(dense_score):
    """Ham dense cosine benzerliğinden güven seviyesi + renk.

    Eşiğe göreli: abstain eşiğinin üstündeki band'e göre Sınırlı/Orta/Yüksek.
    Böylece e5-small ve e5-base gibi farklı cosine ölçeklerinde de tutarlı.
    """
    if dense_score >= ABSTAIN_THRESHOLD + 0.20:
        return "Yüksek", "#1a7f37", "●●●"
    if dense_score >= ABSTAIN_THRESHOLD + 0.08:
        return "Orta", "#bf8700", "●●○"
    return "Sınırlı", "#cf222e", "●○○"


def _highlight(query, text, span=420):
    """İlgili pasajı kırp ve sorgunun ANLAMLI (4+ harf) kelimelerini güvenle vurgula.

    Ham sorgu kelimeleri kullanılır (Türkçe ı/i normalizasyonu YAPILMAZ) → yanlış
    eşleşme olmaz. Tam kelime sınırı aranır. Eşleşme yoksa metnin başı gösterilir.
    """
    # Vurgulanacak kelimeler: sorgudan, 4+ harf, noktalama temizli
    raw = re.findall(r"\w{4,}", query, flags=re.UNICODE)
    stop = {"nasıl", "hangi", "nedir", "için", "mıdır", "midir", "neye", "göre",
            "değerlendirilmesi", "belirlenirken", "koşullarda", "mümkün", "şartları"}
    words = [w for w in raw if w.lower() not in stop]

    # Pasaj başlangıcı: ilk eşleşen kelimenin çevresi
    tl = text.lower()
    pos = -1
    for w in words:
        p = tl.find(w.lower())
        if p != -1 and (pos == -1 or p < pos):
            pos = p
    if pos == -1:
        snippet = text[:span]
        start_cut = False
    else:
        s = max(0, pos - 80)
        snippet = text[s:s + span]
        start_cut = s > 0
    snippet = re.sub(r"\s+", " ", snippet).strip()
    prefix = "… " if start_cut else ""
    suffix = " …" if len(text) > span else ""

    # Tam kelime sınırıyla vurgula (uzun kelimeden kısaya, çakışmayı önle)
    for w in sorted(set(words), key=len, reverse=True):
        snippet = re.sub(rf"(?<!\w)({re.escape(w)})(?!\w)", r"§§\1§§", snippet,
                         flags=re.IGNORECASE)
    return prefix + snippet + suffix


def _build_summary(results):
    """Retrieved kararlardan YAPILANDIRILMIŞ extractive özet (LLM yok → halüsinasyon yok).

    Daire dağılımı, hüküm türleri ve en sık atıf yapılan kanun maddelerini sentezler.
    """
    from collections import Counter
    if not results:
        return ""
    n = len(results)
    # Daire dağılımı
    daireler = Counter(r.get("kurul", "?") for r in results)
    top_daire, top_daire_n = daireler.most_common(1)[0]
    daire_txt = (f"çoğunlukla <b>{top_daire}</b>" if top_daire_n > 1
                 else f"<b>{len(daireler)} farklı daireden</b>")

    # Hüküm türleri (karar metninden)
    huküm_pat = {
        "beraat": r"beraat", "mahkumiyet": r"mahk[uû]miyet",
        "bozma": r"bozulmas|bozma", "onama": r"onan|onama",
        "ret": r"redd|reddine", "kabul": r"kabul[üu]ne",
    }
    huküm_say = Counter()
    for r in results:
        t = r.get("text", "").lower()
        for ad, pat in huküm_pat.items():
            if re.search(pat, t):
                huküm_say[ad] += 1
    huküm_txt = ""
    if huküm_say:
        parts = [f"{ad} ({s})" for ad, s in huküm_say.most_common(3)]
        huküm_txt = " · ".join(parts)

    # En sık atıf yapılan kanun maddeleri
    kanun_say = Counter()
    for r in results:
        for k in (r.get("kanunlar") or []):
            kanun_say[k] += 1
    kanun_txt = ""
    if kanun_say:
        kanun_txt = "; ".join(k for k, _ in kanun_say.most_common(2))

    rows = [f"<b>{n}</b> emsal {daire_txt} geldi."]
    if huküm_txt:
        rows.append(f"<b>Hüküm dağılımı:</b> {huküm_txt}")
    if kanun_txt:
        rows.append(f"<b>Sık atıf:</b> {kanun_txt}")
    inner = "<br>".join(rows)
    return (f"<div class='summary'><div class='sum-icon'>📋</div>"
            f"<div><div class='sum-title'>Özet</div>{inner}"
            f"<div class='sum-note'>Bu özet yalnızca getirilen kararlardan otomatik "
            f"çıkarılmıştır; yorum içermez.</div></div></div>")


def respond(query, top_k, use_rerank):
    if not query or not query.strip():
        return "<div class='hint'>Lütfen bir hukuki soru yazın.</div>"
    query = query.strip()

    active = retriever
    if use_rerank and not USE_RERANKER:
        try:
            from reranker import RerankedRetriever
            active = RerankedRetriever(hybrid, fetch_k=20)
        except Exception:
            active = retriever

    results = active.search(query, top_k=int(top_k))
    q_tokens = tokenize_tr(query)

    # --- Abstention: en iyi sonucun ham cosine benzerliği zayıfsa güvenli kaçınma ---
    top_score = results[0].get("dense_score", 0) if results else 0
    if not results or top_score < ABSTAIN_THRESHOLD:
        return f"""
        <div class='abstain'>
          <div class='abstain-icon'>⚠️</div>
          <div>
            <b>Bu soru için yeterli güvenilirlikte emsal karar bulamadım.</b><br>
            <span>Veri tabanındaki kararlar sorunuzu güvenle yanıtlamak için yetersiz.
            Soruyu farklı ifade etmeyi deneyebilir veya bir hukuk uzmanına danışabilirsiniz.</span>
          </div>
        </div>"""

    # Yakın-duplike pasajları ele (ardışık kararlar çoğu zaman aynı metni paylaşır)
    deduped, seen_sigs = [], []
    for r in results:
        sig = re.sub(r"\W+", "", (r.get("text", "")[:120]).lower())
        if any(sig[:80] == s[:80] for s in seen_sigs):
            continue
        seen_sigs.append(sig)
        deduped.append(r)
    results = deduped

    cards = []
    for i, r in enumerate(results[:int(top_k)], 1):
        level, color, dots = _confidence(r.get("dense_score", 0))
        kurul = r.get("kurul") or "Bilinmeyen Daire"
        esas = r.get("esas_no") or "?"
        karar = r.get("karar_no") or "?"
        tarih = r.get("tarih") or "—"
        snippet = _highlight(query, r.get("text", ""))
        # önce HTML-escape, sonra marker -> <mark>
        snippet = (snippet.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        snippet_html = re.sub(r"§§(.+?)§§", r"<mark>\1</mark>", snippet)
        kanunlar = r.get("kanunlar") or []
        kanun_html = ""
        if kanunlar:
            chips = "".join(f"<span class='chip'>{k}</span>" for k in kanunlar[:3])
            kanun_html = f"<div class='chips'>{chips}</div>"

        # Tam metin (kartın o kararına ait tüm pasajları birleştir)
        kid = r.get("karar_id")
        full_parts = [rec.get("text", "") for rec in records if rec.get("karar_id") == kid]
        full_text = "\n\n".join(dict.fromkeys(full_parts))  # sıra koruyarak benzersiz
        full_text = (full_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        full_text = full_text.replace("\n", "<br>")

        cards.append(f"""
        <details class='card'>
          <summary class='card-head'>
            <span class='rank'>{i}</span>
            <div class='meta'>
              <div class='kurul'>{kurul}</div>
              <div class='ref'>Esas {esas} · Karar {karar} · {tarih}</div>
            </div>
            <div class='conf' style='color:{color}'>
              <span class='dots'>{dots}</span><span class='lvl'>{level} güven</span>
            </div>
          </summary>
          <div class='snippet'>{snippet_html}</div>
          {kanun_html}
          <div class='fulltext'>
            <div class='ft-label'>📄 Karar metni</div>{full_text}
          </div>
        </details>""")

    header = f"""
    <div class='answer-head'>
      <b>{len(cards)} ilgili emsal karar</b> bulundu
      <span class='sub'>— kaynak göstererek, en alakalıdan sıralı</span>
    </div>"""
    disclaimer = """
    <div class='disclaimer'>
      ⚖️ Bu sonuçlar emsal karar araştırmasıdır, <b>hukuki tavsiye değildir</b>.
      Bağlayıcı yorum için kararın tam metnine ve bir hukuk uzmanına başvurun.
    </div>"""
    summary = _build_summary(results[:int(top_k)])
    return summary + header + "<div class='cardgrid'>" + "".join(cards) + "</div>" + disclaimer


CSS = """
/* ===== Açık temayı HER durumda zorla (dark mode dahil) ===== */
.gradio-container, .dark .gradio-container, gradio-app, .dark {
  background:#eef1f6 !important; max-width:1140px !important; margin:auto !important;
  padding:0 20px !important;
  font-family:'Inter','Segoe UI',system-ui,sans-serif !important; }
/* Tüm metinler koyu (beyaz yazı kalmasın) — hero hariç aşağıda override */
.gradio-container, .gradio-container *, .dark .gradio-container * { color:#1f2937; }
.gradio-container .block, .gradio-container .form, .gradio-container .panel,
.dark .block, .dark .form, .gradio-container .gap {
  background:transparent !important; border:none !important; box-shadow:none !important; }
/* Boşlukları sıkıştır */
.gradio-container .gap { gap:8px !important; }
.gradio-container > .main, .gradio-container .contain { padding:0 !important; }

/* ===== HERO ===== */
#hero { background:linear-gradient(135deg,#16243e 0%,#2b4a72 55%,#34537d 100%);
  padding:22px 28px; border-radius:16px; margin:4px 0 14px;
  box-shadow:0 8px 26px rgba(22,36,62,.26); position:relative; overflow:hidden; }
#hero:before { content:'⚖'; position:absolute; right:-6px; top:-26px; font-size:130px;
  opacity:.07; transform:rotate(-12deg); color:#fff; }
#hero *, #hero { color:#fff !important; }
#hero h1 { margin:0; font-size:24px; font-weight:800; letter-spacing:-.4px; }
#hero h1 span { font-weight:400; opacity:.72; font-size:16px; }
#hero .tag { opacity:.92; font-size:13.5px; margin-top:5px; color:#dce6f5 !important; }
#hero .stats { margin-top:12px; font-size:12px; color:#c4d3ea !important;
  background:rgba(255,255,255,.09); display:inline-block; padding:6px 13px; border-radius:9px; }
#hero .stats b { color:#ffd479 !important; font-weight:700; }

/* ===== Arama kutusu ===== */
#searchbox textarea {
  background:#fff !important; color:#1a2942 !important; border:1.5px solid #d4dce8 !important;
  border-radius:12px !important; font-size:15px !important; padding:13px 16px !important;
  box-shadow:0 2px 8px rgba(22,36,62,.05) !important; }
#searchbox textarea:focus { border-color:#2b4a72 !important; box-shadow:0 0 0 3px rgba(43,74,114,.13) !important; }
#searchbox textarea::placeholder { color:#94a3b8 !important; }

/* ===== Arama satırı: buton + opsiyonlar yan yana ===== */
#searchrow { gap:12px !important; align-items:stretch !important; margin-top:2px; }
#gobtn button { background:linear-gradient(135deg,#1f3a5f,#2b4a72) !important; color:#fff !important;
  border:none !important; border-radius:11px !important; font-weight:700 !important; font-size:15px !important;
  height:100% !important; min-height:60px; box-shadow:0 4px 14px rgba(43,74,114,.28) !important; transition:.18s; }
#gobtn button:hover { transform:translateY(-1px); box-shadow:0 6px 20px rgba(43,74,114,.38) !important; }
#opts { background:#fff !important; border:1px solid #e3e9f2 !important; border-radius:11px !important;
  padding:8px 14px !important; }
#opts label, #opts span, #opts .gradio-slider span { color:#475569 !important; font-size:12.5px !important;
  font-weight:500 !important; }
#opts input[type=range] { accent-color:#2b4a72; }

/* ===== Örnek sorular ===== */
#ex { margin-top:4px !important; }
#ex .label-wrap span, #ex > .label, #ex span { color:#64748b !important; font-size:12px !important;
  font-weight:600 !important; }
#ex button, .examples button {
  background:#fff !important; border:1px solid #dbe3ef !important; color:#34507a !important;
  border-radius:18px !important; font-size:12px !important; padding:6px 13px !important;
  font-weight:500 !important; transition:.15s; }
#ex button:hover { background:#eef4fb !important; border-color:#2b4a72 !important; }

/* ===== Özet kartı ===== */
.summary { display:flex; gap:13px; align-items:flex-start; background:linear-gradient(135deg,#f0f5ff,#eaf0fb);
  border:1px solid #d4e0f2; border-radius:13px; padding:15px 18px; margin:14px 0 4px; }
.summary * { color:#1f3a5f !important; }
.sum-icon { font-size:22px; line-height:1; }
.sum-title { font-weight:700; font-size:14px; margin-bottom:4px; color:#16243e !important; }
.summary > div:last-child { font-size:13px; line-height:1.6; }
.sum-note { color:#6b7a90 !important; font-size:11px; margin-top:7px; font-style:italic; }

/* ===== Sonuç başlığı ===== */
.answer-head { font-size:15px; margin:12px 0 12px; color:#1a2942 !important; font-weight:700; }
.answer-head b { color:#1a2942 !important; }
.answer-head .sub { color:#64748b !important; font-weight:400; font-size:13px; }

/* ===== Sonuç ızgarası (2 sütun, geniş ekranda boşluğu doldurur) ===== */
.cardgrid { display:grid; grid-template-columns:1fr 1fr; gap:13px; }
@media (max-width:820px){ .cardgrid { grid-template-columns:1fr; } }

/* ===== Karar kartları (açılır/kapanır) ===== */
.card { border:1px solid #e3e9f2; border-radius:13px; padding:15px 17px; margin-bottom:11px;
  background:#fff !important; box-shadow:0 2px 9px rgba(22,36,62,.05); transition:.2s; }
.card:hover { box-shadow:0 8px 24px rgba(22,36,62,.12); }
.card[open] { box-shadow:0 8px 28px rgba(22,36,62,.14); border-color:#c3d3ea; }
details.card > summary { cursor:pointer; list-style:none; outline:none; }
details.card > summary::-webkit-details-marker { display:none; }
details.card > summary::after { content:'⌄'; position:absolute; right:18px; color:#94a3b8;
  font-size:16px; transition:transform .2s; }
details.card[open] > summary::after { transform:rotate(180deg); }
.card { position:relative; }
.fulltext { display:none; }
details.card[open] .fulltext { display:block; margin-top:12px; padding-top:12px;
  border-top:1px dashed #e3e9f2; font-size:12.5px; line-height:1.65; color:#475569 !important;
  max-height:340px; overflow-y:auto; }
.ft-label { font-weight:700; color:#2b4a72 !important; font-size:12px; margin-bottom:6px; }
.card-head { display:flex; align-items:flex-start; gap:12px; margin-bottom:9px; padding-right:24px; }
.rank { background:linear-gradient(135deg,#1f3a5f,#2b4a72); color:#fff !important; width:28px; height:28px;
  border-radius:8px; display:flex; align-items:center; justify-content:center;
  font-weight:800; font-size:13.5px; flex-shrink:0; }
.meta { flex:1; }
.kurul { font-weight:700; color:#16243e !important; font-size:15px; }
.ref { color:#64748b !important; font-size:12px; margin-top:2px; letter-spacing:.2px; }
.conf { text-align:right; font-size:11px; flex-shrink:0; }
.conf .dots { display:block; font-size:11.5px; letter-spacing:1.5px; margin-bottom:1px; }
.conf .lvl { font-weight:700; }
.snippet { color:#374151 !important; font-size:13.5px; line-height:1.6; }
.snippet mark { background:#ffe9a8 !important; color:#5b4500 !important; padding:0 3px;
  border-radius:3px; font-weight:600; }
.chips { margin-top:9px; }
.chip { display:inline-block; background:#eef3fa !important; color:#34507a !important; font-size:11px;
  padding:3px 10px; border-radius:18px; margin-right:5px; margin-top:4px; font-weight:500; }

/* ===== Abstention ===== */
.abstain { display:flex; gap:14px; background:#fff8f0 !important; border:1px solid #fed7aa;
  border-radius:13px; padding:18px 20px; align-items:flex-start; }
.abstain * { color:#9a3412 !important; }
.abstain-icon { font-size:26px; line-height:1; }
.abstain b { font-size:14.5px; }
.abstain span { color:#b45309 !important; font-size:13px; }

/* ===== Disclaimer ===== */
.disclaimer { margin-top:15px; padding:12px 16px; background:#fff !important;
  border-left:4px solid #2b4a72; border-radius:8px; font-size:12px; color:#475569 !important;
  box-shadow:0 1px 6px rgba(22,36,62,.04); }
.disclaimer b { color:#1a2942 !important; }
.hint { color:#94a3b8 !important; padding:22px; text-align:center; font-size:13.5px; }
.hint b { color:#2b4a72 !important; }
footer, .dark footer { display:none !important; }
"""

reranker_note = " + cross-encoder reranker" if USE_RERANKER else ""
_THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="slate",
                        neutral_hue="slate", font=["Inter", "system-ui", "sans-serif"])
# Sistem dark moddaysa bile açık temayı zorla
_FORCE_LIGHT_JS = """
() => {
  const kill = () => {
    document.body.classList.remove('dark');
    document.querySelectorAll('.dark').forEach(e => e.classList.remove('dark'));
    const app = document.querySelector('gradio-app');
    if (app) app.classList.remove('dark');
  };
  kill(); setTimeout(kill, 300); setTimeout(kill, 1000);
}
"""
with gr.Blocks(title="YargıRAG — Avukat Asistanı") as demo:
    gr.HTML(f"""
    <div id='hero'>
      <h1>⚖️ YargıRAG <span>· Avukat Asistanı</span></h1>
      <div class='tag'>Türkçe Yargıtay &amp; Danıştay kararlarında kaynak göstererek emsal araştırması</div>
      <div class='stats'>📚 <b>{N_KARAR:,}</b> karar &nbsp;·&nbsp; <b>{len(records):,}</b> pasaj
        &nbsp;·&nbsp; alana uyarlanmış <b>legal-e5-tr</b> &nbsp;·&nbsp; hibrit erişim{reranker_note}</div>
    </div>""")

    query = gr.Textbox(label="", elem_id="searchbox",
                       placeholder="Hukuki sorunuzu yazın…  örn: İhtiyati haciz icra takibi sayılır mı?",
                       lines=2, container=False)
    with gr.Row(elem_id="searchrow"):
        btn = gr.Button("🔍  Emsal Ara", variant="primary", scale=3, elem_id="gobtn")
        with gr.Column(scale=4, elem_id="opts"):
            top_k = gr.Slider(3, 10, value=5, step=1, label="Sonuç sayısı")
            rerank_cb = gr.Checkbox(label="Hassas sıralama (reranker — daha yavaş)", value=USE_RERANKER)

    gr.Examples(EXAMPLES, inputs=query, label="Örnek sorular", elem_id="ex")
    output = gr.HTML(value="<div class='hint'>Bir soru yazıp <b>Emsal Ara</b>'ya basın ya da örnek sorulardan seçin.</div>")

    btn.click(respond, [query, top_k, rerank_cb], output)
    query.submit(respond, [query, top_k, rerank_cb], output)


if __name__ == "__main__":
    # Gradio 6.0: css ve theme launch()'a verilir. Açık temayı zorla (?__theme=light).
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                css=CSS, theme=_THEME, js=_FORCE_LIGHT_JS)
