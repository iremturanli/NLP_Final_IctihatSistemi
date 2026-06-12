"""[DENEYSEL — SİSTEME ENTEGRE DEĞİL]

Bu modül üretici (generative) RAG + faithfulness ölçümü için yazılmıştır ANCAK
mevcut sistemde (app_pro.py) KULLANILMAZ. Sistem extractive çalışır: kaynak gösterir,
LLM ile cevap üretmez. Bu modül GELECEK İŞ içindir; tam generative katman ve
faithfulness değerlendirmesi henüz tamamlanmamıştır.
"""
"""Generative RAG + halüsinasyon güvenliği — bu çalışmanın imza katkısı.

İçerir:
  1. Grounded generation : Türkçe LLM, SADECE retrieved kararlara dayanarak cevap
  2. Abstention          : retrieval güveni düşükse "cevap veremiyorum"
  3. Faithfulness        : üretilen cevabın bağlamca desteklenme oranı (NLI)
  4. Citation accuracy   : gösterilen kaynak gerçekten ilgili mi
  5. Genel LLM vs grounded karşılaştırması (güvenlik kanıtı)

Model bağımsız tasarlandı; LLM ve NLI modeli parametre olarak verilir.
Colab A100'de açık Türkçe LLM (örn. ytu-ce-cosmos/Turkish-Llama-8b,
Trendyol/Trendyol-LLM-7b) ile çalışır.
"""
import re


GROUNDED_PROMPT = """Sen bir Türk hukuku uzmanı asistanısın. SADECE aşağıda verilen Yargıtay/Danıştay kararlarına dayanarak cevap ver.

KURALLAR:
- Yalnızca verilen kararlardaki bilgileri kullan. Kendi bilgini EKLEME.
- Her iddiada hangi karara dayandığını [Karar N] şeklinde belirt.
- Verilen kararlar soruyu cevaplamak için YETERSİZSE, sadece şunu yaz: "CEVAP_YOK"
- Uydurma karar, madde veya bilgi ÜRETME.

VERİLEN KARARLAR:
{context}

SORU: {question}

CEVAP:"""


PLAIN_PROMPT = """Sen bir Türk hukuku uzmanı asistanısın. Aşağıdaki hukuki soruyu cevapla.

SORU: {question}

CEVAP:"""


ABSTAIN_TOKEN = "CEVAP_YOK"


def build_context(retrieved, max_docs=5, max_chars_each=800):
    """Retrieved kararları numaralı bağlam metnine çevir."""
    parts = []
    for i, r in enumerate(retrieved[:max_docs], 1):
        meta = f"{r.get('kurul','?')} E.{r.get('esas_no','?')}/K.{r.get('karar_no','?')}"
        text = re.sub(r"\s+", " ", r.get("text", "")[:max_chars_each]).strip()
        parts.append(f"[Karar {i}] ({meta})\n{text}")
    return "\n\n".join(parts)


# ---------- Abstention ----------

def should_abstain(retrieved, score_threshold=0.020, min_docs=1):
    """Retrieval güvenine göre cevap vermeli mi karar ver.

    Düşük skor veya yetersiz döküman → abstain (hukukta güvenli davranış).
    """
    if len(retrieved) < min_docs:
        return True, "yetersiz_dokuman"
    top_score = retrieved[0].get("score", 0)
    if top_score < score_threshold:
        return True, "dusuk_guven"
    return False, None


# ---------- Generation ----------

def generate_answer(llm, question, retrieved, grounded=True, max_new_tokens=350,
                    abstain_check=True):
    """LLM ile cevap üret.

    llm: callable(prompt:str) -> str  (HF pipeline veya wrapper)
    grounded=True: sadece bağlama dayalı + abstention
    grounded=False: genel LLM (kaynak yok) — karşılaştırma için
    """
    if grounded and abstain_check:
        abstain, reason = should_abstain(retrieved)
        if abstain:
            return {"answer": ABSTAIN_TOKEN, "abstained": True, "reason": reason,
                    "grounded": grounded}

    if grounded:
        context = build_context(retrieved)
        prompt = GROUNDED_PROMPT.format(context=context, question=question)
    else:
        prompt = PLAIN_PROMPT.format(question=question)

    raw = llm(prompt, max_new_tokens=max_new_tokens)
    answer = raw.strip()
    abstained = ABSTAIN_TOKEN in answer
    return {"answer": answer, "abstained": abstained, "reason": None, "grounded": grounded,
            "prompt": prompt}


# ---------- Faithfulness (NLI tabanlı) ----------

def split_claims(answer):
    """Cevabı cümle/iddia birimlerine böl."""
    sents = re.split(r"(?<=[.!?])\s+", answer)
    return [s.strip() for s in sents if len(s.strip()) > 15]


def faithfulness_score(nli, answer, context, entail_label="entailment", threshold=0.5):
    """Cevabın her iddiası bağlam tarafından destekleniyor mu (NLI entailment).

    nli: callable(premise, hypothesis) -> {label: prob}  (Türkçe/multilingual NLI)
    Döndürür: {faithfulness, n_claims, supported, unsupported_claims:[...]}
    """
    claims = split_claims(answer)
    if not claims:
        return {"faithfulness": None, "n_claims": 0, "supported": 0, "unsupported_claims": []}
    supported = 0
    unsupported = []
    for c in claims:
        res = nli(premise=context, hypothesis=c)
        p_entail = res.get(entail_label, 0.0)
        if p_entail >= threshold:
            supported += 1
        else:
            unsupported.append(c)
    return {
        "faithfulness": supported / len(claims),
        "n_claims": len(claims),
        "supported": supported,
        "unsupported_claims": unsupported,
    }


# ---------- Citation accuracy ----------

def citation_accuracy(answer, retrieved, relevant_doc_ids=None):
    """Cevapta gösterilen [Karar N] atıfları geçerli/ilgili mi.

    relevant_doc_ids verilirse, atfedilen kararın gerçekten ilgili olup olmadığı kontrol edilir.
    """
    cited = set(int(m) for m in re.findall(r"\[Karar\s+(\d+)\]", answer))
    n_docs = len(retrieved)
    valid = [c for c in cited if 1 <= c <= n_docs]
    out = {"n_citations": len(cited), "valid_citations": len(valid),
           "citation_validity": len(valid) / len(cited) if cited else None}
    if relevant_doc_ids is not None:
        correct = 0
        for c in valid:
            did = retrieved[c-1].get("karar_id")
            if f"d{did}" in relevant_doc_ids:
                correct += 1
        out["citation_relevance"] = correct / len(valid) if valid else None
    return out


# ---------- LLM vs RAG güvenlik karşılaştırması ----------

def safety_comparison(llm, nli, question, retrieved, relevant_doc_ids=None):
    """Aynı soru için genel-LLM vs grounded-RAG karşılaştırması.

    'Şok edici' tablo için: genel LLM kaynak göstermez ve halüsinasyon riski yüksektir;
    grounded RAG kaynak gösterir ve faithfulness ölçülür.
    """
    grounded = generate_answer(llm, question, retrieved, grounded=True)
    plain = generate_answer(llm, question, retrieved, grounded=False, abstain_check=False)

    context = build_context(retrieved)
    g_faith = faithfulness_score(nli, grounded["answer"], context) if not grounded["abstained"] else None
    p_faith = faithfulness_score(nli, plain["answer"], context)
    g_cite = citation_accuracy(grounded["answer"], retrieved, relevant_doc_ids)
    p_cite = citation_accuracy(plain["answer"], retrieved, relevant_doc_ids)

    return {
        "question": question,
        "grounded": {"answer": grounded["answer"], "abstained": grounded["abstained"],
                     "faithfulness": g_faith, "citation": g_cite},
        "plain": {"answer": plain["answer"], "faithfulness": p_faith, "citation": p_cite},
    }


# ---------- HF pipeline wrapper'ları (Colab'de kullanılır) ----------

def make_hf_llm(model_name, device="cuda", load_in_4bit=True):
    """HF causal LM'i callable bir LLM'e çevir (Colab A100/T4 için 4-bit)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    def _llm(prompt, max_new_tokens=350):
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=3500).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 temperature=0.0, pad_token_id=tok.eos_token_id)
        gen = out[0][enc["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True)
    return _llm


def make_hf_nli(model_name="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                device="cuda"):
    """Multilingual NLI'yi callable'a çevir (Türkçe destekli)."""
    from transformers import pipeline
    pipe = pipeline("text-classification", model=model_name, device=0 if device == "cuda" else -1,
                    top_k=None, truncation=True, max_length=512)

    label_map = {"ENTAILMENT": "entailment", "NEUTRAL": "neutral", "CONTRADICTION": "contradiction",
                 "entailment": "entailment", "neutral": "neutral", "contradiction": "contradiction"}

    def _nli(premise, hypothesis):
        res = pipe({"text": premise, "text_pair": hypothesis})
        return {label_map.get(r["label"], r["label"]): r["score"] for r in res}
    return _nli


if __name__ == "__main__":
    print("generative_rag.py — Colab notebook'undan çağrılır.")
    print("  - generate_answer() : grounded/plain")
    print("  - should_abstain()  : güvenli kaçınma")
    print("  - faithfulness_score(), citation_accuracy()")
    print("  - safety_comparison() : LLM vs RAG güvenlik tablosu")
