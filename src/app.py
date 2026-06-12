"""LegalRAG-TR Gradio demo: Türkçe Yargıtay kararları üzerinde retrieval."""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import gradio as gr
from embedder import load_model, load_index
from retriever import HybridRetriever
from answer import format_answer


_PROJECT_ROOT = Path(__file__).parent.parent
INDEX_DIR = os.environ.get("LEGALRAG_INDEX", str(_PROJECT_ROOT / "artifacts" / "index_small_ft"))
MODEL_NAME = os.environ.get("LEGALRAG_MODEL", str(_PROJECT_ROOT / "artifacts" / "e5-small-tr-legal-ft"))
USE_RERANKER = os.environ.get("LEGALRAG_RERANKER", "0") == "1"


print(f"[app] Loading index from {INDEX_DIR}...")
index, embeddings, records = load_index(INDEX_DIR)
print(f"[app] Loaded {len(records)} chunks, {embeddings.shape[1]}-dim embeddings")

print(f"[app] Loading embedding model {MODEL_NAME}...")
model = load_model(MODEL_NAME)
hybrid = HybridRetriever(index, embeddings, records, model)

if USE_RERANKER:
    print("[app] Loading cross-encoder reranker...")
    from reranker import RerankedRetriever
    retriever = RerankedRetriever(hybrid, fetch_k=15)
    print("[app] Ready (hybrid + reranker).")
else:
    retriever = hybrid
    print("[app] Ready (hybrid retrieval).")


EXAMPLES = [
    "İhtiyati haciz icra takibi sayılır mı?",
    "Karşılıksız yararlanma suçunda beraat hangi koşullarda mümkündür?",
    "İdari yargıda yürütmenin durdurulması koşulları nelerdir?",
    "Boşanma davasında nafaka miktarı neye göre belirlenir?",
    "İş akdinin haksız feshi tazminatı nasıl hesaplanır?",
    "4733 sayılı yasaya muhalefet suçunun unsurları",
    "Kasten öldürme suçunda tasarlama unsuru",
]


def respond(query, top_k):
    if not query or not query.strip():
        return "Lütfen bir soru yazın."
    results = retriever.search(query.strip(), top_k=int(top_k))
    return format_answer(query, results, max_results=int(top_k))


with gr.Blocks(title="LegalRAG-TR — Yargıtay İçtihat Asistanı", theme=gr.themes.Soft()) as demo:
    n_kararlar = len(set(r['karar_id'] for r in records))
    model_label = Path(MODEL_NAME).name if "/" in MODEL_NAME else MODEL_NAME.split('/')[-1]
    reranker_tag = " + cross-encoder reranker" if USE_RERANKER else ""
    gr.Markdown(
        "# ⚖️ LegalRAG-TR\n"
        "### Türkçe Yargıtay & Danıştay Kararları için Kaynaklı Retrieval Sistemi\n"
        f"_İndeks: **{n_kararlar} karar** ({len(records)} chunk) · "
        f"Embedding: `{model_label}` · BM25+dense+RRF hibrit retrieval{reranker_tag}_  \n"
        "_Veri: HuggingFace açık kaynak `erdem-erdem/Turkish-Law-Documents-700k-clustered`_"
    )
    with gr.Row():
        with gr.Column(scale=3):
            query = gr.Textbox(
                label="Hukuki sorunuz",
                placeholder="Örn: İhtiyati haciz icra takibi sayılır mı?",
                lines=2,
            )
        with gr.Column(scale=1):
            top_k = gr.Slider(1, 10, value=5, step=1, label="Sonuç sayısı")
    btn = gr.Button("Ara", variant="primary")
    output = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=query, label="Örnek sorgular")
    btn.click(respond, inputs=[query, top_k], outputs=output)
    query.submit(respond, inputs=[query, top_k], outputs=output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
