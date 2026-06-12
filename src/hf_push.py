"""HuggingFace'e model ve benchmark yayını — somut, atıf alınabilir çıktı.

Yükler:
  1. legal-e5-tr modeli (model card ile)
  2. TurkLegalBench dataset'i (dataset card ile)

Kullanım (Colab'de, HF_TOKEN ile login sonrası):
  from hf_push import push_model, push_benchmark
  push_model("artifacts/legal-e5-tr", "kullanici/legal-e5-tr")
  push_benchmark("artifacts/turklegalbench", "kullanici/TurkLegalBench")
"""
from pathlib import Path


MODEL_CARD = """---
language: tr
license: mit
library_name: sentence-transformers
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- legal
- turkish
- retrieval
pipeline_tag: sentence-similarity
base_model: {base_model}
---

# legal-e5-tr — Türkçe Hukuk Domain-Adaptive Embedding Modeli

`{base_model}` modelinin Türkçe Yargıtay ve Danıştay kararları üzerinde
**Multiple Negatives Ranking Loss + hard negative mining** ile fine-tune edilmiş halidir.
Türkçe hukuki bilgi erişimi (retrieval) için optimize edilmiştir.

## Kullanım

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("{repo_id}")
q = model.encode("query: ihtiyati haciz icra takibi sayılır mı")
d = model.encode("passage: ...karar metni...")
```

## Eğitim Verisi
- Kaynak: [erdem-erdem/Turkish-Law-Documents-700k-clustered](https://huggingface.co/datasets/erdem-erdem/Turkish-Law-Documents-700k-clustered)
- {n_pairs} (sorgu, pasaj) çifti, zayıf-denetimli + hard negatives

## Değerlendirme
[TurkLegalBench]({benchmark_url}) üzerinde değerlendirilmiştir. Sonuçlar için makaleye bakınız.

## Atıf
```bibtex
{bibtex}
```
"""

DATASET_CARD = """---
language: tr
license: cc-by-4.0
task_categories:
- text-retrieval
tags:
- legal
- turkish
- benchmark
- retrieval
- beir
pretty_name: TurkLegalBench
---

# TurkLegalBench — Türkçe Hukuk Retrieval Benchmark'ı

Türkçe hukuki bilgi erişimi (retrieval) için **ilk açık benchmark**. BEIR-uyumlu format.

## Yapı
- `corpus.jsonl` : aranabilir Yargıtay/Danıştay kararları (`_id`, `title`, `text`)
- `queries.jsonl`: hukuki sorgular (`_id`, `text`, `type`: semantic/lexical)
- `qrels.tsv`    : ilgililik etiketleri (`query-id`, `corpus-id`, `score`)

## İstatistik
{stats}

## Sürümler
- **silver**: otomatik üretilmiş (ölçek)
- **gold**: insan-doğrulanmış altküme (kalite, relevance 0/1/2)

## Kaynak Veri
[erdem-erdem/Turkish-Law-Documents-700k-clustered](https://huggingface.co/datasets/erdem-erdem/Turkish-Law-Documents-700k-clustered) — Türk devlet kurumlarının kamuya açık karar arşivleri.

## Atıf
```bibtex
{bibtex}
```
"""

DEFAULT_BIBTEX = """@inproceedings{yargirag2026,
  title     = {YargıRAG: Türkçe Hukuki Yapay Zeka için Açık Benchmark, Domain-Adaptive Retrieval ve Halüsinasyon-Güvenli Soru-Cevap Sistemi},
  author    = {Turanlı, İrem},
  booktitle = {...},
  year      = {2026}
}"""


def push_model(local_dir, repo_id, base_model="intfloat/multilingual-e5-base",
               n_pairs="~5000", benchmark_url="", private=False):
    """Fine-tuned modeli model card ile HF'e yükle."""
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import HfApi

    card = MODEL_CARD.format(base_model=base_model, repo_id=repo_id, n_pairs=n_pairs,
                             benchmark_url=benchmark_url, bibtex=DEFAULT_BIBTEX)
    Path(local_dir, "README.md").write_text(card, encoding="utf-8")

    model = SentenceTransformer(local_dir)
    model.push_to_hub(repo_id, private=private)
    print(f"[hf_push] Model pushed: https://huggingface.co/{repo_id}")


def push_benchmark(local_dir, repo_id, private=False):
    """TurkLegalBench'i dataset card ile HF'e yükle."""
    import json
    from huggingface_hub import HfApi

    stats_path = Path(local_dir, "stats.json")
    stats_str = "—"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats_str = "\n".join(f"- {k}: {v}" for k, v in stats.items())

    card = DATASET_CARD.format(stats=stats_str, bibtex=DEFAULT_BIBTEX)
    Path(local_dir, "README.md").write_text(card, encoding="utf-8")

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="dataset")
    print(f"[hf_push] Benchmark pushed: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    print("hf_push.py — HF login sonrası Colab'den çağrılır.")
    print("  push_model(local_dir, 'kullanici/legal-e5-tr')")
    print("  push_benchmark(local_dir, 'kullanici/TurkLegalBench')")
