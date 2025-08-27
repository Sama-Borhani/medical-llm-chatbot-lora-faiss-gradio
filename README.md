
# Medical LLM Chatbot — LoRA-ready + FAISS + Gradio

> **Educational demo only — not medical advice.**  
> A retrieval-augmented medical Q&A chatbot using a base LLM (Camel-5B), semantic search with **FAISS**, cross-encoder **reranking**, and a **Gradio** UI. The app can ingest PDFs, fall back to a bundled text corpus, and generates answers in a structured format.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XhOOYXc-XsYsYXw3OLRSpuQ5cRKHlmax?authuser=1#scrollTo=3P37O5Lg6bzn)

---

##  What this repo does

- Loads `Writer/camel-5b-hf` with **4-bit** quantization (falls back to **8-bit** if needed)  
- Downloads/reads **medical PDFs** and extracts text with **PyPDF2** (graceful fallbacks)  
- Splits text into **semantic chunks**, embeds with `sentence-transformers`, and builds a **FAISS** index  
- **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to refine top candidates  
- Two-pass generation: **outline → expansion** using a **Jinja2** prompt template  
- **Gradio** chat UI with examples and a safety disclaimer

---

##  Architecture (high level)

```

User → Gradio UI → retrieve top-k with FAISS → rerank with cross-encoder
→ compose structured prompt (Jinja2)
→ Camel-5B (quantized) → Outline pass → Expansion pass → Answer (+ sources)

```

---

##  Repository structure

```

.
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ .env.example
├─ notebooks/
│  └─ 04\_inference\_gradio.ipynb        # optional Colab runner
├─ corpus/                              # approved .txt notes (do NOT commit PHI/private data)
├─ index/                               # FAISS files created by the build step
├─ scripts/
│  └─ build\_faiss.py                    # builds FAISS index from /corpus
└─ src/
└─ app.py                            # Gradio app (loads model + index + reranker)

````

---

##  Setup

**Python 3.10+** (Colab is fine)

```bash
pip install -r requirements.txt
````

Create your environment file and set secrets (never hard-code tokens in code):

```bash
cp .env.example .env
# then edit .env:
# HF_TOKEN=hf_********************************
```

> **Security note:** Replace any hard-coded tokens from your old notebook with `HF_TOKEN` in `.env`. The app reads it at runtime.

---

##  Add a corpus

* Put **approved, license-compliant** medical text files in `./corpus/` (e.g., `guidelines_cvd.txt`, `diabetes_overview.txt`).
* You can also point your own PDF URLs and extract text to `.txt` first; the original notebook used WHO/CDC/AHA URLs with fallbacks and a built-in text snippet.

---

##  Build the FAISS index

```bash
python scripts/build_faiss.py
```

Outputs:

* `index/faiss.index`
* `index/metadata.json`

---

##  Run the Gradio app

```bash
python src/app.py
```

Default at `http://127.0.0.1:7860`. In **Colab**, you’ll see a public `gradio.live` link in the cell output. The UI includes sample questions and app info.

**Optional env overrides:**

```bash
MODEL_ID=Writer/camel-5b-hf TOP_K=5 python src/app.py
```

* `MODEL_ID` — HF model id
* `TOP_K` — retrieval depth (default 4)
* `ADAPTER_PATH` — path to a LoRA adapter if/when you train one

---

##  Prompting template

A structured Jinja2 prompt formats the answer as:

1. **Summary** (2–3 sentences)
2. **Detailed Explanation** (bullet points)
3. **References** (source titles)

Two-pass generation: outline first, then expand for final response.

---

##  Reranking

Initial FAISS neighbors are **reranked** with `cross-encoder/ms-marco-MiniLM-L-6-v2` on the model device to improve relevance before composing the prompt.

---

##  Limitations & Safety

* **Not a medical device.** **Educational use only.**
* Avoid private/PHI data; keep `/corpus` content approved and anonymized.
* Answers may be incomplete/incorrect — always verify with primary sources.

---

## Troubleshooting

* **CUDA OOM** → the app loads in 4-bit; if it fails, it automatically retries 8-bit. Close other sessions or use a smaller model.
* **No PDFs / empty corpus** → the app falls back to a bundled text snippet; add `.txt` files under `/corpus` for better answers.

---

## License

Content in `/corpus` must follow its original license. This project does **not** provide medical advice.

```
```
