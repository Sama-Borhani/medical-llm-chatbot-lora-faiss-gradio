import os, json
import numpy as np
import faiss
import gradio as gr
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# Load env variables
load_dotenv()

MODEL_ID     = os.environ.get("MODEL_ID", "Writer/camel-5b-hf")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "").strip()
TOP_K        = int(os.environ.get("TOP_K", "4"))
INDEX_DIR    = os.environ.get("INDEX_DIR", "./index")
EMBED_MODEL  = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN     = os.environ.get("HF_TOKEN", None)

# --- Load FAISS index ---
index_path = os.path.join(INDEX_DIR, "faiss.index")
meta_path  = os.path.join(INDEX_DIR, "metadata.json")
if not (os.path.exists(index_path) and os.path.exists(meta_path)):
    raise SystemExit("FAISS index not found. Run: python scripts/build_faiss.py")

index = faiss.read_index(index_path)
with open(meta_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

embedder = SentenceTransformer(EMBED_MODEL)

def load_doc(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

# --- Load model ---
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    use_auth_token=HF_TOKEN,
    device_map="auto",
    torch_dtype="auto"
)

# Apply LoRA adapter if given
if ADAPTER_PATH:
    print(f"Loading LoRA adapter from {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

textgen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=384,
    temperature=0.2,
    do_sample=False
)

SYSTEM_PROMPT = (
    "You are a careful medical assistant for educational purposes only. "
    "Answer briefly and cite the provided snippets with [#]. "
    "If unsure or outside scope, say so and suggest consulting a clinician."
)

# --- Retrieval ---
def retrieve(query, k=TOP_K):
    q = embedder.encode([query], normalize_embeddings=True)
    scores, idx = index.search(np.array(q, dtype=np.float32), k)
    items = []
    for j in idx[0].tolist():
        if j < 0 or j >= len(metadata):
            continue
        path = metadata[j]["path"]
        text = load_doc(path)
        items.append((text[:1200], path))  # trim long text
    return items

def compose_prompt(question, snippets):
    numbered = []
    for i, (text, path) in enumerate(snippets, start=1):
        numbered.append(f"[{i}] {text}\n(Source: {os.path.basename(path)})")
    context_block = "\n".join(numbered)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context snippets:\n{context_block}\n\n"
        f"Question: {question}\n"
        f"Answer citing snippets like [1], [2]."
    )

def answer(question):
    if not question.strip():
        return "Please enter a question."
    snippets = retrieve(question, TOP_K)
    prompt = compose_prompt(question, snippets)
    out = textgen(prompt)[0]["generated_text"]
    return out

# --- Gradio UI ---
with gr.Blocks(title="Medical LLM Chatbot (Demo)") as demo:
    gr.Markdown("# Medical LLM Chatbot — Demo\n**Educational use only. Not medical advice.**")
    q = gr.Textbox(label="Ask a medical question")
    btn = gr.Button("Get answer")
    a = gr.Markdown()
    btn.click(answer, q, a)

if __name__ == "__main__":
    demo.launch()

