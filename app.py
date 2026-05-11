"""
app.py — Opinion Mining Gradio App (AWS Version)
Loads LoRA adapter from local path (downloaded from S3)
"""
import json, os, re, sys, tempfile, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME      = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_PATH    = str(PROJECT_DIR / "final_adapter")   # downloaded from S3
MAX_SEQ_LENGTH  = 512

SENTIMENT_EMOJI = {
    "positive": "✅", "negative": "❌",
    "neutral": "⬜", "conflict": "⚠️",
}
VALID_SENTIMENTS = {"positive", "negative", "neutral", "conflict"}
EXAMPLES = [
    "The laptop's battery life is outstanding and the keyboard feels great, but the display has poor brightness.",
    "Food was absolutely delicious especially the pasta, but the service was painfully slow.",
    "Solid build quality and blazing fast SSD, but the fan noise is distracting.",
    "The camera keeps crashing, the battery drains ridiculously fast, and customer support was unhelpful.",
    "I loved the cozy atmosphere and the friendly staff, though the pasta was a bit too salty.",
]

_base_model = _ft_model = _tokenizer = None
_adapter_loaded = False


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(review: str) -> str:
    return (
        "You are an aspect-based sentiment analysis assistant.\n"
        "Extract all product/service aspects from the review and classify each as: "
        "positive, negative, neutral, or conflict.\n"
        "Return ONLY a JSON object: {\"aspects\": [{\"term\": \"...\", \"sentiment\": \"...\"}]}\n\n"
        f"Review: {review}\n\nJSON:"
    )


# ── Model loading ─────────────────────────────────────────────────────────────
def _bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

def load_models():
    global _base_model, _ft_model, _tokenizer, _adapter_loaded
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("[LOAD] Tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, use_fast=False
    )
    _tokenizer.pad_token    = _tokenizer.eos_token
    _tokenizer.pad_token_id = _tokenizer.eos_token_id
    print("✅ Tokenizer ready")

    # Use CPU if no GPU available (t2.micro)
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    dtype      = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[LOAD] Base model — device: {device_map}")
    if torch.cuda.is_available():
        _base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=_bnb_config(),
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        _base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    _base_model.eval()
    print("✅ Base model ready")

    if os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
        _ft_model = PeftModel.from_pretrained(_base_model, ADAPTER_PATH)
        _ft_model.eval()
        _adapter_loaded = True
        print(f"✅ Adapter loaded from: {ADAPTER_PATH}")
    else:
        print("⚠️  No adapter found — using base model only")
        _ft_model = _base_model
        _adapter_loaded = False


# ── Inference ─────────────────────────────────────────────────────────────────
def _generate(model, prompt: str) -> str:
    device = next(model.parameters()).device
    inputs = _tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=MAX_SEQ_LENGTH
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _parse_aspects(raw: str):
    def _extract(obj):
        if not isinstance(obj, dict): return None
        if "aspects" in obj and isinstance(obj["aspects"], list):
            valid = []
            for a in obj["aspects"]:
                if not isinstance(a, dict): continue
                term = str(a.get("term", "")).strip()
                sent = str(a.get("sentiment", a.get("polarity", ""))).strip().lower()
                if term and sent in VALID_SENTIMENTS:
                    valid.append({"term": term, "sentiment": sent})
            return valid
        flat = []
        for k, v in obj.items():
            if k in ("overall_sentiment", "overall", "sentiment"): continue
            v_lower = str(v).strip().lower()
            if v_lower in VALID_SENTIMENTS:
                flat.append({"term": str(k).strip(), "sentiment": v_lower})
        return flat if flat else None

    def _try(text):
        try:
            obj = json.loads(text)
            r = _extract(obj)
            if r is not None: return r, True
        except Exception: pass
        return None, False

    aspects, valid = _try(raw)
    if aspects is not None:
        return aspects, valid, "✅ Valid JSON"

    for m in re.finditer(r'\{[^{}]*\}|\{.*?\}', raw, re.DOTALL):
        aspects, valid = _try(m.group())
        if aspects is not None:
            return aspects, valid, "✅ Valid JSON"

    return [], False, f"⚠️ Could not parse JSON\n\n**Raw:**\n```\n{raw[:300]}\n```"


def _overall(aspects):
    if not aspects: return "neutral"
    pos = sum(1 for a in aspects if a["sentiment"] == "positive")
    neg = sum(1 for a in aspects if a["sentiment"] == "negative")
    if pos > 0 and neg > 0: return "mixed"
    if pos > neg: return "positive"
    if neg > pos: return "negative"
    return "neutral"


# ── Tab handlers ──────────────────────────────────────────────────────────────
def analyze_single(review: str):
    if not review.strip():
        return "⚠️ Please enter a review.", "{}", "—"
    raw = _generate(_ft_model, build_prompt(review))
    aspects, json_valid, status = _parse_aspects(raw)
    overall = _overall(aspects)
    if aspects:
        lines = ["| Aspect | Sentiment |", "|--------|-----------|"]
        for a in aspects:
            emoji = SENTIMENT_EMOJI.get(a["sentiment"], "•")
            lines.append(f"| **{a['term']}** | {emoji} {a['sentiment']} |")
        aspects_md = "\n".join(lines)
    else:
        aspects_md = status
    json_out  = json.dumps({"aspects": aspects, "overall_sentiment": overall}, indent=2)
    overall_md = f"**{overall.upper()}** {SENTIMENT_EMOJI.get(overall, '')}"
    return aspects_md, json_out, overall_md


def compare_models(review: str):
    if not review.strip():
        return "⚠️ Please enter a review.", "⚠️ Please enter a review."
    prompt = build_prompt(review)
    def _fmt(model):
        raw = _generate(model, prompt)
        aspects, json_valid, status = _parse_aspects(raw)
        overall = _overall(aspects)
        md = f"**JSON Valid:** {'✅ Yes' if json_valid else '❌ No'}\n\n"
        if aspects:
            md += "| Aspect | Sentiment |\n|--------|----------|\n"
            for a in aspects:
                md += f"| **{a['term']}** | {SENTIMENT_EMOJI.get(a['sentiment'],'•')} {a['sentiment']} |\n"
            md += f"\n**Overall:** {overall.upper()}"
        else:
            md += status
        return md
    return _fmt(_base_model), _fmt(_ft_model)


def process_batch(file_obj, progress=gr.Progress()):
    if file_obj is None:
        return None, "⚠️ No file uploaded."
    lines   = Path(file_obj.name).read_text(encoding="utf-8").splitlines()
    reviews = [l.strip() for l in lines if l.strip()]
    if not reviews:
        return None, "⚠️ File is empty."
    results = []
    for review in progress.tqdm(reviews, desc="Processing"):
        raw     = _generate(_ft_model, build_prompt(review))
        aspects, json_valid, _ = _parse_aspects(raw)
        results.append({
            "review": review, "aspects": aspects,
            "overall_sentiment": _overall(aspects),
            "json_valid": json_valid,
        })

    # Save to S3 uploaded-reviews folder
    try:
        import boto3, datetime
        s3 = boto3.client("s3")
        bucket = os.environ.get("S3_BUCKET", "opinion-mining-artifacts")
        key    = f"uploaded-reviews/batch_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        body   = "\n".join(json.dumps(r) for r in results)
        s3.put_object(Bucket=bucket, Key=key, Body=body)
        print(f"✅ Saved to s3://{bucket}/{key}")
    except Exception as e:
        print(f"⚠️ S3 upload skipped: {e}")

    out = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in results:
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
    out.close()

    n_valid = sum(1 for r in results if r["json_valid"])
    counts  = {}
    for r in results:
        counts[r["overall_sentiment"]] = counts.get(r["overall_sentiment"], 0) + 1
    summary = (
        f"### ✅ Done — {len(results)} reviews processed\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Total reviews | {len(results)} |\n"
        f"| JSON valid | {n_valid} ({100*n_valid//len(results)}%) |\n"
    )
    for s, c in sorted(counts.items()):
        summary += f"| {s.capitalize()} | {c} |\n"
    return out.name, summary


# ── UI ────────────────────────────────────────────────────────────────────────
def build_ui():
    note = "✅ LoRA adapter loaded" if _adapter_loaded else "⚠️ Base model only"
    with gr.Blocks(title="Opinion Mining", theme=gr.themes.Soft(primary_hue="teal")) as demo:
        gr.Markdown(f"""
# 🔍 Opinion Mining — Aspect-Based Sentiment Analysis
**Fine-tuned LLaMA-3.2-3B · LoRA (QLoRA) · Deployed on AWS EC2** &nbsp;|&nbsp; {note}

| Accuracy **76%** | F1-Score **64%** | JSON Validity **99.8%** | Trainable params **0.13%**
""")
        with gr.Tabs():
            with gr.Tab("📝 Single Review"):
                with gr.Row():
                    with gr.Column(scale=3):
                        inp1 = gr.Textbox(label="Review text", lines=4,
                            placeholder="e.g. The battery life is great but the screen is too dim…")
                        gr.Examples(examples=EXAMPLES, inputs=inp1)
                        btn1 = gr.Button("Analyze", variant="primary")
                    with gr.Column(scale=2):
                        overall1 = gr.Markdown(label="Overall Sentiment")
                        aspects1 = gr.Markdown(label="Extracted Aspects")
                json1 = gr.Code(label="JSON Output", language="json", lines=10)
                btn1.click(analyze_single, inputs=inp1, outputs=[aspects1, json1, overall1])

            with gr.Tab("⚖️ Base vs Fine-Tuned"):
                inp2 = gr.Textbox(label="Review text", lines=3, value=EXAMPLES[0])
                btn2 = gr.Button("Compare both models", variant="primary")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ✗ Base Model (zero-shot)")
                        base_out = gr.Markdown()
                    with gr.Column():
                        gr.Markdown("### ✓ LoRA Fine-Tuned")
                        ft_out = gr.Markdown()
                btn2.click(compare_models, inputs=inp2, outputs=[base_out, ft_out])

            with gr.Tab("📦 Batch Mode"):
                gr.Markdown("Upload a `.txt` file — one review per line.")
                file_in  = gr.File(label="Upload (.txt)", file_types=[".txt"])
                btn3     = gr.Button("Process all reviews", variant="primary")
                with gr.Row():
                    summary3 = gr.Markdown()
                    file_out = gr.File(label="Download predictions (.jsonl)")
                btn3.click(process_batch, inputs=file_in, outputs=[file_out, summary3])

        gr.Markdown("*Deployed on AWS EC2 · Model artifacts stored on S3 · Monitored via CloudWatch*")
    return demo


if __name__ == "__main__":
    load_models()
    build_ui().launch(
        server_port=7860,
        server_name="0.0.0.0",
        show_error=True,
    )