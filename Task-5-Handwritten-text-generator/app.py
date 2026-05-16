"""
app.py — Streamlit web demo for the character-level RNN.

I wanted a nice interface so people could play with the model without touching code.
Run with: streamlit run app.py
"""

import os
import sys
import time
import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.model import CharRNN
from src.data_loader import CharVocab
from src.generate import generate
from src.utils import get_device

# Page setup
st.set_page_config(
    page_title="✍️ Handwritten Text Generator",
    page_icon="✍️",
    layout="wide",
)

# Handwritten-style CSS — I spent way too long tweaking the fonts
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@400;700&family=Indie+Flower&display=swap');

  html, body, [class*="css"] { font-family: 'Caveat', cursive; }

  .main-title {
      font-family: 'Caveat', cursive;
      font-size: 3.2rem;
      font-weight: 700;
      color: #2c3e50;
      text-align: center;
      margin-bottom: 0.2rem;
  }
  .subtitle {
      text-align: center;
      color: #7f8c8d;
      font-size: 1.2rem;
      margin-bottom: 2rem;
  }
  .output-box {
      font-family: 'Indie Flower', cursive;
      font-size: 1.35rem;
      line-height: 2.1;
      color: #1a1a2e;
      background: #fdf6e3;
      border-left: 4px solid #e67e22;
      padding: 1.5rem 2rem;
      border-radius: 4px;
      box-shadow: 2px 3px 8px rgba(0,0,0,0.08);
      white-space: pre-wrap;
      min-height: 200px;
  }
  .stButton > button {
      font-family: 'Caveat', cursive;
      font-size: 1.1rem;
      background: #e67e22;
      color: white;
      border: none;
      border-radius: 6px;
      padding: 0.5rem 2rem;
  }
  .stButton > button:hover { background: #d35400; }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    '<div class="main-title">✍️ Handwritten Text Generator</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">My character-level LSTM trained on Tiny Shakespeare<br>'
    'CodSoft ML Internship — Task 5</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# Load model (cached so it doesn't reload every time you tweak sliders)
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    vocab_path = os.path.join(config.DATA_CACHE_DIR, "vocab.pkl")
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best.pt")

    if not os.path.exists(vocab_path):
        return None, None, "vocab_missing"
    if not os.path.exists(ckpt_path):
        return None, None, "ckpt_missing"

    device = get_device(config.DEVICE)
    vocab = CharVocab.load(vocab_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    model = CharRNN(
        vocab_size=vocab.size,
        embedding_dim=cfg["embedding_dim"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=0.0,
        model_type=cfg["model_type"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    info = {
        "epoch": ckpt.get("epoch", "?"),
        "val_loss": ckpt.get("val_loss", float("nan")),
        "params": f"{model.count_parameters():,}",
        "type": cfg["model_type"],
        "layers": cfg["num_layers"],
        "hidden": cfg["hidden_size"],
        "vocab": vocab.size,
    }
    return model, vocab, info


model, vocab, model_info = load_model()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Generation Settings")

    seed_text = st.text_area(
        "Seed Text (primer)", value=config.GEN_SEED_TEXT, height=80
    )
    gen_length = st.slider(
        "Characters to generate", 100, 2000, config.GEN_LENGTH, step=50
    )
    temperature = st.slider(
        "Temperature",
        0.3,
        1.5,
        config.TEMPERATURE,
        step=0.05,
        help="Lower = more predictable, Higher = more creative (and sometimes weird)",
    )
    top_k = st.slider("Top-k (0 = off)", 0, 50, config.TOP_K, step=1)
    top_p = st.slider(
        "Top-p (nucleus)",
        0.0,
        1.0,
        config.TOP_P,
        step=0.05,
        help="Keep the smallest set of tokens with cumulative prob ≥ p",
    )

    st.markdown("---")
    st.header("ℹ️ Model Info")

    if isinstance(model_info, dict):
        st.markdown(f"**Type:** {model_info['type']}")
        st.markdown(f"**Layers:** {model_info['layers']}")
        st.markdown(f"**Hidden size:** {model_info['hidden']}")
        st.markdown(f"**Vocab size:** {model_info['vocab']}")
        st.markdown(f"**Parameters:** {model_info['params']}")
        st.markdown(f"**Trained epochs:** {model_info['epoch']}")
        st.markdown(f"**Val loss:** {model_info['val_loss']:.4f}")
    else:
        st.info("Model not loaded yet. Run training first!")

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    generate_btn = st.button("✍️ Generate Text", use_container_width=True)

with col2:
    stream_mode = st.toggle("Stream output (character by character)", value=True)

# Status messages
if model is None:
    if model_info == "vocab_missing":
        st.error(
            "**Vocabulary not found.** Run `python src/train.py` first. "
            "It'll create the vocab automatically."
        )
    elif model_info == "ckpt_missing":
        st.error(
            "**Checkpoint not found.** Train the model first with `python src/train.py`."
        )
    st.info(
        "**Quick start:**\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "python src/train.py --epochs 30\n"
        "streamlit run app.py\n"
        "```"
    )
    st.stop()

# Generation
output_placeholder = st.empty()
output_placeholder.markdown(
    '<div class="output-box" style="color:#aaa;">Your generated text will appear here …</div>',
    unsafe_allow_html=True,
)

if generate_btn:
    device = get_device(config.DEVICE)

    if stream_mode:
        with st.spinner("Generating …"):
            generated = list(seed_text) if seed_text else ["T"]
            indices = vocab.encode(seed_text) if seed_text else [0]
            if not indices:
                indices = [0]

            import torch.nn.functional as F
            from src.generate import sample_next_char

            model.eval()
            with torch.no_grad():
                inp = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
                hidden = model.init_hidden(1, device)
                _, hidden = model(inp, hidden)
                inp = torch.tensor([[indices[-1]]], dtype=torch.long).to(device)

                for i in range(gen_length):
                    logits, hidden = model(inp, hidden)
                    logits_step = logits[0, -1]
                    next_idx = sample_next_char(logits_step, temperature, top_k, top_p)
                    generated.append(vocab.idx2char[next_idx])
                    inp = torch.tensor([[next_idx]], dtype=torch.long).to(device)

                    if i % 20 == 0:
                        display = "".join(generated)
                        output_placeholder.markdown(
                            f'<div class="output-box">{display}▌</div>',
                            unsafe_allow_html=True,
                        )

            final_text = "".join(generated)
    else:
        with st.spinner("Generating …"):
            t0 = time.time()
            final_text = generate(
                model=model,
                vocab=vocab,
                seed_text=seed_text or "T",
                length=gen_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
            )
            elapsed = time.time() - t0

    output_placeholder.markdown(
        f'<div class="output-box">{final_text}</div>',
        unsafe_allow_html=True,
    )

    # Stats
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Characters generated", gen_length)
    c2.metric("Temperature", temperature)
    c3.metric("Top-k", top_k if top_k > 0 else "off")
    c4.metric("Top-p", top_p if top_p > 0 else "off")

    st.download_button(
        "💾 Download Generated Text",
        data=final_text,
        file_name="generated_text.txt",
        mime="text/plain",
    )
