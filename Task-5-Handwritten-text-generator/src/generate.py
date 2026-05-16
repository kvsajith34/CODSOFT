"""
generate.py — Text generation with temperature, top-k, and nucleus (top-p) sampling.

This is where the magic happens after training. You give it a seed like
"To be, or not to be" and it keeps predicting the next character using
the trained model + some sampling tricks to make it less repetitive.
"""

import os
import sys
import argparse
import logging

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model import CharRNN
from src.data_loader import CharVocab
from src.utils import get_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the k highest logits, zero the rest (by setting to -inf)."""
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    threshold = values[..., -1, None]
    logits[logits < threshold] = float("-inf")
    return logits


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling — only keep the smallest set of tokens whose
    cumulative prob >= p. Makes output more diverse than plain top-k."""
    if p <= 0 or p >= 1:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove_mask = cum_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[remove_mask] = float("-inf")
    logits.scatter_(0, sorted_idx, sorted_logits)
    return logits


def sample_next_char(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> int:
    """Scale by temperature, apply filters, then sample from the distribution."""
    logits = logits.clone().float()

    if temperature != 1.0:
        logits /= temperature

    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def generate(
    model: CharRNN,
    vocab: CharVocab,
    seed_text: str,
    length: int = 500,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.9,
    device: torch.device = None,
) -> str:
    """
    The main generation function. Feeds the seed through the model once to
    warm up the hidden state, then autoregressively samples one char at a time.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    generated = list(seed_text)

    indices = vocab.encode(seed_text)
    if not indices:
        logger.warning("Seed text contains no known characters — using a random start.")
        indices = [0]

    with torch.no_grad():
        inp = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
        hidden = model.init_hidden(1, device)
        _, hidden = model(inp, hidden)

        inp = torch.tensor([[indices[-1]]], dtype=torch.long).to(device)
        for _ in range(length):
            logits, hidden = model(inp, hidden)
            logits_step = logits[0, -1]
            next_idx = sample_next_char(logits_step, temperature, top_k, top_p)
            generated.append(vocab.idx2char[next_idx])
            inp = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return "".join(generated)


def parse_args():
    p = argparse.ArgumentParser(description="Generate text with the trained Char-RNN")
    p.add_argument(
        "--checkpoint", default=os.path.join(config.CHECKPOINT_DIR, "best.pt")
    )
    p.add_argument(
        "--vocab_path", default=os.path.join(config.DATA_CACHE_DIR, "vocab.pkl")
    )
    p.add_argument("--seed_text", default=config.GEN_SEED_TEXT)
    p.add_argument("--length", type=int, default=config.GEN_LENGTH)
    p.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    p.add_argument("--top_k", type=int, default=config.TOP_K)
    p.add_argument("--top_p", type=float, default=config.TOP_P)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--output", default=None, help="Save output to file")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    if not os.path.exists(args.vocab_path):
        logger.error("Vocab not found at %s — run train.py first.", args.vocab_path)
        sys.exit(1)
    vocab = CharVocab.load(args.vocab_path)

    if not os.path.exists(args.checkpoint):
        logger.error("Checkpoint not found at %s — run train.py first.", args.checkpoint)
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    model = CharRNN(
        vocab_size=vocab.size,
        embedding_dim=cfg["embedding_dim"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=0.0,  # no dropout at inference
        model_type=cfg["model_type"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    logger.info(
        "Loaded %s checkpoint (epoch %d, val_loss=%.4f)",
        cfg["model_type"],
        ckpt["epoch"],
        ckpt.get("val_loss", float("nan")),
    )

    text = generate(
        model=model,
        vocab=vocab,
        seed_text=args.seed_text,
        length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )

    print("\n" + "─" * 60)
    print(text)
    print("─" * 60 + "\n")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Output saved to %s", args.output)


if __name__ == "__main__":
    main()
