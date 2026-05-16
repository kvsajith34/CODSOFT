"""
train.py — Training script with checkpointing, LR decay, and nice logging.

Run it like:
    python src/train.py
    python src/train.py --epochs 10 --hidden_size 256
    python src/train.py --resume

I added the resume flag because I kept killing training runs by accident.
"""

import os
import sys
import glob
import time
import logging
import argparse

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

# So we can run "python src/train.py" from project root and still import src.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import load_data
from src.model import CharRNN
from src.utils import get_device, set_seed, format_time, save_training_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_epoch(model, loader, optimizer, criterion, device, clip):
    """One epoch of training with truncated BPTT (we detach hidden each batch)."""
    model.train()
    total_loss = 0.0
    hidden = None

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        if hidden is not None:
            hidden = model.detach_hidden(hidden)
        else:
            hidden = model.init_hidden(x.size(0), device)

        optimizer.zero_grad()
        logits, hidden = model(x, hidden)

        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        # IMPORTANT: Always initialize fresh hidden state for each batch in validation
        hidden = model.init_hidden(x.size(0), device)

        logits, hidden = model(x, hidden)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()

    return total_loss / len(loader)


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, vocab, args):
    """Save full training state so we can resume later."""
    ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch:04d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "vocab_size": vocab.size,
            "config": {
                "embedding_dim": args.embedding_dim,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "model_type": args.model_type,
            },
        },
        ckpt_path,
    )
    # Always keep a fresh "best.pt" copy
    best_path = os.path.join(args.checkpoint_dir, "best.pt")
    torch.save(torch.load(ckpt_path), best_path)
    logger.info("Checkpoint saved → %s", ckpt_path)
    return ckpt_path


def load_latest_checkpoint(checkpoint_dir):
    """Grab the most recent epoch checkpoint for resuming."""
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*.pt")))
    if not checkpoints:
        return None
    return checkpoints[-1]


def parse_args():
    p = argparse.ArgumentParser(description="Train the Char-RNN model")
    p.add_argument("--dataset", default=config.HF_DATASET_NAME)
    p.add_argument("--max_chars", type=int, default=config.MAX_CHARS)
    p.add_argument("--seq_length", type=int, default=config.SEQ_LENGTH)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--lr_decay", type=float, default=config.LR_DECAY)
    p.add_argument("--clip", type=float, default=config.CLIP_GRAD)
    p.add_argument("--embedding_dim", type=int, default=config.EMBEDDING_DIM)
    p.add_argument("--hidden_size", type=int, default=config.HIDDEN_SIZE)
    p.add_argument("--num_layers", type=int, default=config.NUM_LAYERS)
    p.add_argument("--dropout", type=float, default=config.DROPOUT)
    p.add_argument(
        "--model_type", default=config.MODEL_TYPE, choices=["LSTM", "GRU", "RNN"]
    )
    p.add_argument("--valid_split", type=float, default=config.VALID_SPLIT)
    p.add_argument("--save_every", type=int, default=config.SAVE_EVERY)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--checkpoint_dir", default=config.CHECKPOINT_DIR)
    p.add_argument("--output_dir", default=config.OUTPUT_DIR)
    p.add_argument("--cache_dir", default=config.DATA_CACHE_DIR)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    set_seed(args.seed)

    logger.info("Device: %s", device)
    logger.info(
        "Model : %s | layers=%d | hidden=%d | emb=%d",
        args.model_type,
        args.num_layers,
        args.hidden_size,
        args.embedding_dim,
    )

    # Load data (uses cache if available — huge time saver)
    train_loader, valid_loader, vocab = load_data(
        dataset_name=args.dataset,
        max_chars=args.max_chars,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        valid_split=args.valid_split,
        cache_dir=args.cache_dir,
    )

    # Build model
    model = CharRNN(
        vocab_size=vocab.size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_type=args.model_type,
    ).to(device)

    logger.info("Parameters: %s", f"{model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")

    # Resume from last checkpoint if flag is set
    if args.resume:
        ckpt_path = load_latest_checkpoint(args.checkpoint_dir)
        if ckpt_path:
            logger.info("Resuming from %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optim_state"])
            scheduler.load_state_dict(ckpt["sched_state"])
            start_epoch = ckpt["epoch"] + 1
            best_val = ckpt.get("val_loss", float("inf"))
        else:
            logger.warning("No checkpoint found — starting from scratch.")

    logger.info("Starting training … (epochs %d → %d)", start_epoch, args.epochs)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, args.clip
        )
        val_loss = validate(model, valid_loader, criterion, device)
        scheduler.step()

        elapsed = format_time(time.time() - t0)
        lr_now = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr_now)

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | perplexity=%.2f | lr=%.6f | %s",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            2**val_loss,
            lr_now,
            elapsed,
        )

        # Save if it's time or if we hit a new best
        if epoch % args.save_every == 0 or val_loss < best_val:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_loss, val_loss, vocab, args
            )
            if val_loss < best_val:
                best_val = val_loss
                logger.info("★ New best val_loss: %.4f", best_val)

    save_training_history(history, args.output_dir)
    logger.info(
        "Training complete. Best val_loss: %.4f | Perplexity: %.2f",
        best_val,
        2**best_val,
    )


if __name__ == "__main__":
    main()
