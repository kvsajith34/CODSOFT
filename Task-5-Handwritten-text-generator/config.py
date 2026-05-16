"""
config.py — All the knobs and dials in one place.

I hate hunting through code for hyperparameters, so everything lives here.
Change these and the whole project picks them up. Defaults are what worked
well for me on this tiny Shakespeare corpus.
"""

import os

# Where stuff gets saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Dataset — using the HuggingFace version of Karpathy's tiny shakespeare
# It's small but perfect for quick experiments
HF_DATASET_NAME = "Trelis/tiny-shakespeare"
HF_DATASET_SPLIT = "train"
TEXT_COLUMN = "text"
MAX_CHARS = 500_000  # cap it so training doesn't take forever on CPU

# Model architecture — LSTM with 3 layers worked best in my tests
EMBEDDING_DIM = 128
HIDDEN_SIZE = 512
NUM_LAYERS = 3
DROPOUT = 0.3
MODEL_TYPE = "LSTM"  # try "GRU" or "RNN" for comparison

# Training params — these gave ~1.35 train loss after 30 epochs
SEQ_LENGTH = 100
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.002
LR_DECAY = 0.97
CLIP_GRAD = 5.0
SAVE_EVERY = 5
VALID_SPLIT = 0.1

# Generation defaults — temperature 0.8 feels creative but not gibberish
GEN_SEED_TEXT = "To be, or not to be"
GEN_LENGTH = 500
TEMPERATURE = 0.8
TOP_K = 0
TOP_P = 0.9

# Reproducibility & device
RANDOM_SEED = 42
DEVICE = "auto"  # auto picks cuda > mps > cpu
