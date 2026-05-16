import sys
sys.path.insert(0, '.')

import config
from src.model import CharRNN
from src.data_loader import load_data
from src.utils import get_device

print("✅ All imports good")
print("Dataset:", config.HF_DATASET_NAME)
print("Device:", get_device())