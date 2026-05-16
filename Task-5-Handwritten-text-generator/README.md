# ✍️ Handwritten Text Generator — Character-level LSTM

> **CodSoft Machine Learning Internship · Task 5**  
> A clean, well-trained character-level RNN that generates Shakespeare-style text.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) 
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33%2B-FF4B4B)

---

## 📖 Overview

This project implements a **character-level LSTM** model trained on the Tiny Shakespeare dataset. The model learns the statistical patterns of Shakespeare’s writing and can generate realistic, creative text when given a starting prompt.

I built this from scratch during the CodSoft ML Internship with a focus on clean code, good documentation, and a beautiful user interface.

### Key Features
- **Flexible Architecture**: Supports LSTM, GRU, and vanilla RNN
- **Advanced Sampling**: Temperature, Top-k, and Nucleus (Top-p) sampling
- **Beautiful Web Demo**: Streamlit app with handwritten font styling
- **Resume Training & Checkpointing**
- **Caching System**: Fast reload after first run

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repo
git clone <your-repo-url>
cd handwritten-text-rnn-humanized

# Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate # Linux/Mac

pip install -r requirements.txt
---

## Project Structure (clean & simple)

```
handwritten-text-rnn-humanized/
├── src/
│   ├── __init__.py
│   ├── model.py          # CharRNN class (LSTM/GRU/RNN)
│   ├── data_loader.py    # HF download + vocab + sliding window dataset
│   ├── train.py          # training loop + checkpointing + resume
│   ├── generate.py       # sampling with temp / top-k / top-p
│   └── utils.py          # device, seed, plots
│
├── app.py                # Streamlit demo (my favorite part)
├── config.py             # all hyperparameters in one spot
├── requirements.txt
├── .gitignore
├── README.md
└── char_rnn_exploration.ipynb
```

---
2. Train the Model
Bash# Recommended settings (good quality + reasonable time)
python src/train.py --epochs 30 --hidden_size 256 --num_layers 2 --batch_size 32

# Full powerful model (takes longer)
# python src/train.py --epochs 30
3. Generate Text
Bashpython src/generate.py --seed_text "To be, or not to be" --length 600 --temperature 0.7
4. Launch Web App (Best Experience)
Bashstreamlit run app.py

📊 Training Results
Final Model Configuration:

Hidden Size: 256
Layers: 2
Epochs: 30
Parameters: ~946K

Achieved:

Good convergence with validation perplexity around 2.3


🎮 Generation Examples
Seed: To be, or not to be
Temperature: 0.7
To be, or not to be, that is the question...
Whether 'tis nobler in the mind to suffer...
(Try different seeds and temperatures for creative results!)


⚙️ Configuration
All settings are centralized in config.py. You can easily modify:

Model size (HIDDEN_SIZE, NUM_LAYERS)
Training parameters
Generation settings


🌟 What I Learned

How to properly implement character-level language modeling
Importance of hidden state management in RNNs
Balancing model size vs training time on CPU
Building clean, reusable PyTorch code
Creating an engaging Streamlit demo


👨‍💻 Author
Venkata Sai Ajith
CodSoft Machine Learning Intern

📄 License
MIT License — feel free to use, modify, and learn from this project.

