# Mini LLM

A lightweight, educational repository for building and understanding Large Language Models from scratch in PyTorch.

## Repository Structure

```
mini-llm/
├── chatgpt/
│   ├── lesson1.py         # Lesson 1: Character-level Bigram Language Model
│   ├── lesson1.ipynb      # Interactive Jupyter notebook for Lesson 1
│   └── input.txt          # Toy text dataset
└── gemini/
    ├── train.py           # Training script for GPT-2 Decoder-Only Transformer
    ├── generate.py        # Interactive QA and dialogue text generation
    ├── mini_gpt.py        # All-in-one standalone Transformer script
    ├── prepare_qa_data.py # Generator for conversational instruction Q&A dataset
    ├── qa_data.txt        # Generated Q&A instruction dataset
    ├── input.txt          # Tiny Shakespeare dataset
    └── mini_gpt.pt        # Trained model checkpoint
```

## Getting Started

### 1. Requirements

- Python 3.9+
- PyTorch (with Apple Silicon `mps` or CUDA support)

```bash
pip install torch
```

### 2. Bigram Language Model (`chatgpt/`)

Run the character-level Bigram model tutorial:

```bash
cd chatgpt
python3 lesson1.py
```

### 3. Mini GPT Transformer (`gemini/`)

#### Ask Questions (Interactive Mode)
```bash
cd gemini
python3 generate.py
```

#### Train the Transformer from Scratch
```bash
cd gemini
python3 train.py
```
