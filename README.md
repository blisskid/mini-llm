# Mini LLM

A lightweight, educational repository for building, training, and experimenting with Large Language Models from scratch in PyTorch using Jupyter Notebooks.

## Repository Structure

```
mini-llm/
├── chatgpt/
│   ├── lesson1.ipynb          # Lesson 1: Character-level Bigram Language Model
│   └── input.txt              # Toy text dataset
└── gemini/
    ├── mini_gpt.ipynb         # Complete Decoder-only Transformer tutorial on Shakespeare
    ├── prepare_qa_data.ipynb  # Dataset generator for conversational Q&A dialogues
    ├── train.ipynb            # Training pipeline on Q&A data with checkpoint saving
    ├── generate.ipynb         # Interactive Q&A and text generation from checkpoint
    ├── qa_data.txt            # Instruction Q&A dataset
    ├── input.txt              # Tiny Shakespeare dataset
    └── mini_gpt.pt            # Trained model checkpoint
```

## Getting Started

### 1. Requirements

- Python 3.9+
- PyTorch (with Apple Silicon `mps` or Nvidia `cuda` acceleration)
- Jupyter Notebook / JupyterLab / VS Code Jupyter Extension

```bash
pip install torch jupyter
```

### 2. Interactive Notebooks

Open the project in your favorite notebook editor (such as VS Code or JupyterLab):

- **[chatgpt/lesson1.ipynb](chatgpt/lesson1.ipynb)**: Walk through building a Bigram character-level language model.
- **[gemini/mini_gpt.ipynb](gemini/mini_gpt.ipynb)**: Learn the core building blocks of modern LLMs (Causal Multi-Head Self-Attention, Pre-LN, Feed-Forward MLP, Residual connections).
- **[gemini/prepare_qa_data.ipynb](gemini/prepare_qa_data.ipynb)**: Format and generate question-answering training pairs.
- **[gemini/train.ipynb](gemini/train.ipynb)**: Train the 6-layer Transformer on conversational data.
- **[gemini/generate.ipynb](gemini/generate.ipynb)**: Load `mini_gpt.pt` and ask questions directly.
