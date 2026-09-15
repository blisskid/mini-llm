# Build a language model: notebook lessons

Based on the [shared reference conversation](https://chatgpt.com/share/6aa56bca-4a1c-83e9-9153-1edcc7ff7e40). The reference contains a complete Lesson 1, explicit plans for Lessons 2–3, and a broader roadmap. Lessons 2–12 are original teaching implementations developed from that roadmap, not additional lessons copied from the conversation.

## Run the lessons

1. Open this folder in VS Code with notebook support, or start JupyterLab from this folder or the repository root.
2. Select a Python kernel with PyTorch installed. A fresh environment can use `python -m pip install torch jupyterlab`.
3. Open a notebook and run cells from top to bottom with **Shift+Enter**. No notebook depends on another notebook's running kernel.
4. Restart the kernel and rerun from the top for a fresh experiment. Rerunning a training cell generally continues training its current model; rerun model creation to reset weights.

Only PyTorch and the Python standard library are used by lesson code. Lesson 7 uses the modern `torch.amp` API. CPU works for every lesson; Lesson 7 selects CUDA or MPS when available. Lesson 12 writes its checkpoint under `artifacts/`.

## Course index

| Notebook | Topic | What you will build |
| --- | --- | --- |
| [Lesson 1](lesson1.ipynb) | Bigram language model | Tokenize text, train a first model, generate characters |
| [Lesson 2](lesson2.ipynb) | Mini-batches and validation | Random windows, shifted targets, held-out loss |
| [Lesson 3](lesson3.ipynb) | Causal self-attention | Queries, keys, values, masks, multiple heads |
| [Lesson 4](lesson4.ipynb) | Tiny GPT | Embeddings, Transformer blocks, training and generation |
| [Lesson 5](lesson5.ipynb) | Sampling | Temperature, top-k, top-p |
| [Lesson 6](lesson6.ipynb) | Byte-pair encoding | Learn merges and round-trip UTF-8 text |
| [Lesson 7](lesson7.ipynb) | Training efficiency | Parameter memory, devices, accumulation, mixed precision |
| [Lesson 8](lesson8.ipynb) | Instruction tuning | Prompt formatting and answer-only loss |
| [Lesson 9](lesson9.ipynb) | LoRA | Freeze a base model, train adapters, merge weights |
| [Lesson 10](lesson10.ipynb) | Quantization and evaluation | Int8 weight storage, held-out loss and perplexity |
| [Lesson 11](lesson11.ipynb) | Retrieval and grounded answers | TF-IDF retrieval, evidence, RAG prompt construction |
| [Lesson 12](lesson12.ipynb) | Save, reload, and serve | Checkpoint metadata and a local prediction handler |

## Scope and expectations

The existing `input.txt` is deliberately tiny. The notebooks run short experiments so you can inspect each step; fluent generation needs much more data and training. Validation on 16 characters is an illustration, not a reliable quality benchmark. SFT and retrieval lessons use clearly identified synthetic examples. No pretrained models, paid APIs, dataset downloads, or cloud deployments are required.

Lessons 2–5 build the core GPT path. Lessons 6–12 explore the roadmap's later topics. The scaling lesson estimates a larger configuration without allocating or training it. The quantization lesson demonstrates int8 storage with float computation. The retrieval lesson builds a RAG prompt and returns extractive evidence; it does not pretend the tiny GPT can answer instructions. The final lesson prepares a local handler; deploying infrastructure remains a separate exercise.

The original `lesson1.py` and its notebook are preserved.
