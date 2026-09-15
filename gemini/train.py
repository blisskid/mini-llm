"""
train.py - Training script and model definition for MiniGPT.
Trains the Transformer on conversational Q&A data or raw text, and saves weights to mini_gpt.pt.
"""

import argparse
import math
import os
import string
import time
import torch
import torch.nn as nn
from torch.nn import functional as F


def get_device():
    """Detect available accelerator: Apple Silicon (mps) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ==========================================
# Transformer Model Architecture
# ==========================================
class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with upper-triangular masking."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Key, Query, Value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask ensures tokens can only attend to prior positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer Block with Pre-LayerNorm and Residuals."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """Decoder-Only Autoregressive Language Model."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 256,
        n_head: int = 8,
        n_layer: int = 6,
        block_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights
        self.tok_emb.weight = self.head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"

        tok_embeddings = self.tok_emb(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_embeddings = self.pos_emb(pos)
        x = self.drop(tok_embeddings + pos_embeddings)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.7, top_k: int = 40, stop_token_ids=None):
        """Autoregressively sample next tokens with optional early stopping."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            if stop_token_ids and idx_next.item() in stop_token_ids:
                break

        return idx


# ==========================================
# Training Pipeline
# ==========================================
def get_batch(data, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size: int, batch_size: int, eval_iters: int, device: str):
    out = {}
    model.eval()
    for split, dataset in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(dataset, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser(description="Train MiniGPT on text or Q&A dataset.")
    parser.add_argument("--data", type=str, default="qa_data.txt", help="Path to training data (default: qa_data.txt).")
    parser.add_argument("--iters", type=int, default=2500, help="Training iterations (default: 2500).")
    parser.add_argument("--block_size", type=int, default=256, help="Context length (default: 256).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4).")
    parser.add_argument("--checkpoint", type=str, default="mini_gpt.pt", help="Checkpoint destination.")
    args = parser.parse_args()

    # Fallback if qa_data.txt is not yet generated
    dataset_path = args.data
    if not os.path.exists(dataset_path):
        if os.path.exists("input.txt"):
            print(f"[*] '{dataset_path}' not found, falling back to 'input.txt'.")
            dataset_path = "input.txt"
        else:
            raise FileNotFoundError(f"Cannot find dataset at {dataset_path}")

    device = get_device()
    print(f"[*] Hardware accelerator: {device.upper()}")
    torch.manual_seed(1337)

    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Combine text characters with standard printable characters so ANY user input can be encoded
    base_chars = set(text)
    standard_printable = set(string.ascii_letters + string.digits + string.punctuation + " \t\n\r")
    chars = sorted(list(base_chars.union(standard_printable)))
    vocab_size = len(chars)

    print(f"[*] Dataset: {dataset_path} ({len(text):,} characters)")
    print(f"[*] Vocabulary size: {vocab_size} unique characters (covers all standard ASCII + symbols)")

    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s if c in stoi]

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Model configuration
    n_embd = 256
    n_head = 8
    n_layer = 6
    dropout = 0.1

    model = MiniGPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=args.block_size,
        dropout=dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[*] Total Model Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-1)

    print("\n" + "=" * 55)
    print("  Starting Training Loop")
    print("=" * 55)
    start_time = time.time()
    eval_interval = 250
    eval_iters = 50

    for it in range(args.iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, args.block_size, args.batch_size, eval_iters, device)
            elapsed = time.time() - start_time
            print(
                f"Step {it:4d}/{args.iters:4d} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    total_time = time.time() - start_time
    print("=" * 55)
    print(f"[*] Training finished in {total_time:.1f} seconds.")

    # Save model checkpoint
    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "block_size": args.block_size,
            "dropout": dropout,
        },
        "chars": chars,
        "is_qa_model": "qa" in dataset_path,
    }
    torch.save(checkpoint, args.checkpoint)
    print(f"[✓] Checkpoint saved successfully to '{args.checkpoint}'.")
    print(f"[✓] You can now use 'python3 generate.py' to ask questions!")


if __name__ == "__main__":
    main()
