"""
mini_gpt.py - A self-contained, ~150-line Decoder-only Transformer LLM in PyTorch.
Architecture: GPT-2 style (Pre-LayerNorm, Causal Multi-Head Attention, Residuals).
"""

import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

# ==========================================
# 1. Hyperparameters & Hardware Configuration
# ==========================================
batch_size = 64        # How many independent sequences to process in parallel
block_size = 128       # Maximum context length (tokens the model can look back)
max_iters = 1500       # Total training iterations (~1-2 minutes on Mac MPS)
eval_interval = 250    # How often to evaluate train/val loss
learning_rate = 3e-4   # AdamW learning rate
eval_iters = 100       # Number of batches to average for evaluation
n_embd = 192           # Embedding dimension (each token is a vector of this size)
n_head = 6             # Number of attention heads (n_embd // n_head = 32 dim per head)
n_layer = 4            # Number of Transformer blocks
dropout = 0.1          # Dropout rate for regularization

# Device selection: Apple Silicon (mps) > Nvidia (cuda) > CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"[*] Using hardware accelerator: {device.upper()}")

# Seed for reproducibility
torch.manual_seed(1337)

# ==========================================
# 2. Dataset & Character-Level Tokenizer
# ==========================================
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Build vocabulary of unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"[*] Vocabulary size: {vocab_size} unique characters")

# Mappings from characters to integers and vice-versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Split into 90% train, 10% validation
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str):
    """Generate a small batch of inputs X and target next-tokens Y."""
    source = train_data if split == "train" else val_data
    # Pick random starting indices in the text
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    # Target Y is X shifted by exactly one token to the right
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    """Estimate train and validation loss over multiple batches without gradient tracking."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ==========================================
# 3. Transformer Model Architecture
# ==========================================
class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    Computes query-key-value dot products with an upper-triangular mask
    to ensure tokens can only attend to past and current tokens.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Key, Query, Value projections combined into a single linear layer (3x projection)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower-triangular matrix of 1s, shape: (1, 1, block_size, block_size)
        # register_buffer ensures this tensor is saved with the model state but not trained as a parameter
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length (Time), Channels (Embedding dim)

        # Calculate Q, K, V for all heads in batch and move head dimension forward
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, C)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention: Softmax((Q @ K^T) / sqrt(d_k) + Mask) @ V
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # (B, n_head, T, T)
        # Mask future positions with -infinity so their softmax probability becomes 0
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_head, T, head_dim)
        # Re-assemble all head outputs side-by-side: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection back to residual stream
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise Feed-Forward Network: 4x expansion followed by contraction."""

    def __init__(self, n_embd: int, dropout: float):
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
    """Transformer Block with Pre-LayerNorm and Residual Connections."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        # Pre-LayerNorm formulation (GPT-2 style: stable training)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """Complete Decoder-Only Autoregressive Language Model."""

    def __init__(self):
        super().__init__()
        self.block_size = block_size

        # Token and Positional Embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # Stack of Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )

        # Final LayerNorm and Unembedding projection head
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying (standard in GPT): share token embedding and output head weights
        self.tok_emb.weight = self.head.weight

        # Initialize weights
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
        assert T <= self.block_size, f"Context length {T} exceeds block size {self.block_size}"

        # 1. Embed tokens and positions
        tok_embeddings = self.tok_emb(idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T)
        pos_embeddings = self.pos_emb(pos)  # (T, n_embd)
        x = self.drop(tok_embeddings + pos_embeddings)

        # 2. Pass through Transformer blocks
        x = self.blocks(x)

        # 3. Final normalization and logits projection
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        # 4. Compute Cross-Entropy Loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten B and T to (B*T, vocab_size) and (B*T) for cross_entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = 40):
        """
        Autoregressive generation loop:
        Predict next token, append it to context, repeat.
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds the maximum context window
            idx_cond = idx[:, -self.block_size :]

            # Forward the model to get logits for current sequence
            logits, _ = self(idx_cond)
            # Focus only on the last time step: (B, vocab_size)
            logits = logits[:, -1, :] / temperature

            # Top-K filtering: keep only top_k highest probability tokens
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ==========================================
# 4. Training Loop
# ==========================================
if __name__ == "__main__":
    model = MiniGPT().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[*] Total Model Parameters: {param_count:,}")

    # AdamW with PyTorch's decoupled weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-1)

    print("\n" + "=" * 50)
    print("  Beginning Training  ")
    print("=" * 50)
    start_time = time.time()

    for iter in range(max_iters + 1):
        # Periodic evaluation
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            elapsed = time.time() - start_time
            print(
                f"Step {iter:4d}/{max_iters:4d} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        # Get batch and backprop
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping prevents exploding gradients in deep networks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    total_time = time.time() - start_time
    print("=" * 50)
    print(f"[*] Training finished in {total_time:.1f} seconds.")

    # ==========================================
    # 5. Text Generation
    # ==========================================
    print("\n" + "=" * 50)
    print("  Model Generation Sample (Unprompted)  ")
    print("=" * 50)
    # Start generation with a newline token
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_indices = model.generate(context, max_new_tokens=400, temperature=0.8, top_k=40)
    generated_text = decode(generated_indices[0].tolist())
    print(generated_text)
    print("=" * 50)

    # Custom prompt example
    prompt = "JULIET:\nO Romeo, wherefore art thou "
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    # Generate 300 new tokens continuation
    generated_indices = model.generate(
        context, 
        max_new_tokens=300, 
        temperature=0.7,   # Lower = more conservative/coherent; Higher = more random/creative
        top_k=40           # Restrict sampling to top-k probable characters
    )
    print(decode(generated_indices[0].tolist()))

