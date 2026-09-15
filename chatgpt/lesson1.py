import torch
import torch.nn as nn
from torch.nn import functional as F


# --------------------------------------------------
# 1. Load training text
# --------------------------------------------------

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("First 200 characters:")
print(text[:200])

print("\nText length:", len(text))


# --------------------------------------------------
# 2. Build a character-level tokenizer
# --------------------------------------------------

chars = sorted(list(set(text)))

vocab_size = len(chars)

print("\nCharacters:")
print(chars)

print("\nVocabulary size:", vocab_size)


# string -> integer
stoi = {
    ch: i
    for i, ch in enumerate(chars)
}

# integer -> string
itos = {
    i: ch
    for i, ch in enumerate(chars)
}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


# Test tokenizer
test_text = "hello"

print("\nTokenizer test:")
print("Original:", test_text)
print("Encoded :", encode(test_text))
print("Decoded :", decode(encode(test_text)))


# --------------------------------------------------
# 3. Convert the whole dataset to token IDs
# --------------------------------------------------

data = torch.tensor(
    encode(text),
    dtype=torch.long
)

print("\nFirst 20 token IDs:")
print(data[:20])

print("Data shape:", data.shape)


# --------------------------------------------------
# 4. Create input / target pairs
#
# Example:
#
# x = [a, b, c]
# y = [b, c, d]
#
# The model learns:
#
# a -> b
# b -> c
# c -> d
# --------------------------------------------------

x = data[:-1]
y = data[1:]

print("\nFirst 10 input tokens:")
print(x[:10])

print("\nFirst 10 target tokens:")
print(y[:10])


# --------------------------------------------------
# 5. Bigram Language Model
# --------------------------------------------------

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # Each token points to a row containing
        # scores for every possible next token.
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            vocab_size
        )

    def forward(self, idx, targets=None):

        # idx shape:
        # [number_of_tokens]
        #
        # logits shape:
        # [number_of_tokens, vocab_size]

        logits = self.token_embedding_table(idx)

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss


# --------------------------------------------------
# 6. Create the model
# --------------------------------------------------

model = BigramLanguageModel(vocab_size)


# Test model before training
logits, loss = model(x, y)

print("\nInitial loss:")
print(loss.item())


# --------------------------------------------------
# 7. Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-2
)


# --------------------------------------------------
# 8. Training
# --------------------------------------------------

training_steps = 5000

for step in range(training_steps):

    # Forward pass
    logits, loss = model(x, y)

    # Remove gradients from previous training step
    optimizer.zero_grad()

    # Calculate gradients
    loss.backward()

    # Update weights
    optimizer.step()

    if step % 500 == 0:
        print(
            f"step {step}: "
            f"loss = {loss.item():.4f}"
        )


# --------------------------------------------------
# 9. Text generation
# --------------------------------------------------

@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens
):

    for _ in range(max_new_tokens):

        # Predict next-token logits
        logits, _ = model(idx)

        # We only care about prediction
        # after the final token
        logits = logits[-1]

        # Convert logits to probabilities
        probs = F.softmax(
            logits,
            dim=-1
        )

        # Randomly sample next token
        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        # Add next token to sequence
        idx = torch.cat(
            [idx, next_token]
        )

    return idx


# --------------------------------------------------
# 10. Generate some text
# --------------------------------------------------

# Start generation with character "h"
start = torch.tensor(
    [encode("h")[0]],
    dtype=torch.long
)

generated = generate(
    model,
    start,
    max_new_tokens=300
)

print("\nGenerated text:")
print(
    decode(
        generated.tolist()
    )
)