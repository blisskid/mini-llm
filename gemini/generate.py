"""
generate.py - Inference and QA dialogue script for MiniGPT.
Loads the trained checkpoint (mini_gpt.pt) and answers questions properly.
"""

import argparse
import os
import sys
import torch
from train import MiniGPT, get_device


def load_model(checkpoint_path="mini_gpt.pt"):
    """Load model, character mappings, and metadata from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"\n[!] Error: Checkpoint file '{checkpoint_path}' not found.")
        print(f"[*] Please train the model first by running:\n    python3 train.py\n")
        sys.exit(1)

    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    chars = checkpoint["chars"]
    config = checkpoint["config"]
    is_qa = checkpoint.get("is_qa_model", True)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: "".join([itos[i] for i in l])

    model = MiniGPT(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, encode, decode, stoi, itos, device, is_qa


def generate_response(
    model,
    encode,
    decode,
    stoi,
    device,
    user_query: str,
    max_new_tokens: int = 200,
    temperature: float = 0.5,
    top_k: int = 40,
    is_qa: bool = True,
):
    """Formats the query as a conversation turn and extracts the assistant's answer."""
    if is_qa:
        prompt = f"User: {user_query.strip()}\nAssistant: "
    else:
        prompt = user_query

    encoded = encode(prompt)
    if not encoded:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        context = torch.tensor([encoded], dtype=torch.long, device=device)

    # Autoregressively generate tokens
    stop_indicators = ["<|end|>", "\nUser:", "User:"]
    generated_text = ""

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = context[:, -model.block_size :]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            context = torch.cat((context, idx_next), dim=1)
            next_char = decode([idx_next.item()])
            generated_text += next_char

            # Check stop indicators
            if any(stop_str in generated_text for stop_str in stop_indicators):
                break

    # Clean up generated text to extract pure assistant answer
    for stop_str in stop_indicators:
        if stop_str in generated_text:
            generated_text = generated_text.split(stop_str)[0]

    return generated_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Ask questions to the trained MiniGPT assistant.")
    parser.add_argument("--prompt", type=str, default=None, help="Question or prompt to ask.")
    parser.add_argument("--tokens", type=int, default=200, help="Maximum answer length in characters.")
    parser.add_argument("--temp", type=float, default=0.5, help="Temperature (default: 0.5 for accurate answers).")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling limit.")
    parser.add_argument("--checkpoint", type=str, default="mini_gpt.pt", help="Path to checkpoint.")
    parser.add_argument("--raw", action="store_true", help="Generate raw completion without QA prompt wrapper.")

    args = parser.parse_args()

    model, encode, decode, stoi, itos, device, is_qa = load_model(args.checkpoint)
    if args.raw:
        is_qa = False

    # Single Prompt Mode
    if args.prompt is not None:
        answer = generate_response(
            model=model,
            encode=encode,
            decode=decode,
            stoi=stoi,
            device=device,
            user_query=args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temp,
            top_k=args.top_k,
            is_qa=is_qa,
        )
        print(f"\nAssistant: {answer}\n")
        return

    # Interactive Mode
    print("\n" + "=" * 55)
    print("  MiniGPT Assistant (Trained Locally on Apple Silicon)  ")
    print("  Ask a question and press Enter.")
    print("  Type 'quit' or 'exit' to quit.")
    print("=" * 55 + "\n")

    while True:
        try:
            query = input("You > ")
            if not query.strip():
                continue
            if query.strip().lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            answer = generate_response(
                model=model,
                encode=encode,
                decode=decode,
                stoi=stoi,
                device=device,
                user_query=query,
                max_new_tokens=args.tokens,
                temperature=args.temp,
                top_k=args.top_k,
                is_qa=is_qa,
            )
            print(f"Assistant: {answer}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
