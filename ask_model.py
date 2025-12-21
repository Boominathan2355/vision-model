import os
import torch
import torch.nn.functional as F
import argparse
from typing import List

from tokenizer import CustomTokenizer
from builder import VelCoreModelBuilder, load_model

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CHECKPOINT = "pro-model.pt"
TOKENIZER_PATH = "custom_tokenizer.json"
MAX_NEW_TOKENS = None
TEMPERATURE = 0.6
TOP_K = 50
REPETITION_PENALTY = 1.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# UTILS
# ============================================================
#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
def format_output(response: str) -> str:
    """Clean up and format the model response"""
    return response.strip()

@torch.no_grad()
def generate_response(
    model, 
    tokenizer, 
    prompt: str, 
    max_tokens: int = MAX_NEW_TOKENS, 
    temperature: float = TEMPERATURE, 
    top_k: int = TOP_K
):
    model.eval()
    
    # 1. Encode Prompt
    if "User:" not in prompt:
        full_prompt = f"User: {prompt}\nAssistant:"
    else:
        full_prompt = prompt
        
    input_ids = tokenizer.encode(full_prompt)
    
    # Remove PAD tokens
    if hasattr(tokenizer, 'vocab') and '[PAD]' in tokenizer.vocab:
        pad_id = tokenizer.vocab['[PAD]']
        input_ids = [t for t in input_ids if t != pad_id]

    # Remove trailing SEP token if present
    if hasattr(tokenizer, 'vocab') and '[SEP]' in tokenizer.vocab:
        sep_id = tokenizer.vocab['[SEP]']
        if input_ids and input_ids[-1] == sep_id:
            input_ids = input_ids[:-1]

    # Convert to tensor
    curr_ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    dummy_image = torch.zeros(1, 3, 224, 224, device=DEVICE)

    print("Assistant: ", end="", flush=True)
    generated_ids = []
    
    # State for thinking/reasoning
    is_thinking = False
    is_reasoning = False
    
    # Get special token IDs
    think_start_id = tokenizer.vocab.get('[THINK_START]', -1)
    think_end_id = tokenizer.vocab.get('[THINK_END]', -1)
    reason_start_id = tokenizer.vocab.get('[REASON_START]', -1)
    reason_end_id = tokenizer.vocab.get('[REASON_END]', -1)
    sep_id = tokenizer.vocab.get('[SEP]', -1)
    
    # Use a large number if max_tokens is None
    num_to_generate = max_tokens if max_tokens is not None else 2048

    for i in range(num_to_generate):
        outputs = model(text_input=curr_ids, image_input=dummy_image)
        logits = outputs["reasoning_logits"][:, -1, :]
        
        # Penalize UNK
        if hasattr(tokenizer, 'vocab') and '[UNK]' in tokenizer.vocab:
            logits[:, tokenizer.vocab['[UNK]']] = -float('inf')

        # Apply repetition penalty
        for prev_token_id in set(generated_ids):
            if logits[0, prev_token_id] > 0:
                logits[0, prev_token_id] /= REPETITION_PENALTY
            else:
                logits[0, prev_token_id] *= REPETITION_PENALTY

        logits = logits / (temperature if temperature > 0 else 1.0)
        
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        # Handle Thinking / Reasoning Tokens
        if next_token_id == think_start_id:
            is_thinking = True
            print("\n[THINKING] ", end="", flush=True)
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token_id]], device=DEVICE)], dim=1)
            continue
        elif next_token_id == think_end_id:
            is_thinking = False
            print("\n", end="", flush=True)
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token_id]], device=DEVICE)], dim=1)
            continue
        elif next_token_id == reason_start_id:
            is_reasoning = True
            print("[REASONING] ", end="", flush=True)
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token_id]], device=DEVICE)], dim=1)
            continue
        elif next_token_id == reason_end_id:
            is_reasoning = False
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_token_id]], device=DEVICE)], dim=1)
            continue
            
        # Stop condition
        if next_token_id == sep_id:
            break
            
        # Decode and Print
        token_text = tokenizer.decode([next_token_id])
        if token_text:
            # Punctuation spacing fix
            if token_text in ".,!?;:" and len(generated_ids) > 0:
                print("\b" + token_text + " ", end="", flush=True)
            else:
                print(token_text + " ", end="", flush=True)
                
            # Stop if the model starts generating dialogue markers
            # More robust check for multi-token labels
            combined_recent_text = tokenizer.decode(generated_ids[-5:])
            if "User:" in combined_recent_text or "Assistant:" in combined_recent_text:
                # Remove the label tokens from output before finishing
                print("\b" * (len(token_text) + 1), end="", flush=True)
                break
        
        generated_ids.append(next_token_id)
        curr_ids = torch.cat([curr_ids, torch.tensor([[next_token_id]], device=DEVICE)], dim=1)
        
    print("\n")
    return tokenizer.decode(generated_ids)

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VelCore Model Interactive Chat")
    parser.add_argument("--model", type=str, default=DEFAULT_CHECKPOINT, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_PATH, help="Path to tokenizer json")
    parser.add_argument("--lite", action="store_true", help="Force Lite mode if not auto-detected")
    args = parser.parse_args()

    print("=" * 70)
    print(" VELCORE INTERACTIVE CHAT")
    print("=" * 70)
    print(f"Loading from: {args.model}")
    print(f"Device: {DEVICE}")

    # 1. Load Tokenizer
    if os.path.exists(args.tokenizer):
        tokenizer = CustomTokenizer(vocab_path=args.tokenizer)
        print("✓ Tokenizer loaded")
    else:
        print(f"⚠ Tokenizer file {args.tokenizer} not found! Please run training first.")
        return

    # 2. Load Model
    if not os.path.exists(args.model):
        print(f"⚠ Model checkpoint {args.model} not found! Please run training first.")
        return
        
    try:
        model, config = load_model(args.model, tokenizer, mode='lite' if args.lite else None)
        model = model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Interactive Loop
    print("\nModel ready! Type 'exit' or 'quit' to stop.")
    print("-" * 50)

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break
            
            if not user_input:
                continue

            generate_response(model, tokenizer, user_input)
            
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()

    print("Goodbye!")

if __name__ == "__main__":
    main()
