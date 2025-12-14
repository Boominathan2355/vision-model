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

DEFAULT_CHECKPOINT = "Pro-model.pt"
TOKENIZER_PATH = "custom_tokenizer.json"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_K = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# UTILS
# ============================================================

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

    # Remove trailing SEP token if present (so model continues generating)
    if hasattr(tokenizer, 'vocab') and '[SEP]' in tokenizer.vocab:
        sep_id = tokenizer.vocab['[SEP]']
        if input_ids and input_ids[-1] == sep_id:
            input_ids = input_ids[:-1]

    # Convert to tensor
    curr_ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    
    # Dummy image for the multimodal architecture (since we are doing text-only interaction)
    dummy_image = torch.zeros(1, 3, 224, 224, device=DEVICE)

    print("Assistant: ", end="", flush=True)
    generated_ids = []
    
    for _ in range(max_tokens):
        # 2. Forward Pass
        outputs = model(text_input=curr_ids, image_input=dummy_image)
        
        # 3. Get Logits
        logits = outputs["reasoning_logits"][:, -1, :]
        
        # 4. Filter / Sample
        if hasattr(tokenizer, 'vocab') and '[UNK]' in tokenizer.vocab:
            logits[:, tokenizer.vocab['[UNK]']] = -float('inf')

        logits = logits / temperature
        
        # Top-K
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        # Debug print
        # print(f"[{next_token_id}:{tokenizer.decode([next_token_id])}]", end=" ", flush=True)

        # 5. Stop Conditions
        if hasattr(tokenizer, 'vocab'):
            if next_token_id == tokenizer.vocab.get('[SEP]', -1):
                break
            if next_token_id == tokenizer.vocab.get('[PAD]', -1):
                break
        
        # 6. Append
        generated_ids.append(next_token_id)
        
        # Efficiently append to input for next step
        curr_ids = torch.cat([curr_ids, torch.tensor([[next_token_id]], device=DEVICE)], dim=1)
        
        # Stop if we hit max length
        if len(generated_ids) >= max_tokens:
            break
            
    # Final decode
    full_response = tokenizer.decode(generated_ids)
    print(full_response)
    print("\n")
    return full_response

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
