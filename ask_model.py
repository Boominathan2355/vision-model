import torch
import os
from tokenizer import CustomTokenizer
from builder import VelCoreModelBuilder

print("="*70)
print(" VELCORE MODEL - INTERACTIVE Q&A")
print("="*70)

# Load model and tokenizer
# Load model and tokenizer
print("\nLoading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = CustomTokenizer(vocab_path='custom_tokenizer.json')

# Load checkpoint first to get the correct vocabulary
checkpoint_path = 'VelCore-Pro.pt'
if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint {checkpoint_path} not found!")
    exit(1)

checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Reconstruct tokenizer from checkpoint
tokenizer = CustomTokenizer()
tokenizer.vocab = checkpoint['tokenizer_vocab']
tokenizer.max_length = checkpoint['tokenizer_max_length']
tokenizer.thinking_tokens = checkpoint['tokenizer_thinking_tokens']
tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
print(f"✓ Tokenizer loaded from checkpoint (Vocab size: {len(tokenizer.vocab)})")

# Build model with correct vocab size
vocab_size = len(tokenizer.vocab)
model = VelCoreModelBuilder.build_pro_model(vocab_size)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"Model loaded ({model.mode} mode)")
print(f"Device: {device}")

# Domain mapping
domains = {0: 'Mathematics', 1: 'Physics', 2: 'Chemistry', 
           3: 'Biology', 4: 'Computer Science', 5: 'Engineering'}

print("\n" + "="*70)
print("Interact with VELCORE (Type 'quit' or 'exit' to stop)")
print("="*70)

def generate_answer(model, tokenizer, prompt, max_new_tokens=100, device='cpu'):
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
    # Strip padding tokens (ID 4)
    pad_token_id = tokenizer.vocab.get('[PAD]', 4)
    if pad_token_id in input_ids:
        # Find first occurrence of padding and slice
        try:
            first_pad = input_ids.index(pad_token_id)
            input_ids = input_ids[:first_pad]
        except ValueError:
            pass
            
    # Add [CLS] at start if not present
    if tokenizer.vocab.get('[CLS]') not in input_ids:
        input_ids = [tokenizer.vocab.get('[CLS]', 6)] + input_ids
    
    curr_ids = torch.tensor([input_ids], device=device)
    
    # Generate tokens
    generated = []
    
    for _ in range(max_new_tokens):
        # Stop if we reach max sequence length
        if curr_ids.shape[1] >= 512:
            break
            
        # Prepare inputs
        dummy_image = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            outputs = model(curr_ids, dummy_image)
        
        # Get next token logits (last position)
        next_token_logits = outputs['reasoning_logits'][:, -1, :]
        
        # Greedy decode
        next_token_id = torch.argmax(next_token_logits, dim=-1).item()
        
        # Stop conditions
        if next_token_id == tokenizer.vocab.get('[SEP]', 7):
            break
        if next_token_id == tokenizer.vocab.get('[PAD]', 4):
            break
            
        generated.append(next_token_id)
        
        # Append to input for next step
        curr_ids = torch.cat([curr_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
            
    return tokenizer.decode(generated)

while True:
    try:
        q = input("\nEnter your question: ").strip()
        if q.lower() in ('quit', 'exit'):
            break
        if not q:
            continue
            
        print("Thinking...")
        
        # Step 1: Predict Domain
        # Format: "Question: {q}" to get domain classification
        input_text = f"Question: {q}"
        text_tokens = tokenizer.encode(input_text)
        text_tensor = torch.tensor([text_tokens]).to(device)
        image_tensor = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            outputs = model(text_tensor, image_tensor)
        
        logits = outputs['logits'].cpu()
        pred_class = torch.argmax(logits, dim=1).item()
        pred_domain = domains.get(pred_class, 'General')
        confidence = torch.softmax(logits, dim=1).max().item()
        
        print(f"  Predicted Domain: {pred_domain} ({confidence:.1%})")
        
        # Step 2: Generate Answer
        # Format: "Domain: {domain} Question: {q} Answer:"
        prompt = f"Domain: {pred_domain} Question: {q} Answer:"
        
        answer = generate_answer(model, tokenizer, prompt, device=device)
        
        # Clean answer
        answer = answer.replace('[PAD]', '').replace('[UNK]', '').strip()
        
        print(f"  Generated Answer: {answer}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        break
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

print("\nGoodbye!")
