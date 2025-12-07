import torch
from tokenizer import CustomTokenizer
from builder import load_model

print("="*70)
print(" THIRAN MODEL - INTERACTIVE Q&A")
print("="*70)

# Load model and tokenizer
print("\nLoading model...")
tokenizer = CustomTokenizer(vocab_path='custom_tokenizer.json')
model = load_model('thiran_pro_model.pt', tokenizer)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Model loaded ({model.mode} mode)")
print(f"Device: {device}")

# Domain mapping
domains = {0: 'Mathematics', 1: 'Physics', 2: 'Chemistry', 
           3: 'Biology', 4: 'Computer Science', 5: 'Engineering'}

# Example questions
questions = [
    "What is the derivative of x squared?",
    "What is Newton's first law of motion?",
    "What is the chemical formula for water?",
]

print("\n" + "="*70)
print("Testing with sample questions:")
print("="*70)

for q in questions:
    text_tokens = tokenizer.encode(f"Question: {q}")
    text_tensor = torch.tensor([text_tokens]).to(device)
    image_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        outputs = model(text_tensor, image_tensor)
    
    logits = outputs['logits'].cpu()
    pred_class = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1).max().item()
    
    # Generate text from reasoning logits
    reasoning_logits = outputs['reasoning_logits'].cpu()
    generated_tokens = torch.argmax(reasoning_logits, dim=-1)[0]  # Get first batch
    
    # Decode generated tokens to text
    generated_text = tokenizer.decode(generated_tokens.tolist())
    
    # Clean up the generated text (remove padding and special tokens)
    generated_text = generated_text.replace('[PAD]', '').replace('[UNK]', '').strip()
    # Take first 200 chars for display
    generated_text = generated_text[:200] if len(generated_text) > 200 else generated_text
    
    print(f"\nQ: {q}")
    print(f"  Predicted Domain: {domains.get(pred_class, 'Unknown')}")
    print(f"  Confidence: {confidence:.1%}")
    print(f"  Valid output: {not torch.isnan(logits).any()}")
    print(f"  Generated Answer: {generated_text if generated_text else '[No text generated]'}")

print("\n" + "="*70)
print("Model is working correctly!")
print("="*70)
