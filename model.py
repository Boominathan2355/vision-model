import torch
from tokenizer import CustomTokenizer
from architecture import ImagePreprocessor, VelCoreModel
from builder import VelCoreModelBuilder, save_model

# ==================== Example Usage ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Building VelCore Models from Scratch with Custom Tokenizer")
    print("=" * 60)
    
    # Step 1: Create and build custom tokenizer
    print("\n[1/5] Creating custom tokenizer...")
    custom_tokenizer = CustomTokenizer(max_length=512)
    
    # Sample texts to build vocabulary
    sample_texts = [
        "What is in this image? Let me analyze it carefully.",
        "This appears to be a visual representation of data.",
        "The image shows multiple objects and patterns.",
        "Image analysis requires careful examination of details.",
        "Visual understanding is crucial for reasoning tasks.",
        "Let me think about what I see in the image.",
        "The model should reason about visual content.",
        "Understanding multimodal data is complex.",
        "Text and images together provide richer context.",
        "Reasoning through visual information helps solve problems."
    ]
    
    print(f"✓ Building vocabulary from {len(sample_texts)} sample texts...")
    custom_tokenizer.build_vocab(sample_texts, min_freq=1)
    print(f"✓ Custom tokenizer created")
    print(f"  - Vocab size: {len(custom_tokenizer.vocab)}")
    print(f"  - Max length: {custom_tokenizer.max_length}")
    print(f"  - Sample tokens: {list(custom_tokenizer.vocab.keys())[:10]}")
    
    # Step 2: Build fresh Pro model with custom tokenizer
    print("\n[2/5] Building VelCore Pro model from scratch...")
    pro_model = VelCoreModel(mode='pro', tokenizer_vocab_size=len(custom_tokenizer.vocab))
    pro_model.tokenizer = custom_tokenizer
    pro_model.apply(VelCoreModelBuilder._init_weights)
    print(f"✓ Pro model built successfully")
    print(f"  - Hidden size: 768")
    print(f"  - Num heads: 12")
    print(f"  - Fusion layers: 6")
    print(f"  - Reasoning steps: 4")
    print(f"  - Vocab size: {len(pro_model.tokenizer.vocab)}")
    
    # Step 3: Build fresh Lite model with custom tokenizer
    print("\n[3/5] Building VelCore Lite model from scratch...")
    lite_model = VelCoreModel(mode='lite', tokenizer_vocab_size=len(custom_tokenizer.vocab))
    lite_model.tokenizer = custom_tokenizer
    lite_model.apply(VelCoreModelBuilder._init_weights)
    print(f"✓ Lite model built successfully")
    print(f"  - Hidden size: 384")
    print(f"  - Num heads: 8")
    print(f"  - Fusion layers: 3")
    print(f"  - Reasoning steps: 2")
    print(f"  - Vocab size: {len(lite_model.tokenizer.vocab)}")
    
    # Step 4: Prepare example inputs
    print("\n[4/5] Preparing example inputs...")
    tokenizer = pro_model.tokenizer
    
    # Example text using custom tokenizer
    text = "What is in this image? Let me think about the visual content carefully."
    text_tokens = tokenizer.encode(text)
    text_tensor = torch.tensor([text_tokens])
    
    # Dummy image tensor
    image_tensor = torch.randn(1, 3, 224, 224)
    print(f"✓ Text input: '{text}'")
    print(f"✓ Text tokens (first 20): {text_tokens[:20]}")
    print(f"✓ Text input shape: {text_tensor.shape}")
    print(f"✓ Image input shape: {image_tensor.shape}")
    
    # Step 5: Run inference on fresh Pro model
    print("\n[5/5] Running inference on fresh Pro model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    pro_model = pro_model.to(device)
    text_tensor = text_tensor.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = pro_model(text_tensor, image_tensor)
        reasoning_text = pro_model.think(text_tensor, image_tensor)
    
    print(f"✓ Inference completed successfully")
    print(f"\nOutput shapes:")
    print(f"  - Classification logits: {outputs['logits'].shape}")
    print(f"  - Reasoning features: {outputs['features']['reasoned_features'].shape}")
    print(f"  - Fused features: {outputs['features']['fused_features'].shape}")
    print(f"  - Text features: {outputs['features']['text_features'].shape}")
    print(f"  - Visual features: {outputs['features']['visual_features'].shape}")
    if 'reconstructed_image' in outputs:
        print(f"  - Reconstructed image: {outputs['reconstructed_image'].shape}")
    
    print(f"\nReasoning output (first 100 chars): {reasoning_text[:100]}...")
    print("\n" + "=" * 60)
    print("✓ Models built and tested successfully from scratch!")
    print("✓ Custom tokenizer initialized and integrated!")
    print("=" * 60)