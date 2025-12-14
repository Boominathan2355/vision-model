import torch
from tokenizer import CustomTokenizer
from architecture import ImagePreprocessor
from builder import VelCoreModelBuilder, save_model

# ==================== Example Usage ====================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" VELCORE MODEL - Building from Scratch with Custom Tokenizer")
    print("=" * 70)
    
    # Step 1: Create and build tokenizer
    print("\n[STEP 1] Creating Custom Tokenizer...")
    tokenizer = CustomTokenizer(max_length=512)
    
    # Sample texts for vocabulary building
    sample_texts = [
        "What is in this image?",
        "Can you analyze this visual content?",
        "Describe what you see here.",
        "Tell me about the image content.",
        "What objects are visible in this picture?",
        "Analyze the visual information provided.",
        "Let me think about this image carefully.",
        "I need to reason through this step by step.",
        "This requires careful analysis and thinking.",
        "Understanding images requires deep reasoning.",
        "The image shows various objects and patterns.",
        "Visual understanding is crucial for analysis.",
        "Complex reasoning about images is challenging.",
        "Multimodal learning combines text and vision.",
        "Deep analysis of visual data is important.",
    ]
    
    tokenizer.build_vocab(sample_texts, min_freq=1)
    print(f"✓ Tokenizer created")
    print(f"  - Vocab size: {len(tokenizer.vocab)}")
    print(f"  - Max sequence length: {tokenizer.max_length}")
    print(f"  - Special tokens: {len(tokenizer.thinking_tokens)}")
    
    # Step 2: Build Pro model
    print("\n[STEP 2] Building VelCore Pro Model from scratch...")
    pro_model = VelCoreModelBuilder.build_pro_model(len(tokenizer.vocab))
    pro_model.eval()
    
    # Step 3: Build Lite model
    print("\n[STEP 3] Building VelCore Lite Model from scratch...")
    lite_model = VelCoreModelBuilder.build_lite_model(len(tokenizer.vocab))
    lite_model.eval()
    
    # Step 4: Prepare input
    print("\n[STEP 4] Preparing example inputs...")
    text = "What is in this image? Let me analyze it carefully."
    text_tokens = tokenizer.encode(text)
    text_tensor = torch.tensor([text_tokens])
    
    # Create dummy image
    image_tensor = torch.randn(1, 3, 224, 224)
    print(f"✓ Text tokenized: {len(text_tokens)} tokens")
    print(f"✓ Text tensor shape: {text_tensor.shape}")
    print(f"✓ Image tensor shape: {image_tensor.shape}")
    
    # Step 5: Run inference
    print("\n[STEP 5] Running inference with Pro model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    pro_model = pro_model.to(device)
    text_tensor = text_tensor.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = pro_model(text_tensor, image_tensor)
    
    print(f"✓ Inference completed successfully!")
    print(f"  - Classification logits shape: {outputs['logits'].shape}")
    print(f"  - Reasoning logits shape: {outputs['reasoning_logits'].shape}")
    print(f"  - Text features shape: {outputs['features']['text_features'].shape}")
    print(f"  - Visual features shape: {outputs['features']['visual_features'].shape}")
    print(f"  - Fused features shape: {outputs['features']['fused_features'].shape}")
    print(f"  - Reasoned features shape: {outputs['features']['reasoned_features'].shape}")
    if 'reconstructed_image' in outputs:
        print(f"  - Reconstructed image shape: {outputs['reconstructed_image'].shape}")
    
    # Step 6: Save models and tokenizer
    print("\n[STEP 6] Saving models and tokenizer...")
    save_model(pro_model, tokenizer, 'Pro-model.pt')
    save_model(lite_model, tokenizer, 'Lite-model.pt')
    tokenizer.save('custom_tokenizer.json')
    
    # Step 7: Test Lite model
    print("\n[STEP 7] Testing Lite model...")
    lite_model = lite_model.to(device)
    with torch.no_grad():
        lite_outputs = lite_model(text_tensor, image_tensor)
    
    print(f"✓ Lite model inference completed!")
    print(f"  - Classification logits shape: {lite_outputs['logits'].shape}")
    print(f"  - Total parameters: {sum(p.numel() for p in lite_model.parameters()):,}")
    
    print("\n" + "=" * 70)
    print(" ✓ SUCCESS: VelCore models built and tested from scratch!")
    print("=" * 70 + "\n")
