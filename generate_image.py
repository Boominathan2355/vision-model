"""
VelCore Image Generator - Generate images from text prompts
"""
import torch
from PIL import Image
import numpy as np
from tokenizer import CustomTokenizer
from builder import load_model
import os


def generate_image_from_text(model, tokenizer, prompt: str, device: str = 'cpu') -> np.ndarray:
    """
    Generate an image from a text prompt
    
    Args:
        model: VelCore PRO model
        tokenizer: CustomTokenizer instance
        prompt: Text description for image generation
        device: 'cuda' or 'cpu'
        
    Returns:
        image: numpy array [224, 224, 3] in uint8 format
    """
    model.eval()
    
    # Tokenize the prompt
    tokens = tokenizer.encode(prompt)
    text_tensor = torch.tensor([tokens]).to(device)
    
    # Generate image
    generated = model.generate_image(text_tensor)
    
    # Convert from [-1, 1] to [0, 255]
    image = generated[0].cpu().numpy()  # [3, 224, 224]
    image = (image + 1) / 2  # [-1, 1] -> [0, 1]
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    image = image.transpose(1, 2, 0)  # [224, 224, 3]
    
    return image


def save_image(image: np.ndarray, path: str):
    """Save numpy image to file"""
    pil_image = Image.fromarray(image)
    pil_image.save(path)
    print(f"✓ Image saved to: {path}")


def main():
    print("=" * 70)
    print(" VELCORE IMAGE GENERATOR")
    print(" Generate images from text prompts")
    print("=" * 70)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'VelCore-Pro.pt'
    tokenizer_path = 'custom_tokenizer.json'
    output_dir = 'generated_images'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n⚠ Model not found: {model_path}")
        print("Please train the VelCore PRO model first using train_final.py")
        return
    
    # Load tokenizer and model
    print(f"\n[1] Loading tokenizer from {tokenizer_path}...")
    tokenizer = CustomTokenizer(vocab_path=tokenizer_path)
    
    print(f"[2] Loading model from {model_path}...")
    model = load_model(model_path, tokenizer)
    model = model.to(device)
    model.eval()
    
    print(f"[3] Device: {device}")
    print(f"    Model mode: {model.mode}")
    
    if model.mode != 'pro':
        print("\n⚠ Image generation requires VelCore PRO model!")
        return
    
    # Sample prompts
    prompts = [
        "A beautiful sunset over the ocean",
        "A red apple on a wooden table",
        "A geometric triangle shape",
        "A mathematical equation on a blackboard",
        "A colorful abstract pattern"
    ]
    
    print(f"\n[4] Generating images from {len(prompts)} prompts...")
    print("-" * 70)
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: \"{prompt}\"")
        
        # Generate image
        image = generate_image_from_text(model, tokenizer, prompt, device)
        
        # Save image
        filename = f"generated_{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        save_image(image, filepath)
    
    print("\n" + "=" * 70)
    print(f" ✓ Generated {len(prompts)} images in '{output_dir}/' directory")
    print("=" * 70)
    
    print("\n[NOTE] The generated images will appear random/abstract.")
    print("For meaningful images, the model needs to be trained on")
    print("text-image pair datasets like LAION or COCO Captions.")


if __name__ == "__main__":
    main()
