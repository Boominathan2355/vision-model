# VelCore - Multimodal Vision-Language Model

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1.0-brightgreen.svg)](https://pytorch.org/)
[![License: Private](https://img.shields.io/badge/License-Private-yellow.svg)](LICENSE)

</div>

> A multimodal AI model combining vision and language understanding with advanced reasoning and image generation capabilities. VelCore offers two variants: **Pro** (~254M parameters) and **Lite** (~24M parameters).

## ✨ Features

| Feature | Pro | Lite | Description |
|---------|-----|------|-------------|
| **Domain Classification** | ✅ | ✅ | Classifies into Math, Physics, Chemistry, Biology, CS, Engineering |
| **Text Reasoning** | ✅ | ✅ | Generates reasoning/answer text from questions |
| **Image Understanding** | ✅ | ✅ | Processes and understands 224×224 images |
| **Multimodal Fusion** | ✅ | ✅ | Combines text + image for reasoning |
| **Image Reconstruction** | ✅ | ❌ | Reconstructs input images |
| **Image Generation** | ✅ | ❌ | Generates images from text prompts |

## 📋 Requirements

- Python 3.10+
- PyTorch 2.1.0+
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for Pro model)

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Boominathan2355/vision-model.git
cd vision-model
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from tokenizer import CustomTokenizer
from builder import VelCoreModelBuilder

# Build model
tokenizer = CustomTokenizer(max_length=512)
tokenizer.build_vocab(["Sample text"], min_freq=1)
model = VelCoreModelBuilder.build_pro_model(len(tokenizer.vocab))

# Inference
text_tokens = tokenizer.encode("What is 2+2?")
text_tensor = torch.tensor([text_tokens])
image = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    outputs = model(text_tensor, image)
```

### Image Generation (Pro Only)

```python
# Generate image from text
model = VelCoreModelBuilder.build_pro_model(vocab_size)
text_tokens = tokenizer.encode("A beautiful sunset")
text_tensor = torch.tensor([text_tokens])

generated_image = model.generate_image(text_tensor)  # [1, 3, 224, 224]
```

## 📊 Model Architecture

| Spec | VelCore-Pro | VelCore-Lite |
|------|-------------|--------------|
| Hidden Size | 768 | 384 |
| Attention Heads | 12 | 8 |
| Fusion Layers | 6 | 3 |
| Reasoning Steps | 4 | 2 |
| Parameters | ~254M | ~24M |

## 📁 Project Structure

```
VelCore/
├── architecture.py       # Model architecture (VelCoreModel, ImageGenerator)
├── builder.py            # Model builder and save/load utilities
├── tokenizer.py          # Custom tokenizer implementation
├── train_final.py        # Training on Turing-Open-Reasoning
├── train_geometry.py     # Training on Geometry3K (real images)
├── train_gsm8k.py        # Training on GSM8K math problems
├── train_socratic.py     # Training on GSM8K Socratic
├── ask_model.py          # Interactive Q&A script
├── generate_image.py     # Text-to-image generation script
├── evaluate_model.py     # Model evaluation script
└── main.py               # Example usage script
```

## 🎯 Training Scripts

| Script | Dataset | Samples | Description |
|--------|---------|---------|-------------|
| `train_final.py` | Turing-Open-Reasoning | 1000 | Multi-domain reasoning |
| `train_geometry.py` | Geometry3K | 2000 | Real geometry diagram images |
| `train_gsm8k.py` | GSM8K | 5000 | Math word problems |
| `train_socratic.py` | GSM8K Socratic | 5000 | Socratic method reasoning |

```bash
python train_final.py      # Train on reasoning dataset
python train_geometry.py   # Train with real images
```

## 💡 Example Scripts

```bash
python ask_model.py        # Interactive Q&A
python generate_image.py   # Generate images from text
python evaluate_model.py   # Evaluate model performance
```

## 📦 Model Outputs

```python
{
    'logits': Tensor,              # Classification logits [batch, 10]
    'reasoning_logits': Tensor,    # Text generation logits [batch, seq, vocab]
    'features': {
        'text_features': Tensor,
        'visual_features': Tensor,
        'fused_features': Tensor,
        'reasoned_features': Tensor
    },
    'reconstructed_image': Tensor  # Pro only [batch, 150528]
}
```

## 🔧 Troubleshooting

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
CUDA_VISIBLE_DEVICES="" python train_final.py

# Reduce memory
# Edit config: batch_size = 2
```

## 📄 License

This project is licensed under the Private License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📚 References

- Vaswani et al. (2017) - Attention Is All You Need
- Dosovitskiy et al. (2020) - An Image is Worth 16x16 Words
- Li et al. (2022) - BLIP: Bootstrapping Language-Image Pre-training

---

Made with ❤️ by the VelCore team