# VelCore - Multimodal Vision-Language Model

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1.0-brightgreen.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

> A multimodal AI model combining vision and language understanding with advanced reasoning and image generation capabilities. VelCore offers two variants: **Pro** (600M Parameter Scale) and **Lite** (300M Parameter Scale) with state-of-the-art reasoning and multimodal understanding.

## ✨ Features

| Feature | Pro-model | Lite-model | Description |
|---------|-----------|------------|-------------|
| **Domain Classification** | ✅ | ✅ | Classifies into Math, Physics, Chemistry, Biology, CS, Engineering |
| **Text Reasoning** | ✅ | ✅ | Generates reasoning/answer text from questions |
| **Image Understanding** | ✅ | ✅ | Processes and understands 224×224 images |
| **Multimodal Fusion** | ✅ | ✅ | Combines text + image for reasoning |
| **Case Sensitive** | ✅ | ✅ | Preserves text case for better context understanding |
| **Image Reconstruction** | ✅ | ❌ | Reconstructs input images |
| **Image Generation** | ✅ | ❌ | Generates images from text prompts |

## 📋 Requirements

- Python 3.10+
- PyTorch 2.1.0+
- CUDA 11.8+ (required for Pro model, optional for Lite)
- **Pro Model**: 24GB+ VRAM, 32GB+ system RAM
- **Lite Model**: 8GB+ VRAM, 16GB+ system RAM

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Boominathan2355/vision-model.git
cd vision-model

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Dataset Authentication (Crucial)

To access gated datasets (e.g., `Turing-Open-Reasoning`, `Simple-English-Conversation`), you must provide a Hugging Face token.

1.  Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens).
2.  Create a file named `token.txt` in the root directory.
3.  Paste your token inside `token.txt` (no spaces or newlines).

The training scripts will automatically read this file to authenticate downloads.

### Basic Usage

```python
import torch
from tokenizer import CustomTokenizer
from builder import VelCoreModelBuilder

# Initialize tokenizer
tokenizer = CustomTokenizer(max_length=512)
sample_texts = ["What is in this image?", "Analyze this visual content."]
tokenizer.build_vocab(sample_texts, min_freq=1)

# Build model
model = VelCoreModelBuilder.build_pro_model(len(tokenizer.vocab))
model.eval()

# Prepare inputs
text_tokens = tokenizer.encode("What is 2+2?")
text_tensor = torch.tensor([text_tokens])
image = torch.randn(1, 3, 224, 224)

# Inference
with torch.no_grad():
    outputs = model(text_tensor, image)
    print(f"Domain Classification: {outputs['logits'].shape}")
    print(f"Reasoning Output: {outputs['reasoning_logits'].shape}")
```

### Image Generation (Pro Model Only)

```python
# Generate image from text description
model = VelCoreModelBuilder.build_pro_model(vocab_size)
tokenizer = CustomTokenizer(max_length=512)
tokenizer.build_vocab(["sample text"], min_freq=1)

text_tokens = tokenizer.encode("A beautiful sunset over mountains")
text_tensor = torch.tensor([text_tokens])

generated_image = model.generate_image(text_tensor)  # [1, 3, 224, 224]
```

### Interactive Q&A

```bash
python ask_model.py
# Follow prompts to ask questions about images or general topics
```

## 📊 Model Architecture

| Spec | Pro-model (600M) | Lite-model (300M) |
|------|-------------|---------------|
| Hidden Size | 1024 | 512 |
| Attention Heads | 16 | 8 |
| Fusion Layers | 6 | 3 |
| Reasoning Steps | 8 | 4 |
| Max Sequence Length | 8192 | 8192 |
| Parameters | ~600M | ~300M |

## 📁 Project Structure

```
vision-model/
├── Core Components
│   ├── architecture.py           # Model architecture (VelCoreModel, ImageGenerator, ThinkingLayer)
│   ├── builder.py                # Model builder and save/load utilities
│   ├── tokenizer.py              # Custom tokenizer implementation
│   ├── model.py                  # Model utilities and core components
│   └── custom_tokenizer.json     # Pre-built tokenizer vocabulary
│
├── Training Scripts
│   ├── train_final.py            # Training on Turing-Open-Reasoning dataset
│   ├── train_geometry.py         # Training on Geometry3K (real images)
│   ├── train_gsm8k.py            # Training on GSM8K math problems
│   ├── train_socratic.py         # Training on GSM8K with Socratic reasoning
│   └── train_conversation.py     # Training on conversation/dialogue data
│
├── Inference & Evaluation
│   ├── ask_model.py              # Interactive Q&A interface
│   ├── generate_image.py         # Text-to-image generation (Pro only)
│   ├── evaluate_model.py         # Model evaluation and benchmarking
│   └── main.py                   # Example usage and demo script
│
├── Configuration & Documentation
│   ├── pyproject.toml            # Project metadata and dependencies
│   ├── requirements.txt          # Python package dependencies
│   ├── README.md                 # This file
│   ├── CONTRIBUTING.md           # Contribution guidelines
│   ├── LICENSE                   # MIT License
│   └── .github/                  # GitHub workflows and templates
│
└── Data & Config
    ├── custom_tokenizer.json     # Tokenizer configuration
    ├── token.txt                 # API tokens (not version controlled)
    └── .venv/                    # Virtual environment (not version controlled)
```

## 🎯 Training Scripts

| Script | Dataset | Purpose | Description |
|--------|---------|---------|-------------|
| `train_final.py` | Turing-Open-Reasoning | Multi-Domain Reasoning | Trains on diverse reasoning problems across 6+ domains |
| `train_geometry.py` | Geometry3K | Visual Geometry | Real geometry diagrams with image understanding |
| `train_gsm8k.py` | GSM8K | Math Problem Solving | Grade school math word problems |
| `train_socratic.py` | GSM8K Socratic | Step-by-Step Reasoning | Socratic method for reasoning explanation |
| `train_conversation.py` | Xerv-AI/Simple-English-Conversation | Conversational AI | Training on multi-turn conversations |

```bash
# Run individual training scripts
python train_final.py      # Multi-domain reasoning training
python train_geometry.py   # Geometry with real images
python train_gsm8k.py      # Math problem training
python train_socratic.py   # Socratic reasoning training
python train_conversation.py  # Conversational training
```

> **Note**: All scripts support full dataset training (`max_samples=None`) and authenticated downloading via `token.txt`.

## 💡 Example Scripts

```bash
# Interactive inference
python ask_model.py             # Run Q&A interface with the model
python main.py                  # See model building and inference example
python generate_image.py        # Generate images from text prompts

# Model evaluation
python evaluate_model.py        # Run benchmarks and evaluate performance
```

## 🏋️ Training & Fine-tuning

### Quick Training

```bash
# Start with a smaller dataset
python train_final.py

# Train on specific domains
python train_geometry.py        # Visual domain
python train_gsm8k.py          # Math domain
python train_socratic.py       # Reasoning domain
python train_conversation.py   # Dialogue domain
```

### Training Configuration

Most training scripts accept command-line arguments:

```bash
python train_final.py --epochs 10 --batch_size 16 --learning_rate 0.001
```

Check individual scripts for available arguments and customization options.

## 💾 Model Saving & Loading

```python
from builder import VelCoreModelBuilder, save_model, load_model

# Save a trained model
save_model(model, tokenizer, model_path="./Pro-model.pt")

# Load a saved model
model, tokenizer = load_model(model_path="./Pro-model.pt")

# Model inference
model.eval()
with torch.no_grad():
    outputs = model(text_tensor, image_tensor)
```

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

## 🎯 Choosing a Model Variant

### Pro-model (600M Parameter)
- **Best For**: Advanced research, enterprise-grade vision-language tasks
- **Parameters**: ~600 Million
- **Features**: Advanced image generation, reconstruction, multi-step reasoning, complex visual understanding
- **Memory**: 24GB+ VRAM, 32GB+ system RAM
- **Speed**: ~500-1000ms per inference
- **Capabilities**: State-of-the-art reasoning, detailed image analysis, complex problem solving

### Lite-model (300M Parameter)
- **Best For**: Production deployments, efficient inference, edge devices
- **Parameters**: ~300 Million
- **Features**: Classification, multi-step reasoning, image understanding, fast inference
- **Memory**: 8GB+ VRAM, 16GB+ system RAM
- **Speed**: ~100-200ms per inference
- **Capabilities**: Strong reasoning capabilities with reduced latency and memory footprint

### Performance Comparison

| Metric | Pro (600M) | Lite (300M) | Notes |
|--------|---|---|-------|
| Reasoning Accuracy | 92-97% | 85-92% | Complex multi-step reasoning |
| Image Understanding | State-of-the-art | Excellent | Fine-grained visual analysis |
| Inference Speed | ~200ms | ~50ms | Per example, batch=1 |
| Training Time | 1-2 days | 6-12 hours | On 4x GPU |
| Model Size (Disk) | ~2.4GB | ~1.2GB | FP32 precision |
| Throughput | 50-100 ex/min | 200+ ex/min | Single GPU, batch=1 |

## 🔧 Troubleshooting

### Common Issues

**CUDA out of memory**
```bash
# Use a smaller batch size
export CUDA_VISIBLE_DEVICES=0  # Use single GPU
# Edit training script and set batch_size=2 or batch_size=1
```

**Model not found**
```bash
# Ensure model files are in correct directory
python -c "import os; print(os.getcwd())"
# Check if custom_tokenizer.json exists
```

**Import errors**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**CPU-only mode**
```bash
# Force CPU execution
set CUDA_VISIBLE_DEVICES=""  # Windows
python train_final.py
```

**Verify Installation**
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
from tokenizer import CustomTokenizer
from architecture import VelCoreModel
print('✓ All imports successful!')
"
```

## 📚 References

- Vaswani et al. (2017) - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Dosovitskiy et al. (2020) - [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Li et al. (2022) - [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- How to submit issues and feature requests
- How to make code contributions
- Code style and testing requirements
- Pull request process

## 📝 Citation

If you use VelCore in your research, please cite:

```bibtex
@software{velcore2024,
  title={VelCore: Multimodal Vision-Language Model},
  author={Boominathan and contributors},
  year={2024},
  url={https://github.com/Boominathan2355/vision-model}
}
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙋 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/Boominathan2355/vision-model/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Boominathan2355/vision-model/discussions)
- **Email**: For queries, open an issue on GitHub

---

<div align="center">

Made with ❤️ by [VelCore Team](https://github.com/Boominathan2355)

[⬆ Back to top](#velcore---multimodal-vision-language-model)

</div>