# Thiran - Multimodal Vision-Language Model

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1.0-brightgreen.svg)](https://pytorch.org/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Private Repository](https://img.shields.io/badge/Repository-Private-orange.svg)](#)

> A sophisticated multimodal AI model combining vision and language understanding with advanced reasoning capabilities. Thiran offers two variants: **Pro** (256M parameters) for maximum performance and **Lite** (21.5M parameters) for efficient deployment.
>
> **⚠️ PROPRIETARY SOFTWARE** - This is a private repository. Unauthorized access, reproduction, or distribution is strictly prohibited.

## 🎯 Features

- **Dual Architecture**: Pro and Lite model variants for different use cases
- **Multimodal Learning**: Seamless integration of visual and textual information
- **Advanced Reasoning**: Multi-step reasoning module for complex analysis
- **Custom Tokenizer**: Specialized vocabulary building for domain-specific tasks
- **Indic Language Support**: Pre-trained on Indic Instruct dataset with English optimization
- **Vision-Language Fusion**: Sophisticated fusion mechanisms for multimodal understanding
- **Image Reconstruction**: Capability to reconstruct visual representations

## 📋 Requirements

- Python 3.10+
- PyTorch 2.1.0
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for Pro model)

## 🚀 Quick Start

### Installation

```bash
# Clone the private repository (requires access)
git clone https://github.com/Boominathan2355/vision-model.git
cd vision-model

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** This is a private repository. Access credentials are required for cloning.

### Basic Usage

```python
import torch
from tokenizer import CustomTokenizer
from builder import ThiranModelBuilder
from architecture import ImagePreprocessor

# Initialize tokenizer
tokenizer = CustomTokenizer(max_length=512)
sample_texts = ["What is in this image?", "Analyze the visual content."]
tokenizer.build_vocab(sample_texts, min_freq=1)

# Build model
model = ThiranModelBuilder.build_pro_model(len(tokenizer.vocab))
model.eval()

# Prepare inputs
text = "What is in this image?"
text_tokens = tokenizer.encode(text)
text_tensor = torch.tensor([text_tokens])

image = torch.randn(1, 3, 224, 224)  # Dummy image tensor

# Inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
text_tensor = text_tensor.to(device)
image = image.to(device)

with torch.no_grad():
    outputs = model(text_tensor, image)
    
print(f"Classification logits: {outputs['logits'].shape}")
print(f"Reasoning logits: {outputs['reasoning_logits'].shape}")
```

## 📚 Model Architecture

### Pro Model (256M parameters)
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Fusion Layers**: 6
- **Reasoning Steps**: 4
- **Best for**: Maximum performance, research, production systems

### Lite Model (21.5M parameters)
- **Hidden Size**: 384
- **Attention Heads**: 8
- **Fusion Layers**: 3
- **Reasoning Steps**: 2
- **Best for**: Edge devices, mobile, resource-constrained environments

## 🏋️ Training

### Training with Indic Instruct Dataset

```bash
python train_final.py
```

**Configuration:**
- Dataset: Indic Instruct (English, 500 samples)
- Batch Size: 4
- Learning Rate: 2e-4
- Epochs: 2
- Optimizer: AdamW

### Custom Training

Edit `train_final.py` to customize:
- `max_samples`: Number of training samples
- `batch_size`: Batch size for training
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimizer

```python
config.batch_size = 8
config.num_epochs = 5
config.learning_rate = 1e-4
```

## 📁 Project Structure

```
thiran/
├── architecture.py           # Model architecture definitions
├── builder.py               # Model builder and utilities
├── tokenizer.py             # Custom tokenizer implementation
├── model.py                 # Model definitions
├── main.py                  # Example inference script
├── train_final.py           # Training script with Indic dataset
├── train_indic_direct.py    # Alternative training script
├── train_with_indic_data.py # Dataset integration script
├── custom_tokenizer.json    # Saved tokenizer
├── thiran_pro_model.pt      # Saved Pro model
├── thiran_lite_model.pt     # Saved Lite model
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔧 Key Components

### Architecture (`architecture.py`)
- **ImagePreprocessor**: Image normalization and preprocessing
- **ThinkingLayer**: Multi-head attention with FFN for reasoning
- **ReasoningModule**: Multi-step reasoning with cross-attention
- **MultimodalEncoder**: Vision and language encoding
- **FusionModule**: Multimodal fusion mechanisms
- **ThiranModel**: Main model class

### Builder (`builder.py`)
- **ThiranModelBuilder**: Builds Pro and Lite variants
- Model initialization and weight management
- Save/load utilities

### Tokenizer (`tokenizer.py`)
- **CustomTokenizer**: Domain-specific tokenization
- Vocabulary building from corpus
- Token encoding/decoding
- Special tokens for reasoning

## 📊 Model Outputs

The model returns a dictionary with:
```python
{
    'logits': torch.Tensor,              # Classification logits
    'reasoning_logits': torch.Tensor,    # Reasoning output logits
    'features': {
        'text_features': torch.Tensor,        # Text embeddings
        'visual_features': torch.Tensor,      # Visual embeddings
        'fused_features': torch.Tensor,       # Fused representations
        'reasoned_features': torch.Tensor,    # Reasoned features
    },
    'reconstructed_image': torch.Tensor  # Reconstructed image (Pro only)
}
```

## 🎓 Dataset

### Indic Instruct Dataset
- **Source**: `ai4bharat/indic-instruct-data-v0.1`
- **Configuration**: English only (anudesh split)
- **Format**: Parquet files
- **Size**: 5234+ samples available
- **Authentication**: Hugging Face token required

To use your own token:
```python
hf_token = "hf_your_token_here"
dataset = IndicInstructDataset(tokenizer, hf_token=hf_token)
```

## 🔐 Security

- **Hugging Face Token**: Never commit tokens to version control
- Use environment variables for sensitive credentials
- See `.gitignore` for files excluded from Git

## 💾 Model Persistence

### Saving Models
```python
from builder import save_model

save_model(model, tokenizer, 'my_model.pt')
tokenizer.save('my_tokenizer.json')
```

### Loading Models
```python
from builder import load_model

model = load_model('my_model.pt', tokenizer)
```

## 🚀 Performance

### Inference Speed (CPU)
- Pro Model: ~2-5 seconds per sample
- Lite Model: ~0.5-1 second per sample

### Memory Usage
- Pro Model: ~1GB
- Lite Model: ~200MB

*Note: Times vary based on hardware and sequence length*

## 🐛 Troubleshooting

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode
CUDA_VISIBLE_DEVICES="" python train_final.py
```

### Dataset Loading
```bash
# Update Hugging Face token
huggingface-cli login

# Or set environment variable
set HF_TOKEN=your_token_here
```

### Memory Issues
```python
# Reduce batch size
config.batch_size = 2

# Use Lite model instead
model = ThiranModelBuilder.build_lite_model(vocab_size)
```

## 📝 License

This project is licensed under a **Proprietary License** - see the [LICENSE](LICENSE) file for details.

**⚠️ IMPORTANT:** This software contains proprietary and confidential information. 
- Unauthorized use, reproduction, or distribution is strictly prohibited
- All intellectual property rights are reserved
- Access is restricted to authorized personnel only

For licensing inquiries, contact: support@thiran.ai

## 🔐 Security & Confidentiality

- **Classification**: PROPRIETARY
- **Access Control**: Private repository with restricted access
- **Data Protection**: All source code and models are confidential
- **Usage Rights**: Internal use only unless explicitly authorized

## 👥 Access & Contributing

This is a **private repository**. Access is restricted to authorized team members only.

For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md) (internal use only)

## 📞 Contact & Support

- **Issues**: GitHub Issues
- **Email**: support@thiran.ai
- **Documentation**: [Full Docs](./docs)

## 🙏 Acknowledgments

- Built with PyTorch and Hugging Face transformers
- Inspired by vision-language models like CLIP and BLIP
- Dataset: ai4bharat/indic-instruct-data-v0.1
- Thanks to the open-source community

## 📚 References

- Vaswani et al. (2017) - Attention Is All You Need
- Dosovitskiy et al. (2020) - An Image is Worth 16x16 Words
- Li et al. (2022) - BLIP: Bootstrapping Language-Image Pre-training
- Radford et al. (2021) - Learning Transferable Visual Models From Natural Language Supervision (CLIP)

## 📄 Changelog

### [1.0.0] - 2025-12-06
- Initial release
- Pro and Lite model variants
- Training with Indic Instruct dataset
- Custom tokenizer implementation
- Full inference pipeline

---

**Made with ❤️ by the Thiran team**
#   v i s i o n - m o d e l 
 
 