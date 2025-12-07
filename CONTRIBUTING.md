# Contributing to VelCore

Thank you for your interest in contributing to VelCore! 🎉

## How to Contribute

### Reporting Bugs

1. Check if the issue already exists in [Issues](https://github.com/Boominathan2355/vision-model/issues)
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, PyTorch version, OS)

### Feature Requests

1. Open an issue with the `enhancement` label
2. Describe the feature and its use case
3. Discuss implementation approach

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests and ensure code quality
5. Commit with clear messages: `git commit -m "Add: feature description"`
6. Push to your fork: `git push origin feature/your-feature`
7. Open a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters
- Add docstrings for classes and functions
- Keep functions focused and concise

## Project Structure

```
VelCore/
├── architecture.py    # Model architecture
├── builder.py         # Model builder utilities
├── tokenizer.py       # Custom tokenizer
├── train_*.py         # Training scripts
├── ask_model.py       # Inference script
└── generate_image.py  # Image generation
```

## Testing

```bash
# Run basic tests
python main.py
python ask_model.py
python generate_image.py
```

## Areas to Contribute

- [ ] Add more training datasets
- [ ] Improve image generation quality
- [ ] Add web interface
- [ ] Performance optimizations
- [ ] Documentation improvements
- [ ] Unit tests

## Questions?

Open an issue or reach out to the maintainers.

---

Thank you for contributing! 🚀
