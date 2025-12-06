# Contributing to Thiran

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents) for different ways to help and details about how this project handles them.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Improving The Documentation](#improving-the-documentation)
- [Styleguides](#styleguides)
  - [Commit Messages](#commit-messages)
  - [Code Style](#code-style)
- [Join The Project Team](#join-the-project-team)

## Code of Conduct

This project and everyone participating in it is governed by the [Thiran Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

## I Have a Question

> If you want to ask a question, we assume that you have read the available [Documentation](README.md).

Before you ask a question, it is best to search for existing [Issues](https://github.com/yourrepo/thiran/issues) that might help you. In case that you have found a suitable issue and still need clarification, you can write your question in this issue. It is also advisable to search the internet for answers first.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](https://github.com/yourrepo/thiran/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (python, pytorch, cuda, etc.).

We will then take care of the issue as soon as possible.

## I Want To Contribute

### Reporting Bugs

#### Before Submitting a Bug Report

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side (e.g. using incompatible environment variables or outdated dependencies).
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/yourrepo/thiran/issues?q=label%3Abug).
- Collect information about the bug:
  - Stack trace (traceback)
  - OS, Platform and Version (Windows, Linux, macOS)
  - Python version
  - PyTorch version
  - CUDA version (if applicable)
  - Your input and the output
  - Can you reliably reproduce the issue?

#### How Do I Submit a Good Bug Report?

> You must never report security related issues, vulnerabilities or bugs including sensitive information to the issue tracker, or elsewhere in public. Instead sensitive bugs must be sent by email to <support@thiran.ai>.

We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- **Use a clear and descriptive title**
- **Describe the exact steps which reproduce the problem** in as many details as possible.
- **Provide specific examples to demonstrate the steps**. Include links to files or GitHub projects, or copy/pasteable snippets, which you use in those examples.
- **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
- **Explain which behavior you expected to see instead and why.**
- **Include screenshots and animated GIFs if possible.**
- **Include your environment details.**

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for Thiran, including completely new features and minor improvements to existing functionality.

#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the documentation carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](https://github.com/yourrepo/thiran/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.

#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://github.com/yourrepo/thiran/issues).

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior** and **outline the expected behavior**
- **Explain why this enhancement would be useful**

### Your First Code Contribution

#### Local development

1. Fork the repository
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/thiran.git
   cd thiran
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest flake8 black
   ```

5. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

6. Make your changes and test them:
   ```bash
   # Run tests
   pytest tests/
   
   # Format code
   black .
   
   # Lint code
   flake8 .
   ```

7. Commit your changes (see [Commit Messages](#commit-messages)):
   ```bash
   git add .
   git commit -m "Add your message here"
   ```

8. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

9. Open a Pull Request on GitHub

### Improving The Documentation

- Fix typos and grammatical errors
- Add examples or clarifications
- Update outdated information
- Improve code comments

## Styleguides

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line
- Consider starting the commit message with an applicable emoji:
  - 🎨 `:art:` Improve structure/format
  - ⚡️ `:zap:` Improve performance
  - 🔧 `:wrench:` Configuration files
  - 📦 `:package:` Add package/dependency
  - 🐛 `:bug:` Fix bug
  - ✨ `:sparkles:` Introduce new feature
  - 📝 `:memo:` Documentation
  - ✅ `:white_check_mark:` Add tests
  - 🔒 `:lock:` Security fix

Example:
```
✨ Add reasoning module to Pro model

- Implement multi-step reasoning with cross-attention
- Add reasoning_logits to model outputs
- Update training script to handle reasoning loss

Fixes #123
```

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use 4 spaces for indentation
- Use `black` for code formatting
- Add docstrings to all functions and classes
- Use type hints where applicable
- Keep lines under 100 characters when possible

Example:
```python
def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    model_name: str = "Model"
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        config: Training configuration
        epoch: Current epoch number
        model_name: Name of the model for logging
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        # Training code here
        pass
    
    return total_loss / len(train_loader)
```

## Join The Project Team

If you would like to join the Thiran team, please reach out to us at support@thiran.ai with:
- Your GitHub profile
- Your interest areas
- Your relevant experience
- Why you want to contribute

---

**Thanks for contributing to Thiran! 🙌**
