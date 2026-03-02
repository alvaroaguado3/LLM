# LLM Demo Project

A simple demo project showcasing text generation using Transformers and PyTorch.

## Overview

This project demonstrates basic usage of the Hugging Face Transformers library for text generation using pre-trained language models.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the demo script:
```bash
python main.py
```

The script will:
1. Load the GPT-2 model (downloads on first run)
2. Generate text completions for sample prompts
3. Display the results

## Customization

- **Change the model**: Edit `main.py` and replace `'gpt2'` with other models like `'gpt2-medium'`, `'gpt2-large'`, or other compatible models from Hugging Face Model Hub.
- **Adjust generation parameters**: Modify `max_length`, `temperature`, and other parameters in the `generator()` call.
- **Add your own prompts**: Update the `prompts` list in `main.py`.

## Project Structure

```
LLM/
├── main.py              # Main demo script
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Model Hub](https://huggingface.co/models)
- [PyTorch Documentation](https://pytorch.org/docs)
