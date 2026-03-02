# LLM Demo Project

## Project Overview
A Python-based demo project for text generation using Hugging Face Transformers and PyTorch.

## Technology Stack
- Python 3.11
- Transformers 5.2.0
- PyTorch 2.10.0
- GPT-2 model

## Setup Instructions
1. Virtual environment is configured in `venv/`
2. Dependencies are installed via `requirements.txt`
3. Python environment is configured for VS Code

## Running the Project
Execute the demo:
```bash
source venv/bin/activate
python main.py
```

Or use the configured Python interpreter:
```bash
/Users/alvaroaguado/repos/LLM/venv/bin/python main.py
```

## Project Structure
- `main.py` - Main demo script for text generation
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `venv/` - Virtual environment

## Development Guidelines
- Use the virtual environment for all Python operations
- Model files are cached in `.cache/` after first download
- Modify prompts in `main.py` to generate different text
