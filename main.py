"""
Simple LLM Demo using Transformers
This script demonstrates basic text generation using a pre-trained language model.
"""

from transformers import pipeline

def main():
    print("Loading language model...")
    
    # Using a smaller model for quick testing
    # You can replace with other models like "gpt2-medium", "gpt2-large", etc.
    generator = pipeline('text-generation', model='gpt2')
    
    print("Model loaded successfully!\n")
    
    # Example prompts
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence is",
        "In a world where technology"
    ]
    
    print("Generating text completions...\n")
    
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        result = generator(
            prompt,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7
        )
        print(f"Generated: {result[0]['generated_text']}\n")
        print("-" * 80 + "\n")

if __name__ == "__main__":
    main()
