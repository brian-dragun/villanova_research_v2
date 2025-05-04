#!/usr/bin/env python3
"""
Run Model Script

This script allows you to quickly run a model with a prompt to test it.
"""

import argparse
import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import AVAILABLE_MODELS, TEST_PROMPT
from llm_analysis.core.model_loader import generate_text, load_model, load_tokenizer

def list_available_models():
    """List all available models with their details."""
    print("\n🔍 Available Models:")
    print("-" * 80)
    print(f"{'Key':<15} | {'Name':<35} | {'Auth Required':<12} | Details")
    print("-" * 80)
    
    for key, details in AVAILABLE_MODELS.items():
        print(f"{key:<15} | {details['name']:<35} | {'Yes' if details['requires_auth'] else 'No':<12} | {details['details']}")
    
    print("-" * 80)
    print("\nUsage example: python run_model.py --model gpt-neo-125m")

def main():
    parser = argparse.ArgumentParser(description='Run a language model with a prompt')
    parser.add_argument('--model', type=str, default='gpt-neo-125m', 
                        help='Model key or full HuggingFace model identifier')
    parser.add_argument('--prompt', type=str, default=TEST_PROMPT,
                        help='Prompt text to send to the model')
    parser.add_argument('--max-length', type=int, default=100,
                        help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for text generation (higher = more random)')
    parser.add_argument('--list-models', action='store_true',
                        help='List available model shortcuts')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    print(f"\n🚀 Running model: {args.model}")
    print(f"📝 Prompt: {args.prompt}")
    
    try:
        response = generate_text(
            args.model, 
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        print("\n✅ Model Response:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        
    except Exception as e:
        print(f"\n❌ Error running model: {e}")
        print("\nTry using --list-models to see available models")
        sys.exit(1)

if __name__ == "__main__":
    main()