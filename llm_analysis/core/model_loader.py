"""
Model Loading and Training Module

This module handles loading and fine-tuning LLM models.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from parent package
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODEL_PATHS, MODEL_NAME, MODEL_CONFIG, get_model_by_key, get_model_paths, HF_TOKEN, AVAILABLE_MODELS

def load_model(model_name_or_path, **kwargs):
    """
    Load a model from Hugging Face or local directory.
    
    Args:
        model_name_or_path: Name or path of the model to load
        **kwargs: Additional arguments to pass to from_pretrained
        
    Returns:
        The loaded model
    """
    print(f"🔄 Loading model {model_name_or_path}...")
    
    # Check if model_name_or_path is a model key in our config
    if model_name_or_path in AVAILABLE_MODELS:
        model_info = AVAILABLE_MODELS[model_name_or_path]
        model_name = model_info["name"]
        requires_auth = model_info["requires_auth"]
    else:
        model_name = model_name_or_path
        # Default to requiring auth for meta-llama models
        requires_auth = "meta-llama" in model_name.lower()
    
    # Set up loading parameters
    load_params = {**MODEL_CONFIG, **kwargs}
    
    # Only include auth token if needed
    if not requires_auth:
        load_params.pop("use_auth_token", None)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_params)
        model.to(device)
        print(f"✅ Model loaded successfully: {model_name}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def load_tokenizer(model_name_or_path, **kwargs):
    """
    Load a tokenizer for the specified model.
    
    Args:
        model_name_or_path: Name or path of the model
        **kwargs: Additional arguments to pass to from_pretrained
        
    Returns:
        The loaded tokenizer
    """
    # Check if model_name_or_path is a model key in our config
    if model_name_or_path in AVAILABLE_MODELS:
        model_name = AVAILABLE_MODELS[model_name_or_path]["name"]
        requires_auth = AVAILABLE_MODELS[model_name_or_path]["requires_auth"]
    else:
        model_name = model_name_or_path
        # Default to requiring auth for meta-llama models
        requires_auth = "meta-llama" in model_name.lower()
    
    # Set up loading parameters
    load_params = {
        "trust_remote_code": True,
        **kwargs
    }
    
    # Only include auth token if needed
    if requires_auth and HF_TOKEN:
        load_params["use_auth_token"] = HF_TOKEN
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_params)
        print(f"✅ Tokenizer loaded successfully for model: {model_name}")
        return tokenizer
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        raise

def load_or_train_model(model_name_or_path=None, skip_finetune=False):
    """
    Load a model from a directory or train it if it doesn't exist.
    
    Args:
        model_name_or_path: Name of model or path to the model directory
        skip_finetune: If True, raise an error if model_dir doesn't exist
        
    Returns:
        The loaded model
    """
    # Use default model if none provided
    if model_name_or_path is None:
        model_name_or_path = MODEL_NAME
    
    # Get the model name (not path) for logging
    if model_name_or_path in AVAILABLE_MODELS:
        display_name = AVAILABLE_MODELS[model_name_or_path]["name"]
    else:
        display_name = model_name_or_path
        
    # Determine if we're loading from a pre-trained model or a local directory
    if os.path.isdir(model_name_or_path):
        model_dir = model_name_or_path
    else:
        # Get the path based on the provided model name
        model_paths = get_model_paths(model_name_or_path)
        model_dir = model_paths["finetuned"]

    print(f"🔍 Looking for fine-tuned model at '{model_dir}'...")

    if not os.path.isdir(model_dir):
        if skip_finetune:
            print(f"⚠️ Model directory '{model_dir}' not found and skip_finetune is True")
            print(f"🔄 Loading base model from Hugging Face instead: {display_name}")
            return load_model(model_name_or_path)
        
        print(f"🚀 No model directory found at '{model_dir}'. Starting fine-tuning...")
        train_model(model_name_or_path, model_dir)
    else:
        print(f"✅ Found model directory at '{model_dir}', using pre-trained model.")

    try:
        model = load_model(model_dir)
        return model
    except Exception as e:
        print(f"⚠️ Error loading fine-tuned model: {e}")
        print(f"🔄 Loading base model from Hugging Face instead: {display_name}")
        return load_model(model_name_or_path)

def train_model(base_model, output_dir):
    """
    Fine-tune a pre-trained model and save it to output_dir.
    
    Args:
        base_model: Name or path of the base model to fine-tune
        output_dir: Directory to save the fine-tuned model
    """
    # Import training functionality from the root directory
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from llm_train import fine_tune_model
        fine_tune_model(base_model, output_dir)
    except ImportError:
        # Placeholder that would be replaced with actual code from llm_train.py
        print("❌ Training module (llm_train.py) not found or couldn't be imported.")
        raise NotImplementedError("Training functionality will be imported from llm_train.py")
    
def generate_text(model_name_or_path, prompt, max_length=100, temperature=0.7):
    """
    Generate text from a prompt using the specified model.
    
    Args:
        model_name_or_path: Name or path of the model to use
        prompt: Text prompt to generate from
        max_length: Maximum length of generated text
        temperature: Temperature for text generation (higher = more random)
        
    Returns:
        Generated text
    """
    # Resolve model name if it's a key
    if model_name_or_path in AVAILABLE_MODELS:
        model_name = AVAILABLE_MODELS[model_name_or_path]["name"]
        requires_auth = AVAILABLE_MODELS[model_name_or_path]["requires_auth"]
    else:
        model_name = model_name_or_path
        # Default to requiring auth for meta-llama models
        requires_auth = "meta-llama" in model_name.lower()
    
    print(f"🤖 Generating text using model: {model_name}")
    
    # Load the model and tokenizer with appropriate auth
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model with auth token if required
    load_params = {"trust_remote_code": True}
    if requires_auth and HF_TOKEN:
        load_params["use_auth_token"] = HF_TOKEN
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_params)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_params)
    except Exception as e:
        print(f"❌ Error loading model or tokenizer: {e}")
        raise
    
    model.to(device)
    model.eval()
    
    print(f"📝 Prompt: {prompt}")

    # Tokenize the input prompt with an attention mask
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
    except Exception as e:
        print(f"❌ Error tokenizing input: {e}")
        raise

    # Generate an answer
    try:
        # Check if pad_token_id is available, otherwise use eos_token_id
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        
        # Create generation config
        gen_config = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_length": max_length,
            "num_beams": 5,
            "no_repeat_ngram_size": 2,
            "temperature": temperature,
            "top_k": 50,
            "early_stopping": True,
            "pad_token_id": pad_token_id
        }
        
        # Remove parameters that might not be supported by all models
        if "falcon" in model_name.lower():
            gen_config.pop("no_repeat_ngram_size", None)
        
        generated_ids = model.generate(**gen_config)
        answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"✅ Generation successful")
        return answer
    except Exception as e:
        print(f"❌ Error during text generation: {e}")
        raise