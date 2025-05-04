"""
Model Evaluation Metrics Module

This module provides functions for evaluating LLM performance.
"""

import os
import torch
import math
import time
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from ..core.utils import debug_print, Timer, log_section, ensure_dir, log_success, log_error
from ..config import MODEL_NAME, MODEL_PATHS, EVAL_CONFIG

# Define the checkpoint file in the data folder
CHECKPOINT_FILE = os.path.join("data", "eval_checkpoint.txt")

def calculate_perplexity(model, tokenizer, texts, max_length=EVAL_CONFIG["max_seq_length"], 
                         batch_size=EVAL_CONFIG["batch_size"]):
    """
    Calculate perplexity for a list of texts.
    
    Args:
        model: Language model
        tokenizer: Tokenizer for the model
        texts: List of text strings
        max_length: Maximum sequence length for evaluation
        batch_size: Batch size for processing
        
    Returns:
        Perplexity score (lower is better)
    """
    # Ensure model is in evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    total_loss = 0.0
    total_tokens = 0
    
    # Create batches
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    with torch.no_grad():
        for text_batch in tqdm(batches, desc="Calculating perplexity"):
            # Tokenize batch
            inputs = tokenizer(text_batch, return_tensors="pt", padding=True, 
                              truncation=True, max_length=max_length)
            
            # Skip empty batches
            if inputs["input_ids"].size(1) == 0:
                continue
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass with labels for loss calculation
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            
            # Count tokens (excluding padding)
            num_tokens = inputs["input_ids"].ne(tokenizer.pad_token_id).sum().item()
            total_loss += loss * num_tokens
            total_tokens += num_tokens
    
    # Calculate perplexity
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

def load_evaluation_dataset(dataset_name=EVAL_CONFIG["dataset"], 
                           split=EVAL_CONFIG["split"]):
    """
    Load a dataset for evaluation.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split to use
        
    Returns:
        List of text samples
    """
    try:
        dataset = load_dataset(dataset_name, split=split)
        
        # Extract texts and filter empty examples
        texts = []
        for example in dataset:
            if "text" in example and example["text"].strip():
                texts.append(example["text"].strip())
        
        debug_print(f"Loaded {len(texts)} samples from {dataset_name} ({split})")
        return texts
    except Exception as e:
        log_error(f"Failed to load dataset: {e}")
        return []

def evaluate_model_with_checkpointing(model_path, dataset_split=EVAL_CONFIG["split"], 
                                     batch_size=EVAL_CONFIG["batch_size"]):
    """
    Evaluate a model with checkpoint support for resuming interrupted evaluations.
    
    Args:
        model_path: Path to the model directory or weights file
        dataset_split: Dataset split to use for evaluation
        batch_size: Batch size for processing
        
    Returns:
        Perplexity score
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    debug_print(f"Evaluating model at {model_path} on {device}")
    
    # Load model with error handling
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if os.path.isdir(model_path):
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
            # Handle different state dict formats
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
            except Exception as e:
                debug_print(f"Standard loading failed: {e}. Trying alternative loading...")
                state_dict = torch.load(model_path, map_location=device)
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                else:
                    model.load_state_dict(state_dict)
    except Exception as e:
        log_error(f"Failed to load model: {e}")
        return None
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset_texts = load_evaluation_dataset(split=dataset_split)
    if not dataset_texts:
        log_error("No evaluation data available")
        return None
    
    # Check for checkpoint
    start_index = 0
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                start_index = int(f.read().strip())
                debug_print(f"Resuming evaluation from index {start_index}")
        except Exception as e:
            debug_print(f"Could not read checkpoint file, starting from index 0: {e}")
    
    # Skip processed examples
    dataset_texts = dataset_texts[start_index:]
    
    # Calculate perplexity
    with Timer("Perplexity calculation"):
        perplexity = calculate_perplexity(model, tokenizer, dataset_texts, 
                                         batch_size=batch_size)
    
    # Remove checkpoint file after successful evaluation
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    log_success(f"Perplexity of model at {model_path}: {perplexity:.2f}")
    return perplexity

def evaluate_model(model_path, dataset_split=EVAL_CONFIG["split"], 
                  batch_size=EVAL_CONFIG["batch_size"]):
    """
    Public interface for model evaluation.
    
    Args:
        model_path: Path to the model directory or weights file
        dataset_split: Dataset split to use for evaluation
        batch_size: Batch size for processing
        
    Returns:
        Perplexity score
    """
    log_section(f"Model Evaluation: {os.path.basename(model_path)}")
    
    return evaluate_model_with_checkpointing(model_path, dataset_split, batch_size)