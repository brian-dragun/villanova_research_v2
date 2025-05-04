"""
Model Pruning Module

This module implements various pruning strategies for LLM weight reduction.
"""

import os
import torch
import numpy as np

from ..core.utils import debug_print, Timer, log_section, ensure_dir, log_success
from ..config import MODEL_PATHS, ANALYSIS_CONFIG
from ..sensitivity.super_weights import compute_weight_statistics, find_super_weights

def prune_model_weights(model, threshold=None, strategy="magnitude", sensitivity_data=None):
    """
    Prune model weights using the specified strategy.
    
    Args:
        model: The PyTorch model to prune
        threshold: Threshold value for pruning (interpretation depends on strategy)
        strategy: Pruning strategy ("magnitude", "sensitivity", "random")
        sensitivity_data: Pre-computed sensitivity metrics (required for "sensitivity" strategy)
        
    Returns:
        pruned_model: The pruned model
        stats: Pruning statistics
    """
    # Set default threshold if not provided
    if threshold is None:
        threshold = ANALYSIS_CONFIG["pruning_threshold"]
        
    # Statistics to track pruning results
    stats = {
        "total_params": 0,
        "pruned_params": 0,
        "sparsity": 0,
        "pruned_by_layer": {},
    }
    
    # Apply pruning based on the strategy
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Count original parameters
        param_size = param.numel()
        stats["total_params"] += param_size
        
        # Get the mask based on the strategy
        if strategy == "magnitude":
            # Magnitude-based pruning (prune smallest weights)
            mask = torch.abs(param.data) > threshold * torch.max(torch.abs(param.data))
        
        elif strategy == "sensitivity":
            # Sensitivity-based pruning (requires sensitivity data)
            if sensitivity_data is None or name not in sensitivity_data:
                debug_print(f"No sensitivity data for {name}, using magnitude pruning")
                mask = torch.abs(param.data) > threshold * torch.max(torch.abs(param.data))
            else:
                # Get sensitivity scores
                scores = sensitivity_data[name]["sensitivity"]
                # Create mask based on sensitivity
                mask = scores > threshold * torch.max(scores)
        
        elif strategy == "random":
            # Random pruning
            mask = torch.rand_like(param.data) > threshold
            
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}")
        
        # Apply the mask
        param.data = param.data * mask
        
        # Count pruned parameters
        pruned_count = param_size - torch.sum(mask).item()
        stats["pruned_params"] += pruned_count
        stats["pruned_by_layer"][name] = {
            "total": param_size,
            "pruned": pruned_count,
            "sparsity": pruned_count / param_size
        }
    
    # Calculate overall sparsity
    stats["sparsity"] = stats["pruned_params"] / stats["total_params"]
    
    return model, stats

def prune_model_structured(model, importance_scores, prune_percentage=0.1):
    """
    Apply structured pruning to the model (prune entire structures like heads or neurons).
    
    Args:
        model: The PyTorch model to prune
        importance_scores: Dictionary mapping structures to their importance scores
        prune_percentage: Percentage of structures to prune
        
    Returns:
        pruned_model: The pruned model
        stats: Pruning statistics
    """
    # This is a placeholder for structured pruning implementation
    # In practice, you would identify important structures and prune them
    
    debug_print("Structured pruning not yet implemented")
    return model, {"method": "structured", "pruned_percentage": prune_percentage}

def compute_pruning_sensitivity(model, eval_func):
    """
    Compute sensitivity to pruning for each layer.
    
    Args:
        model: The PyTorch model to analyze
        eval_func: Evaluation function that returns a performance metric
        
    Returns:
        Dictionary mapping layer names to pruning sensitivity scores
    """
    baseline_score = eval_func(model)
    sensitivity = {}
    
    # Create a copy for experimentation
    import copy
    test_model = copy.deepcopy(model)
    
    # Test each layer's sensitivity to pruning
    for name, param in test_model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Save original parameter
        original_param = param.data.clone()
        
        # Apply small pruning (e.g., 1%)
        mask = torch.rand_like(param.data) > 0.01
        param.data = param.data * mask
        
        # Evaluate
        score = eval_func(test_model)
        
        # Compute sensitivity
        sensitivity[name] = {
            "baseline": baseline_score,
            "pruned": score,
            "impact": abs(baseline_score - score) / baseline_score
        }
        
        # Restore original parameter
        param.data.copy_(original_param)
    
    return sensitivity

def apply_pruning(model_path, output_path, strategy="magnitude", threshold=None):
    """
    Load a model, apply pruning, and save the pruned model.
    
    Args:
        model_path: Path to the original model
        output_path: Path to save the pruned model
        strategy: Pruning strategy
        threshold: Pruning threshold
    
    Returns:
        stats: Pruning statistics
    """
    log_section("Model Pruning")
    
    # Load the model
    from transformers import AutoModelForCausalLM
    debug_print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    
    # Apply pruning
    with Timer(f"Pruning model using {strategy} strategy"):
        pruned_model, stats = prune_model_weights(model, threshold, strategy)
    
    # Save the pruned model
    ensure_dir(os.path.dirname(output_path))
    debug_print(f"Saving pruned model to {output_path}")
    torch.save(pruned_model.state_dict(), output_path)
    
    # Log results
    log_success(f"Pruned {stats['pruned_params']} parameters ({stats['sparsity']:.2%} sparsity)")
    
    return stats

def prune_model(input_path=MODEL_PATHS["finetuned"], 
                output_path=MODEL_PATHS["pruned"],
                strategy="magnitude",
                threshold=None):
    """
    Main entry point for model pruning.
    
    Args:
        input_path: Path to the input model
        output_path: Path to save the pruned model
        strategy: Pruning strategy
        threshold: Pruning threshold
    
    Returns:
        stats: Pruning statistics
    """
    return apply_pruning(input_path, output_path, strategy, threshold)