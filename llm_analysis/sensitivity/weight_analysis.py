"""
Weight Sensitivity Analysis Module

This module consolidates functionality from llm_analyze_sensitivity.py and 
llm_weight_sensitivity_analysis.py to provide comprehensive weight sensitivity analysis.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from ..core.utils import debug_print, Timer, log_section, ensure_dir
from ..config import MODEL_NAME, MODEL_PATHS, OUTPUT_DIR, ANALYSIS_CONFIG

def compute_hessian_diagonal(loss_func, param_data, subset_size=None):
    """
    Compute the diagonal of the Hessian matrix for a given parameter.
    
    Args:
        loss_func: Function that computes the loss
        param_data: Parameter data tensor
        subset_size: Number of elements to analyze (None for all)
        
    Returns:
        Diagonal of the Hessian
    """
    device = param_data.device
    
    # If subset_size is specified, choose random indices
    if subset_size is not None and subset_size < param_data.numel():
        flat_indices = torch.randperm(param_data.numel(), device=device)[:subset_size]
        indices = np.unravel_index(flat_indices.cpu().numpy(), param_data.shape)
    else:
        # Use all elements
        indices = np.where(torch.ones_like(param_data).cpu().numpy())
        
    results = torch.zeros(len(indices[0]), device=device)
    
    # Compute second derivative for each selected element
    for i in range(len(indices[0])):
        idx = tuple(ind[i] for ind in indices)
        
        # Forward pass with +ε
        param_data_plus = param_data.clone()
        param_data_plus[idx] += 1e-4
        loss_plus = loss_func(param_data_plus)
        
        # Forward pass with -ε
        param_data_minus = param_data.clone()
        param_data_minus[idx] -= 1e-4
        loss_minus = loss_func(param_data_minus)
        
        # Second derivative approximation
        results[i] = (loss_plus - 2 * loss_func(param_data) + loss_minus) / (1e-4 ** 2)
    
    # Reshape results back to original tensor shape if needed
    if subset_size is None or subset_size >= param_data.numel():
        return results.view(param_data.shape)
    else:
        # For subset, return flat tensor and indices
        return results, indices

def identify_sensitive_weights(model, loss_func, threshold_z=2.5):
    """
    Identify sensitive weights in a model using Hessian-based analysis.
    
    Args:
        model: The PyTorch model to analyze
        loss_func: Function to compute loss for a parameter
        threshold_z: Z-score threshold for identifying outliers
        
    Returns:
        Dictionary mapping parameter names to their sensitivity metrics
    """
    debug_print(f"Running weight sensitivity analysis with threshold z={threshold_z}")
    
    sensitivity_by_param = {}
    all_scores = []
    
    # Process each parameter
    for name, param in model.named_parameters():
        if param.requires_grad:
            with Timer(f"Analyzing {name}"):
                # Compute Hessian diagonals for a subset of the parameter
                subset_size = min(ANALYSIS_CONFIG["subsample_size"], param.numel())
                hessian_diag, indices = compute_hessian_diagonal(
                    loss_func, 
                    param.data, 
                    subset_size=subset_size
                )
                
                # Store results
                sensitivity_by_param[name] = {
                    'hessian_diag': hessian_diag,
                    'indices': indices,
                    'shape': param.shape,
                    'mean': hessian_diag.mean().item(),
                    'std': hessian_diag.std().item(),
                    'max': hessian_diag.max().item(),
                    'min': hessian_diag.min().item(),
                }
                
                # Collect all scores for global analysis
                all_scores.extend(hessian_diag.cpu().tolist())
    
    # Compute global statistics
    global_mean = np.mean(all_scores)
    global_std = np.std(all_scores)
    
    # Identify super weights (outliers)
    super_weights_by_param = {}
    for name, metrics in sensitivity_by_param.items():
        hessian_diag = metrics['hessian_diag']
        
        # Compute z-scores
        z_scores = (hessian_diag - global_mean) / global_std
        
        # Find outliers
        outlier_mask = (z_scores > threshold_z)
        outlier_indices = torch.where(outlier_mask)
        
        if outlier_indices[0].size(0) > 0:
            # Map flat indices back to parameter indices
            param_indices = []
            for i in range(len(outlier_indices[0])):
                idx = tuple(outlier_indices[j][i] for j in range(len(outlier_indices)))
                param_idx = tuple(metrics['indices'][j][idx[0]] for j in range(len(metrics['indices'])))
                param_indices.append(param_idx)
            
            # Store super weight information
            super_weights_by_param[name] = {
                'count': outlier_indices[0].size(0),
                'indices': param_indices,
                'values': hessian_diag[outlier_indices].tolist(),
                'z_scores': z_scores[outlier_indices].tolist(),
            }
    
    return sensitivity_by_param, super_weights_by_param

def visualize_weight_sensitivity(sensitivity_by_param, output_dir=OUTPUT_DIR):
    """
    Visualize weight sensitivity across model layers.
    
    Args:
        sensitivity_by_param: Dictionary of sensitivity metrics by parameter
        output_dir: Directory to save visualizations
    """
    ensure_dir(output_dir)
    
    # Extract layer names and sensitivity metrics
    layer_names = []
    mean_sensitivities = []
    max_sensitivities = []
    
    for name, metrics in sensitivity_by_param.items():
        layer_names.append(name)
        mean_sensitivities.append(metrics['mean'])
        max_sensitivities.append(metrics['max'])
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot mean sensitivities
    plt.subplot(2, 1, 1)
    plt.bar(range(len(layer_names)), mean_sensitivities, alpha=0.7)
    plt.xticks([], [])
    plt.title("Mean Weight Sensitivity by Layer")
    plt.ylabel("Mean Sensitivity")
    
    # Plot max sensitivities
    plt.subplot(2, 1, 2)
    plt.bar(range(len(layer_names)), max_sensitivities, alpha=0.7, color='orange')
    plt.xticks(range(len(layer_names)), [name.split('.')[-2] for name in layer_names], 
               rotation=90, fontsize=8)
    plt.title("Max Weight Sensitivity by Layer")
    plt.ylabel("Max Sensitivity")
    plt.xlabel("Layers")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'layerwise_sensitivity.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Sensitivity visualization saved to {output_path}")
    
    # Create super weights visualization
    plt.figure(figsize=(14, 8))
    plt.bar(range(len(layer_names)), [metrics.get('max', 0) for name, metrics in sensitivity_by_param.items()], alpha=0.7)
    plt.xticks(range(len(layer_names)), [name.split('.')[-2] for name in layer_names], 
               rotation=90, fontsize=8)
    plt.title("Layer-wise Super Weights Distribution")
    plt.ylabel("Maximum Sensitivity")
    plt.xlabel("Layers")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'layerwise_superweights.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Super weights visualization saved to {output_path}")

def run_sensitivity_analysis(model=None, input_text=None):
    """
    Run comprehensive weight sensitivity analysis on the model.
    
    Args:
        model: PyTorch model to analyze (loads from MODEL_PATHS["finetuned"] if None)
        input_text: Text to use for loss computation
        
    Returns:
        Dictionary containing sensitivity metrics and super weights
    """
    log_section("Weight Sensitivity Analysis")
    
    # Load model if not provided
    if model is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATHS["finetuned"], trust_remote_code=True)
    
    # Create a dummy loss function for demonstration
    # In practice, this should be replaced with a proper loss function
    # based on the input_text
    def loss_func(param_data):
        return torch.sum(param_data ** 2)
    
    # Run the sensitivity analysis
    with Timer("Complete sensitivity analysis"):
        sensitivity_by_param, super_weights_by_param = identify_sensitive_weights(
            model, 
            loss_func,
            threshold_z=2.5
        )
    
    # Visualize the results
    visualize_weight_sensitivity(sensitivity_by_param)
    
    return {
        "sensitivity_by_param": sensitivity_by_param,
        "super_weights_by_param": super_weights_by_param
    }

def conduct_weight_experiments():
    """Run a series of weight sensitivity experiments for research."""
    log_section("Weight Sensitivity Experiments")
    
    # This function would integrate code from llm_weight_sensitivity_analysis.py
    # to run various experiments on weight sensitivity
    
    # For now, it's a placeholder that calls the main analysis function
    results = run_sensitivity_analysis()
    
    # Additional experiments would be implemented here
    
    return results

if __name__ == "__main__":
    conduct_weight_experiments()