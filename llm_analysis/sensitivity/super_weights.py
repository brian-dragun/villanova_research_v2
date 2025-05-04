"""
Super Weights Analysis Module

This module identifies and analyzes "super weights" - the weights that have
disproportionate influence on model behavior.
"""

import torch
import numpy as np
from collections import defaultdict

from ..core.utils import debug_print, Timer, log_section, ensure_dir
from ..config import ANALYSIS_CONFIG

def compute_weight_statistics(param_data):
    """
    Compute summary statistics for a parameter tensor.
    
    Args:
        param_data: Parameter data tensor
    
    Returns:
        Dictionary of statistics
    """
    # Convert to numpy for statistical operations
    if isinstance(param_data, torch.Tensor):
        param_np = param_data.detach().cpu().numpy()
    else:
        param_np = param_data
        
    # Compute statistics
    stats = {
        'mean': float(np.mean(param_np)),
        'std': float(np.std(param_np)),
        'min': float(np.min(param_np)),
        'max': float(np.max(param_np)),
        '25th': float(np.percentile(param_np, 25)),
        'median': float(np.median(param_np)),
        '75th': float(np.percentile(param_np, 75)),
        'abs_mean': float(np.mean(np.abs(param_np))),
        'skew': float(compute_skewness(param_np)),
        'kurtosis': float(compute_kurtosis(param_np)),
    }
    
    return stats

def compute_skewness(data):
    """Compute the skewness of a distribution."""
    n = len(data)
    if n == 0:
        return 0
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0
        
    # Compute skewness
    skew = np.sum(((data - mean) / std) ** 3) / n
    return skew

def compute_kurtosis(data):
    """Compute the kurtosis of a distribution."""
    n = len(data)
    if n == 0:
        return 0
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0
        
    # Compute kurtosis
    kurt = np.sum(((data - mean) / std) ** 4) / n - 3  # -3 for excess kurtosis
    return kurt

def find_super_weights(model, threshold_z=2.5):
    """
    Identify super weights in a model based on their magnitude.
    
    Args:
        model: PyTorch model to analyze
        threshold_z: Z-score threshold for identifying outliers
        
    Returns:
        Dictionary of super weights by layer
    """
    debug_print(f"Finding super weights with threshold z={threshold_z}")
    
    # Collect all weight values for global statistics
    all_weights = []
    layer_weights = {}
    
    # First pass: collect weights
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights = param.data.detach().cpu().numpy().flatten()
            all_weights.extend(weights)
            layer_weights[name] = weights
    
    # Compute global statistics
    global_mean = np.mean(all_weights)
    global_std = np.std(all_weights)
    debug_print(f"Global weight stats: Mean={global_mean:.6f}, Std={global_std:.6f}")
    
    # Second pass: identify super weights
    super_weights = {}
    total_super_weights = 0
    
    for name, weights in layer_weights.items():
        # Compute z-scores
        z_scores = (weights - global_mean) / global_std
        
        # Find outliers based on z-score
        outlier_indices = np.where(np.abs(z_scores) > threshold_z)[0]
        
        if len(outlier_indices) > 0:
            super_weights[name] = {
                'count': len(outlier_indices),
                'indices': outlier_indices.tolist(),
                'values': weights[outlier_indices].tolist(),
                'z_scores': z_scores[outlier_indices].tolist(),
            }
            total_super_weights += len(outlier_indices)
    
    debug_print(f"Found {total_super_weights} super weights across {len(super_weights)} layers")
    
    return super_weights

def analyze_super_weight_impact(model, test_function, super_weights):
    """
    Analyze the impact of super weights on model performance.
    
    Args:
        model: PyTorch model to analyze
        test_function: Function that returns a performance metric
        super_weights: Dictionary of super weights by layer
        
    Returns:
        Dictionary of impact metrics
    """
    debug_print("Analyzing super weight impact on model performance")
    
    # Baseline performance
    baseline_performance = test_function(model)
    
    # Create a deep copy of the model for experiments
    import copy
    test_model = copy.deepcopy(model)
    
    # Impact analysis results
    impact_by_layer = {}
    
    # Analyze each layer with super weights
    for name, sw_info in super_weights.items():
        # Get the parameter to modify
        for param_name, param in test_model.named_parameters():
            if param_name == name:
                # Save original values
                original_values = param.data.clone()
                
                # Zero out super weights
                flat_param = param.data.view(-1)
                for idx in sw_info['indices']:
                    flat_param[idx] = 0.0
                
                # Test performance
                modified_performance = test_function(test_model)
                
                # Calculate impact
                impact = abs(baseline_performance - modified_performance) / baseline_performance
                
                # Store results
                impact_by_layer[name] = {
                    'baseline': baseline_performance,
                    'modified': modified_performance,
                    'impact': impact,
                    'super_weight_count': sw_info['count'],
                }
                
                # Restore original values
                param.data.copy_(original_values)
                break
    
    return impact_by_layer

def run_super_weight_analysis(model, test_function=None):
    """
    Run full super weight analysis on the model.
    
    Args:
        model: PyTorch model to analyze
        test_function: Function that returns a performance metric
        
    Returns:
        Dictionary of super weight analysis results
    """
    log_section("Super Weight Analysis")
    
    with Timer("Finding super weights"):
        super_weights = find_super_weights(model, threshold_z=2.5)
    
    # If test function is provided, analyze impact
    impact_metrics = {}
    if test_function is not None:
        with Timer("Analyzing super weight impact"):
            impact_metrics = analyze_super_weight_impact(model, test_function, super_weights)
    
    # Collect layer-wise statistics
    layer_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_stats[name] = compute_weight_statistics(param.data)
    
    return {
        'super_weights': super_weights,
        'impact_metrics': impact_metrics,
        'layer_stats': layer_stats,
    }