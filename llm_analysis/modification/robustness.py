"""
Robustness Testing Module

This module tests model robustness by applying controlled noise to weights.
"""

import os
import torch
import numpy as np

from ..core.utils import debug_print, Timer, log_section, ensure_dir, log_success
from ..config import MODEL_PATHS, EPSILON

def add_noise_to_weights(model, epsilon=EPSILON, distribution="gaussian", layer_scaling=None):
    """
    Add noise to model weights with optional layer-specific scaling.
    
    Args:
        model: PyTorch model to modify
        epsilon: Scale of noise to add
        distribution: Type of noise distribution ("gaussian", "uniform")
        layer_scaling: Dictionary mapping layer names to noise scaling factors
        
    Returns:
        model: Model with added noise
        stats: Dictionary with noise statistics
    """
    stats = {"total_params": 0, "noise_stats_by_layer": {}}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Count parameters
        param_size = param.numel()
        stats["total_params"] += param_size
        
        # Apply layer-specific scaling if provided
        local_epsilon = epsilon
        if layer_scaling and name in layer_scaling:
            local_epsilon *= layer_scaling[name]
        
        # Generate noise based on distribution
        if distribution == "gaussian":
            # Gaussian noise with mean 0 and std proportional to weight magnitude
            noise = torch.randn_like(param.data) * local_epsilon * param.data.std()
        elif distribution == "uniform":
            # Uniform noise between -epsilon and +epsilon times weight magnitude
            noise = (torch.rand_like(param.data) * 2 - 1) * local_epsilon * param.data.abs().mean()
        else:
            raise ValueError(f"Unsupported noise distribution: {distribution}")
        
        # Add noise to the parameter
        original_data = param.data.clone()
        param.data = param.data + noise
        
        # Calculate noise statistics
        with torch.no_grad():
            relative_change = (param.data - original_data).abs().mean() / original_data.abs().mean()
            
            stats["noise_stats_by_layer"][name] = {
                "epsilon": local_epsilon,
                "original_std": original_data.std().item(),
                "noise_std": noise.std().item(),
                "relative_change": relative_change.item(),
            }
    
    return model, stats

def apply_targeted_noise(model, target_layers=None, epsilon=EPSILON):
    """
    Apply noise only to specific layers or components.
    
    Args:
        model: PyTorch model to modify
        target_layers: List of layer names to target (None for all)
        epsilon: Scale of noise to add
        
    Returns:
        model: Model with added noise
        stats: Dictionary with noise statistics
    """
    stats = {"targeted_layers": target_layers if target_layers else "all"}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if target_layers and not any(layer in name for layer in target_layers):
            continue
        
        # Add noise to targeted layer
        noise = torch.randn_like(param.data) * epsilon * param.data.std()
        param.data = param.data + noise
    
    return model, stats

def apply_bit_flips(model, bit_flip_rate=0.0001):
    """
    Simulate random bit flips in model weights.
    
    Args:
        model: PyTorch model to modify
        bit_flip_rate: Probability of each bit flipping
        
    Returns:
        model: Model with bit flips
        stats: Dictionary with bit flip statistics
    """
    stats = {"total_bits_flipped": 0, "total_bits": 0}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Convert to byte representation for bit manipulation
        param_bytes = param.data.cpu().numpy().tobytes()
        bits_array = np.unpackbits(np.frombuffer(param_bytes, dtype=np.uint8))
        
        # Calculate number of bits and apply random flips
        num_bits = len(bits_array)
        stats["total_bits"] += num_bits
        
        # Create a mask of bits to flip
        flip_mask = np.random.random(num_bits) < bit_flip_rate
        bits_array[flip_mask] = 1 - bits_array[flip_mask]
        
        # Count flipped bits
        num_flipped = np.sum(flip_mask)
        stats["total_bits_flipped"] += num_flipped
        
        if num_flipped > 0:
            # Convert back to bytes and then to the tensor
            flipped_bytes = np.packbits(bits_array).tobytes()
            flipped_data = np.frombuffer(flipped_bytes, dtype=param.data.cpu().numpy().dtype)
            flipped_data = flipped_data.reshape(param.data.shape)
            param.data = torch.tensor(flipped_data, device=param.data.device)
    
    stats["flip_rate_achieved"] = stats["total_bits_flipped"] / stats["total_bits"]
    
    return model, stats

def apply_robustness_test(model_name, output_path, epsilon=EPSILON, distribution="gaussian"):
    """
    Load a model, apply noise, and save the modified model.
    
    Args:
        model_name: Name of the model to load
        output_path: Path to save the noisy model
        epsilon: Scale of noise to apply
        distribution: Type of noise distribution
        
    Returns:
        stats: Dictionary with noise statistics
    """
    log_section("Robustness Testing")
    
    # Load the model
    from transformers import AutoModelForCausalLM
    debug_print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    
    # Apply noise
    with Timer(f"Applying {distribution} noise with epsilon={epsilon}"):
        noisy_model, stats = add_noise_to_weights(model, epsilon, distribution)
    
    # Save the noisy model
    ensure_dir(os.path.dirname(output_path))
    debug_print(f"Saving noisy model to {output_path}")
    torch.save(noisy_model.state_dict(), output_path)
    
    # Log results
    avg_change = np.mean([s["relative_change"] for s in stats["noise_stats_by_layer"].values()])
    log_success(f"Applied noise with average relative change: {avg_change:.2%}")
    
    return stats