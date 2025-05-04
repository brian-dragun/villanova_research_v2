"""
Bit-level Analysis Module

This module implements bit-level sensitivity analysis and bit flip simulations.
"""

import torch
import numpy as np
import struct
import matplotlib.pyplot as plt
from copy import deepcopy

from ..core.utils import debug_print, Timer, log_section, ensure_dir
from ..config import MODEL_NAME, ANALYSIS_CONFIG, OUTPUT_DIR

def float_to_binary(value):
    """
    Convert a float to its binary representation.
    
    Args:
        value: Float value
        
    Returns:
        Binary string representation
    """
    # Get IEEE 754 binary representation
    binary = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', value))
    return binary

def binary_to_float(binary):
    """
    Convert a binary string to float.
    
    Args:
        binary: Binary string representation
        
    Returns:
        Float value
    """
    # Convert binary string to bytes and then to float
    byte_data = int(binary, 2).to_bytes(4, byteorder='big')
    return struct.unpack('!f', byte_data)[0]

def flip_bit(value, position):
    """
    Flip a specific bit in the binary representation of a float.
    
    Args:
        value: Float value
        position: Bit position to flip (0-31)
        
    Returns:
        Float with bit flipped
    """
    binary = float_to_binary(value)
    
    # Flip the bit at the specified position
    bit_list = list(binary)
    bit_list[position] = '1' if bit_list[position] == '0' else '0'
    flipped_binary = ''.join(bit_list)
    
    return binary_to_float(flipped_binary)

def analyze_bit_sensitivity(model, test_function, bit_flips_per_layer=ANALYSIS_CONFIG["bit_flips_per_layer"]):
    """
    Analyze how bit flips affect model performance.
    
    Args:
        model: PyTorch model to analyze
        test_function: Function to evaluate model performance
        bit_flips_per_layer: Number of bit flips to try per layer
        
    Returns:
        Dictionary of bit sensitivity results
    """
    log_section("Bit-level Sensitivity Analysis")
    
    # Dictionary to store results
    bit_sensitivity = {}
    
    # Get baseline performance
    baseline_perf = test_function(model)
    debug_print(f"Baseline performance: {baseline_perf:.4f}")
    
    # Create a deep copy of the model for experiments
    test_model = deepcopy(model)
    
    # Dictionary to track most sensitive bits
    most_sensitive_bits = []
    
    # Iterate through layers
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        debug_print(f"Analyzing parameter: {name}")
        bit_sensitivity[name] = {
            'exponent_impact': 0.0,
            'sign_impact': 0.0,
            'mantissa_impact': 0.0,
            'most_sensitive_position': None,
            'worst_impact': 0.0,
        }
        
        # Get parameter to modify in test model
        test_param = None
        for test_name, test_p in test_model.named_parameters():
            if test_name == name:
                test_param = test_p
                break
        
        if test_param is None:
            continue
        
        # Sample weights to test
        num_weights = param.numel()
        if bit_flips_per_layer < num_weights:
            indices = torch.randperm(num_weights)[:bit_flips_per_layer]
        else:
            indices = torch.arange(num_weights)
        
        # Test each sampled weight
        for idx in indices:
            flat_idx = idx.item()
            flat_param = test_param.data.view(-1)
            original_value = flat_param[flat_idx].item()
            
            # Test each bit position
            for bit_pos in range(32):
                # Copy original value
                flat_param[flat_idx] = original_value
                
                # Flip bit
                flipped_value = flip_bit(original_value, bit_pos)
                flat_param[flat_idx] = flipped_value
                
                # Test performance
                try:
                    perf = test_function(test_model)
                    
                    # Calculate impact
                    impact = abs(baseline_perf - perf) / baseline_perf
                    
                    # Track results by bit type
                    if bit_pos == 0:  # Sign bit
                        bit_sensitivity[name]['sign_impact'] += impact
                    elif 1 <= bit_pos <= 8:  # Exponent bits
                        bit_sensitivity[name]['exponent_impact'] += impact
                    else:  # Mantissa bits
                        bit_sensitivity[name]['mantissa_impact'] += impact
                    
                    # Track most impactful bit
                    if impact > bit_sensitivity[name]['worst_impact']:
                        bit_sensitivity[name]['worst_impact'] = impact
                        bit_sensitivity[name]['most_sensitive_position'] = bit_pos
                        
                        most_sensitive_bits.append({
                            'layer': name,
                            'weight_idx': flat_idx,
                            'bit_position': bit_pos,
                            'impact': impact,
                            'original_value': original_value,
                            'flipped_value': flipped_value
                        })
                except Exception as e:
                    debug_print(f"Error testing bit flip: {e}")
                
                # Restore original value
                flat_param[flat_idx] = original_value
        
        # Normalize by number of tests
        tests_per_type = bit_flips_per_layer
        if tests_per_type > 0:
            bit_sensitivity[name]['sign_impact'] /= tests_per_type
            bit_sensitivity[name]['exponent_impact'] /= tests_per_type
            bit_sensitivity[name]['mantissa_impact'] /= tests_per_type
    
    # Sort sensitive bits by impact
    most_sensitive_bits.sort(key=lambda x: x['impact'], reverse=True)
    
    return {
        'bit_sensitivity_by_layer': bit_sensitivity,
        'most_sensitive_bits': most_sensitive_bits[:10],  # Top 10
        'baseline_performance': baseline_perf
    }

def visualize_bit_sensitivity(bit_sensitivity_results, output_dir=OUTPUT_DIR):
    """
    Visualize bit sensitivity results.
    
    Args:
        bit_sensitivity_results: Results from analyze_bit_sensitivity
        output_dir: Directory to save visualizations
    """
    ensure_dir(output_dir)
    
    bit_sensitivity = bit_sensitivity_results['bit_sensitivity_by_layer']
    
    # Prepare data for plotting
    layer_names = []
    sign_impacts = []
    exponent_impacts = []
    mantissa_impacts = []
    worst_impacts = []
    
    for name, metrics in bit_sensitivity.items():
        layer_names.append(name.split('.')[-2])
        sign_impacts.append(metrics['sign_impact'])
        exponent_impacts.append(metrics['exponent_impact'])
        mantissa_impacts.append(metrics['mantissa_impact'])
        worst_impacts.append(metrics['worst_impact'])
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(layer_names))
    width = 0.2
    
    ax.bar(x - width*1.5, sign_impacts, width, label='Sign Bit')
    ax.bar(x - width/2, exponent_impacts, width, label='Exponent Bits')
    ax.bar(x + width/2, mantissa_impacts, width, label='Mantissa Bits')
    ax.bar(x + width*1.5, worst_impacts, width, label='Worst Case')
    
    ax.set_ylabel('Normalized Impact')
    ax.set_title('Bit Sensitivity by Layer and Bit Type')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=90)
    ax.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'bit_sensitivity.png')
    plt.savefig(output_path)
    plt.close()
    
    # Create visualization of most sensitive bits
    most_sensitive = bit_sensitivity_results['most_sensitive_bits']
    
    impacts = [item['impact'] for item in most_sensitive]
    positions = [f"{item['layer'].split('.')[-2]}:{item['bit_position']}" for item in most_sensitive]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(positions, impacts)
    ax.set_ylabel('Performance Impact')
    ax.set_title('Most Sensitive Bits')
    ax.set_xticklabels(positions, rotation=90)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'most_sensitive_bits.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Bit sensitivity visualizations saved to {output_dir}")

def ablation_analysis(model, test_function, target_layers=None):
    """
    Perform ablation study by zeroing out different components.
    
    Args:
        model: PyTorch model to analyze
        test_function: Function to evaluate model performance
        target_layers: List of layer names to target (None for all)
        
    Returns:
        Dictionary of ablation results
    """
    log_section("Weight Ablation Analysis")
    
    # Get baseline performance
    baseline_perf = test_function(model)
    debug_print(f"Baseline performance: {baseline_perf:.4f}")
    
    # Deep copy for testing
    test_model = deepcopy(model)
    
    ablation_results = {}
    
    # Iterate through layers
    for name, param in test_model.named_parameters():
        if not param.requires_grad:
            continue
            
        if target_layers and not any(layer in name for layer in target_layers):
            continue
        
        # Skip bias terms for this analysis
        if 'bias' in name:
            continue
            
        debug_print(f"Ablating {name}")
        
        # Save original data
        original_data = param.data.clone()
        
        # Zero out the parameter
        param.data.zero_()
        
        try:
            # Test performance
            ablated_perf = test_function(test_model)
            
            # Calculate impact
            impact = abs(baseline_perf - ablated_perf) / baseline_perf
            
            ablation_results[name] = {
                'baseline': baseline_perf,
                'ablated': ablated_perf,
                'impact': impact
            }
        except Exception as e:
            debug_print(f"Error during ablation test: {e}")
            ablation_results[name] = {'error': str(e)}
        
        # Restore original data
        param.data.copy_(original_data)
    
    return {
        'baseline_performance': baseline_perf,
        'ablation_by_layer': ablation_results
    }

def run_bit_level_and_ablation_analysis(model=None, prompt=None):
    """
    Run both bit-level analysis and ablation studies.
    
    Args:
        model: PyTorch model to analyze (loads from MODEL_NAME if None)
        prompt: Text prompt for testing
        
    Returns:
        Dictionary with combined results
    """
    # Load model if not provided
    if model is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Create a simple test function (dummy implementation)
    def test_function(test_model):
        # In a real scenario, this would evaluate model performance
        # For now, return a random score for demonstration
        return np.random.random() * 0.1 + 0.9
    
    # Run bit sensitivity analysis
    bit_results = analyze_bit_sensitivity(model, test_function)
    
    # Run ablation analysis
    ablation_results = ablation_analysis(model, test_function)
    
    # Visualize results
    visualize_bit_sensitivity(bit_results)
    
    return {
        'bit_sensitivity': bit_results,
        'ablation': ablation_results
    }