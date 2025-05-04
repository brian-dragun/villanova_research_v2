"""
Memory-optimized model loading utilities for ESA research.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple, Optional, Any, Union
from config import (
    MODEL_CONFIG,
    get_model_by_key, 
    get_model_paths, 
    is_model_cached
)

def load_model_optimized(
    model_name: str, 
    load_8bit: bool = True,
    device_map: str = "auto",
    gradient_checkpointing: bool = True,
) -> Tuple[Any, Any]:
    """
    Load a model with optimized memory settings.
    
    Args:
        model_name: Model name or key
        load_8bit: Whether to use 8-bit quantization
        device_map: How to distribute model across devices
        gradient_checkpointing: Whether to enable gradient checkpointing
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Convert key to full name if needed
    model_name = get_model_by_key(model_name)
    
    # Check if model is cached
    is_cached, cache_type, cache_path = is_model_cached(model_name)
    
    print(f"⏳ Loading model: {model_name}")
    
    # Set up model configuration
    model_kwargs = MODEL_CONFIG.copy()
    
    # Check if bitsandbytes is available for quantization
    use_quantization = load_8bit
    if use_quantization:
        try:
            import bitsandbytes
            print(f"✅ Using bitsandbytes version: {bitsandbytes.__version__}")
        except ImportError:
            print("⚠️ bitsandbytes not available, disabling 8-bit quantization")
            use_quantization = False
    
    # Configure quantization settings properly
    if "load_in_8bit" in model_kwargs:
        model_kwargs.pop("load_in_8bit")
    
    # Only use quantization if explicitly requested and available
    if use_quantization:
        try:
            # Try using the newer BitsAndBytesConfig approach
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True, 
                llm_int8_threshold=6.0
            )
            print("✅ Using BitsAndBytesConfig for 8-bit quantization")
        except ImportError:
            # Fall back to older method
            model_kwargs["load_in_8bit"] = True
            print("⚠️ Using legacy 8-bit quantization")
    
    # Set device mapping
    model_kwargs["device_map"] = device_map
    
    # Remove any problematic settings that might cause errors
    problematic_keys = ["use_flash_attention", "use_memory_efficient_attention"]
    for key in problematic_keys:
        if key in model_kwargs:
            model_kwargs.pop(key)
    
    # Load tokenizer first (faster)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        print(f"⚠️ Error loading tokenizer: {e}")
        print("⚠️ Trying with minimal configuration...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load the model with memory optimizations
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    except Exception as e:
        print(f"⚠️ Error loading model with optimizations: {e}")
        print("⚠️ Trying with minimal configuration...")
        # Try with minimal settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto"
        )
    
    # Apply additional memory optimizations
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Report memory usage
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        max_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"🧠 GPU Memory: {allocated_memory:.2f} GB / {max_memory:.2f} GB")
    
    return model, tokenizer

def get_layerwise_parameter_count(model: torch.nn.Module) -> Dict[str, int]:
    """
    Get parameter counts for each layer in the model.
    
    Args:
        model: The PyTorch model
    
    Returns:
        Dictionary mapping layer names to parameter counts
    """
    result = {}
    
    # First pass: collect all parameters by layer name
    for name, param in model.named_parameters():
        # Extract layer name - depends on model architecture
        if "layers." in name:
            # Format like "model.layers.0.self_attn.q_proj.weight"
            parts = name.split(".")
            try:
                layer_idx = int(parts[2])
                layer_type = parts[3]  # self_attn, mlp, etc.
                layer_name = f"layer_{layer_idx}_{layer_type}"
            except (IndexError, ValueError):
                layer_name = name.split(".")[0]
        else:
            # Just use first component of name
            layer_name = name.split(".")[0]
            
        # Count parameters
        param_count = param.numel()
        
        if layer_name in result:
            result[layer_name] += param_count
        else:
            result[layer_name] = param_count
    
    return result

def estimate_memory_usage(model_name: str) -> Dict[str, float]:
    """
    Estimate memory requirements for loading and using a model.
    
    Args:
        model_name: Name or key of the model
    
    Returns:
        Dictionary with memory estimates in GB
    """
    # Convert key to full name if needed
    model_name = get_model_by_key(model_name)
    
    # Rough estimates based on parameter counts for common models
    param_counts = {
        "gpt-neo-125m": 125e6,
        "gpt-neo-1.3B": 1.3e9,
        "gpt-j-6B": 6e9,
        "llama-2-7b": 7e9,
        "mistral-7B": 7e9,
        "falcon-rw-1b": 1e9,
        "opt-350m": 350e6,
        "opt-1.3b": 1.3e9,
        "bloom-560m": 560e6,
        "bloom-1b1": 1.1e9,
    }
    
    # Get parameter count, with fallback to rough estimate
    param_count = None
    for key, count in param_counts.items():
        if key in model_name.lower():
            param_count = count
            break
    
    if param_count is None:
        # Fallback estimate based on name
        if "7b" in model_name.lower():
            param_count = 7e9
        elif "1.3" in model_name.lower():
            param_count = 1.3e9
        elif "125m" in model_name.lower():
            param_count = 125e6
        else:
            # Conservative default
            param_count = 1e9
    
    # Memory estimates in GB
    # - FP32: 4 bytes per parameter
    # - FP16: 2 bytes per parameter
    # - INT8: 1 byte per parameter
    # - Activation memory: ~1.2x parameter memory for transformer models
    # - Optimizer states: ~2x parameter memory for Adam-like optimizers
    fp32_model_size = param_count * 4 / (1024**3)
    fp16_model_size = param_count * 2 / (1024**3)
    int8_model_size = param_count / (1024**3)
    
    return {
        "parameters": param_count,
        "fp32_model_size_gb": fp32_model_size,
        "fp16_model_size_gb": fp16_model_size,
        "int8_model_size_gb": int8_model_size,
        "activation_memory_fp16_gb": fp16_model_size * 1.2,
        "optimizer_states_fp32_gb": fp32_model_size * 2,
        "total_training_memory_fp16_gb": fp16_model_size + (fp16_model_size * 1.2) + (fp32_model_size * 2),
        "total_inference_memory_int8_gb": int8_model_size + (int8_model_size * 0.6),
    }

def print_model_memory_requirements(model_name: str) -> None:
    """
    Print memory requirements for a model in a nice format.
    
    Args:
        model_name: Name or key of model
    """
    model_name = get_model_by_key(model_name)
    memory = estimate_memory_usage(model_name)
    
    print(f"\n📊 Memory Requirements for {model_name}")
    print(f"  Parameters: {memory['parameters']:,.0f}")
    print(f"  Model Size (FP32): {memory['fp32_model_size_gb']:.2f} GB")
    print(f"  Model Size (FP16): {memory['fp16_model_size_gb']:.2f} GB")
    print(f"  Model Size (INT8): {memory['int8_model_size_gb']:.2f} GB")
    print(f"  Inference Memory (INT8): {memory['total_inference_memory_int8_gb']:.2f} GB")
    print(f"  Training Memory (FP16): {memory['total_training_memory_fp16_gb']:.2f} GB")
    
    # Add warning if the model is likely too large
    available_memory = 0
    if torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  Available GPU Memory: {available_memory:.2f} GB")
        
        if memory['total_inference_memory_int8_gb'] > available_memory * 0.9:
            print(f"\n⚠️  WARNING: This model likely won't fit in GPU memory for inference.")
            print(f"   Consider using 8-bit quantization or CPU offloading.")
        
        if memory['total_training_memory_fp16_gb'] > available_memory * 0.9:
            print(f"\n⚠️  WARNING: This model won't fit in GPU memory for training.")
            print(f"   Consider gradient checkpointing, 8-bit optimizers, or smaller models.")
    
    # Add recommendation
    if memory['parameters'] > 1e9:  # > 1B params
        print(f"\n💡 Recommendation: Use 8-bit quantization and gradient checkpointing for efficiency.")