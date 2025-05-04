"""
Configuration Module for LLM Analysis

This module contains all configuration settings for the LLM analysis pipeline.
"""

import os
from pathlib import Path

# Base directory setup
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Base model from HuggingFace

# Paths for model variants
MODEL_PATHS = {
    "finetuned": os.path.join(DATA_DIR, "gpu_llm_finetuned_llama27bhf"),
    "pruned": os.path.join(DATA_DIR, "gpu_llm_pruned_llama27bhf.pth"),
    "noisy": os.path.join(DATA_DIR, "gpu_llm_noisy_llama27bhf.pth"),
}

# Epsilon value for noise testing
EPSILON = 0.05

# Test prompt for model evaluation
TEST_PROMPT = "What movie is 'In a galaxy far, far away' from?"

# Training parameters
TRAIN_CONFIG = {
    "epochs": 3,
    "learning_rate": 2e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 512,
}

# Analysis parameters
ANALYSIS_CONFIG = {
    "pruning_threshold": 0.1,  # Threshold for weight pruning
    "subsample_size": 1000,     # Subsample size for sensitivity analysis
    "sensitivity_iterations": 5, # Number of iterations for sensitivity analysis
    "bit_flips_per_layer": 10,  # Number of bit flips per layer in bit-level analysis
}

# Evaluation parameters
EVAL_CONFIG = {
    "dataset": "wikitext-2-raw-v1",
    "split": "validation",
    "batch_size": 4,
    "max_seq_length": 128,
}

# Adversarial testing configuration
ADVERSARIAL_CONFIG = {
    "attack_methods": ["fgsm", "pgd"],
    "epsilon_range": [0.01, 0.05, 0.1],
    "pgd_steps": 5,
    "pgd_alpha": 0.01,
}

# Visualization settings
VISUALIZATION_CONFIG = {
    "colormap": "viridis",
    "dpi": 300,
    "figsize": (12, 8),
}