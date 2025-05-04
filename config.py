# config.py
import os
from pathlib import Path

# Base directory setup - ensure we're using current working directory
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "model_cache"  # Local cache for models

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Load Hugging Face token from .env file if available
HF_TOKEN = None
env_path = BASE_DIR / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.startswith("HUGGINGFACE_TOKEN="):
                HF_TOKEN = line.strip().split("=", 1)[1]
                break

# Available models configuration
AVAILABLE_MODELS = {
    # Llama models
    "llama-2-7b": {
        "name": "meta-llama/Llama-2-7b-hf",
        "requires_auth": True,
        "details": "7B parameter base Llama model"
    },
    "llama-2-7b-chat": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "requires_auth": True,
        "details": "7B parameter chat-tuned Llama model"
    },
    
    # GPT-Neo/GPT-J models
    "gpt-neo-125m": {
        "name": "EleutherAI/gpt-neo-125m",
        "requires_auth": False,
        "details": "125M parameter model, good for quick testing"
    },
    "gpt-neo-1.3b": {
        "name": "EleutherAI/gpt-neo-1.3B",
        "requires_auth": False,
        "details": "1.3B parameter model"
    },
    "gpt-j-6b": {
        "name": "EleutherAI/gpt-j-6B",
        "requires_auth": False,
        "details": "6B parameter model"
    },
    
    # OPT models
    "opt-350m": {
        "name": "facebook/opt-350m",
        "requires_auth": False,
        "details": "350M parameter OPT model"
    },
    "opt-1.3b": {
        "name": "facebook/opt-1.3b",
        "requires_auth": False,
        "details": "1.3B parameter OPT model"
    },
    
    # BLOOM models
    "bloom-560m": {
        "name": "bigscience/bloom-560m",
        "requires_auth": False,
        "details": "560M parameter BLOOM model"
    },
    "bloom-1b1": {
        "name": "bigscience/bloom-1b1",
        "requires_auth": False,
        "details": "1.1B parameter BLOOM model"
    },
    
    # Pythia models
    "pythia-70m": {
        "name": "EleutherAI/pythia-70m",
        "requires_auth": False,
        "details": "70M parameter model, very small"
    },
    "pythia-410m": {
        "name": "EleutherAI/pythia-410m",
        "requires_auth": False,
        "details": "410M parameter model"
    },
    
    # Falcon models
    "falcon-rw-1b": {
        "name": "tiiuae/falcon-rw-1b",
        "requires_auth": False,
        "details": "1B parameter Falcon model"
    },
    
    # Mistral models
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-v0.1",
        "requires_auth": False,
        "details": "7B parameter Mistral model"
    }
}

# Default model selection
DEFAULT_MODEL_KEY = "llama-2-7b"  # Changed default to Llama
MODEL_NAME = AVAILABLE_MODELS[DEFAULT_MODEL_KEY]["name"]

# Function to get model name from key
def get_model_by_key(model_key):
    """Get full model name from short key."""
    if model_key in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_key]["name"]
    return model_key  # Return as-is if not found (assuming it's a full path)

# Function to generate model paths based on model name
def get_model_paths(model_name):
    """Generate model paths based on model name."""
    # Extract a simplified name for directories
    if "/" in model_name:
        simple_name = model_name.split("/")[-1].lower()
    else:
        simple_name = model_name.lower()
    
    return {
        "finetuned": os.path.join(DATA_DIR, f"gpu_llm_finetuned_{simple_name}"),
        "pruned": os.path.join(DATA_DIR, f"gpu_llm_pruned_{simple_name}.pth"),
        "noisy": os.path.join(DATA_DIR, f"gpu_llm_noisy_{simple_name}.pth"),
        "cached": os.path.join(CACHE_DIR, simple_name),
    }

# Paths for model variants - using the default model initially
MODEL_PATHS = get_model_paths(MODEL_NAME)

# Model loading configuration
MODEL_CONFIG = {
    "use_auth_token": HF_TOKEN,
    "trust_remote_code": True,
    "device_map": "auto",  # Automatically distribute model across available GPUs
    "load_in_8bit": False,  # Set to True to load in 8-bit precision (requires bitsandbytes)
    "cache_dir": CACHE_DIR,  # Use local cache
}

# Function to check if model is already downloaded/cached
def is_model_cached(model_name):
    """Check if a model is already downloaded and cached locally."""
    paths = get_model_paths(model_name)
    # Check if finetuned model exists
    if os.path.exists(paths["finetuned"]):
        return True, "finetuned", paths["finetuned"]
    # Check if pruned model exists
    if os.path.exists(paths["pruned"]):
        return True, "pruned", paths["pruned"]
    # Check if in model cache
    if os.path.exists(paths["cached"]):
        return True, "cached", paths["cached"]
    return False, None, None

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

# Visualization configuration
VISUALIZATION_CONFIG = {
    "figsize": (12, 8),
    "dpi": 100,
    "style": "seaborn-v0_8-whitegrid"
}

# Flag to determine if we should try to look for saved models in parent dirs
USE_PARENT_DATA_DIRECTORIES = False

# If enabled, also look in parent directory's data folder for models
if USE_PARENT_DATA_DIRECTORIES:
    PARENT_DIR = Path(os.path.dirname(BASE_DIR))
    PARENT_DATA_DIR = PARENT_DIR / "villanova_research" / "data"
    if os.path.exists(PARENT_DATA_DIR):
        print(f"Note: Will also check {PARENT_DATA_DIR} for saved models if not found locally")
else:
    PARENT_DATA_DIR = None

