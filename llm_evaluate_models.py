import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math
import time
from config import MODEL_NAME
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the checkpoint file in the data folder.
CHECKPOINT_FILE = os.path.join("data", "eval_checkpoint.txt")
SAVE_EVERY = 100  # Save checkpoint every 100 examples.
MAX_LENGTH = 128  # Maximum sequence length for evaluation

def evaluate_model(model_path, dataset_split="validation", batch_size=4):
    """
    Evaluates a language model by computing its perplexity on a dataset.
    
    Args:
        model_path: Path to the model directory or weights file
        dataset_split: Which split of the dataset to use ("validation" or "test")
        batch_size: Batch size for evaluation (increase for faster eval if memory allows)
        
    Returns:
        Perplexity score (lower is better)
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating model at {model_path} on {device}")
    
    # Load model with error handling
    try:
        if os.path.isdir(model_path):
            logger.info(f"Loading model from directory: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        else:
            logger.info(f"Loading base model {MODEL_NAME} and weights from: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
            # Load with error handling for potentially different state dict formats
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
            except Exception as e:
                logger.warning(f"Standard loading failed: {e}. Trying alternative loading method...")
                # Sometimes models are saved as just the state dict without 'model_state_dict' key
                state_dict = torch.load(model_path, map_location=device)
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                else:
                    model.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    model.to(device)
    model.eval()
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise
    
    # Load dataset
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=dataset_split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    total_loss = 0.0
    total_tokens = 0
    
    # Determine starting index from checkpoint (if exists)
    start_index = 0
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                start_index = int(f.read().strip())
            logger.info(f"Resuming evaluation from index {start_index}")
        except Exception as e:
            logger.warning(f"Could not read checkpoint file, starting from index 0: {e}")
    
    # Pre-process the dataset to remove empty examples and tokenize
    valid_examples = []
    for i, example in enumerate(dataset):
        if i < start_index:
            continue
            
        text = example["text"].strip()
        if not text:  # Skip empty examples
            continue
            
        valid_examples.append(text)
    
    # Process in batches for better efficiency
    logger.info(f"Processing {len(valid_examples)} examples in batches of {batch_size}")
    
    # Create batches
    batches = [valid_examples[i:i + batch_size] for i in range(0, len(valid_examples), batch_size)]
    
    with torch.no_grad():  # Disable gradient calculation for inference
        for batch_idx, text_batch in enumerate(tqdm(batches, desc="Evaluating")):
            current_idx = start_index + (batch_idx * batch_size)
            
            # Tokenize batch
            inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
            
            if inputs["input_ids"].size(1) == 0:
                continue
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            
            # Count tokens (excluding padding)
            num_tokens = inputs["input_ids"].ne(tokenizer.pad_token_id).sum().item()
            total_loss += loss * num_tokens
            total_tokens += num_tokens
            
            # Save checkpoint every SAVE_EVERY batches
            if (batch_idx + 1) % SAVE_EVERY == 0:
                checkpoint_idx = current_idx + len(text_batch)
                with open(CHECKPOINT_FILE, "w") as f:
                    f.write(str(checkpoint_idx))
                    
                # Log progress
                elapsed = time.time() - start_time
                logger.info(f"Processed {checkpoint_idx} examples in {elapsed:.2f}s. Current avg loss: {total_loss/total_tokens:.4f}")
    
    # Remove checkpoint file after evaluation is complete
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    if total_tokens == 0:
        logger.error("No valid tokens found in dataset for evaluation.")
        return None

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.2f}s")
    print(f"✅ Perplexity of model at {model_path}: {perplexity:.2f}")
    return perplexity

if __name__ == "__main__":
    # Get path from command line if provided
    import sys
    if len(sys.argv) > 1:
        model_paths = sys.argv[1:]
        for path in model_paths:
            evaluate_model(path)
    else:
        evaluate_model("data/gpu_llm_finetuned_llama27bhf")  # Loads from directory
        evaluate_model("data/gpu_llm_pruned_llama27bhf.pth")  # Loads from a file
        evaluate_model("data/gpu_llm_noisy_llama27bhf.pth")  # Loads from a file
