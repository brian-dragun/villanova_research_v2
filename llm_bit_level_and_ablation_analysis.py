import torch
import numpy as np
import math
import torch.nn.functional as F
import torch.autograd as autograd
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, TEST_PROMPT

# --- Helper Function to Flip a Bit ---
def flip_bit(tensor, bit_position):
    """
    Flip the bit at the given position for a float32 tensor.
    The tensor is assumed to be 1D.
    """
    # Convert tensor to a numpy array as int32 view
    np_tensor = tensor.detach().cpu().numpy()
    int_tensor = np_tensor.view(np.int32)
    # Flip the specified bit using XOR:
    flipped_int_tensor = int_tensor ^ (1 << bit_position)
    # Convert back to float32:
    flipped_np_tensor = flipped_int_tensor.view(np.float32)
    return torch.tensor(flipped_np_tensor, device=tensor.device)

# --- Bit-level Sensitivity Analysis Functions ---

def bit_sensitivity_analysis_for_param(model, inputs, loss_fn, param, element_index=0, bit_positions=range(24, 32)):
    """
    For a given parameter (flattened) and a chosen element (by index),
    flip bits at positions specified in bit_positions and record the change in loss.
    Returns a dictionary mapping bit positions to the loss change.
    """
    # Use param.data to avoid modifying a leaf tensor that requires grad.
    flat_param = param.data.view(-1)
    # Get the original value of the chosen element
    original_value = flat_param[element_index].clone()
    
    # Get baseline loss
    outputs = model(**inputs)
    baseline_loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), inputs["input_ids"].view(-1))
    
    sensitivity = {}
    for bit in bit_positions:
        # Flip the bit for this element (unsqueeze to 1D)
        perturbed_value = flip_bit(original_value.unsqueeze(0), bit)[0]
        # Replace the chosen element in a no_grad block
        with torch.no_grad():
            backup = original_value.clone()
            flat_param[element_index] = perturbed_value
        outputs_perturbed = model(**inputs)
        perturbed_loss = loss_fn(outputs_perturbed.logits.view(-1, outputs_perturbed.logits.size(-1)),
                                  inputs["input_ids"].view(-1))
        sensitivity[bit] = (perturbed_loss - baseline_loss).item()
        # Restore the original value in a no_grad block
        with torch.no_grad():
            flat_param[element_index] = backup
    return sensitivity

# --- Ablation Study Functions ---

def evaluate_model(model, tokenizer, prompt=TEST_PROMPT):
    """
    Evaluate the model on a prompt and return the perplexity.
    (This is a rough metric for demonstration.)
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    outputs = model(**inputs, labels=inputs.input_ids)
    loss = outputs.loss.item()
    perplexity = math.exp(loss)
    return perplexity

def ablation_study_param(model, evaluate_fn, param, prune_fraction=0.01):
    """
    Ablate (zero out) a fraction of the most sensitive weights in the parameter.
    Here we use absolute value as a simple proxy.
    Returns the quality metric (e.g., perplexity) after ablation.
    """
    flat_param = param.data.view(-1)
    k = int(prune_fraction * flat_param.numel())
    if k == 0:
        return None
    backup = flat_param.clone()
    # Get indices of the top-k weights by absolute value
    indices = torch.topk(flat_param.abs(), k, largest=True).indices
    with torch.no_grad():
        flat_param[indices] = 0.0
    quality = evaluate_fn(model)
    # Restore original parameter values
    with torch.no_grad():
        flat_param.copy_(backup)
    return quality

# --- Main Demonstration ---

def run_bit_level_and_ablation_analysis(prompt=TEST_PROMPT):
    print("Running bit-level sensitivity analysis and ablation study with prompt:")
    print(f"  {prompt}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Select a parameter for analysis, e.g., input embeddings
    param = model.get_input_embeddings().weight  # shape: [vocab_size, hidden_size]
    element_index = 0  # Representative element
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    sensitivity = bit_sensitivity_analysis_for_param(model, inputs, loss_fn, param, element_index)
    
    print("\nBit-level sensitivity for parameter element at index", element_index)
    for bit, delta_loss in sensitivity.items():
        print(f"  Bit position {bit}: Loss change = {delta_loss:.6f}")
    
    # Ablation Study
    eval_fn = lambda m: evaluate_model(m, tokenizer, prompt=prompt)
    baseline_quality = eval_fn(model)
    print("\nBaseline model perplexity:", baseline_quality)
    
    prune_fraction = 0.01
    quality_after_ablation = ablation_study_param(model, eval_fn, param, prune_fraction)
    print(f"Model perplexity after ablating top {prune_fraction*100:.1f}% of sensitive weights: {quality_after_ablation}")

def main():
    run_bit_level_and_ablation_analysis()

if __name__ == "__main__":
    main()
