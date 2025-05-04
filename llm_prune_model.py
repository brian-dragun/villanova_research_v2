import torch
from transformers import AutoModelForCausalLM
from llm_super_weights import compute_weight_statistics
from config import MODEL_NAME

def compute_sensitivity(model):
    """Compute a simple sensitivity score (mean absolute weight) for each parameter."""
    sensitivity_scores = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            sensitivity_scores[name] = param.abs().mean().item()
    return sensitivity_scores

def prune_layerwise(model, sensitivity_scores, prune_ratio=0.05):
    """Prune model weights in-place based on a sensitivity threshold."""
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad and name in sensitivity_scores:
            threshold = prune_ratio * sensitivity_scores[name]
            mask = param.abs() >= threshold
            param.data.mul_(mask)
    return model

def prune_model(model_dir, pruned_model_path, prune_ratio=0.05):
    """
    Loads the entire directory with from_pretrained(...),
    prunes in-place, and saves the state dict to pruned_model_path.
    """
    # 1) Load from directory containing model.safetensors
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    model.eval()
    
    # 2) Compute sensitivity and prune
    sensitivity_scores = compute_sensitivity(model)
    print("Computed sensitivity scores for pruning.")
    model = prune_layerwise(model, sensitivity_scores, prune_ratio)
    
    # 3) Save pruned state dict
    torch.save(model.state_dict(), pruned_model_path)
    print(f"✅ Pruned model saved as {pruned_model_path}")

if __name__ == "__main__":
    prune_model("data/llm_finetuned", "data/llm_pruned.pth")
