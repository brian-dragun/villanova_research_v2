import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoModelForCausalLM
from config import MODEL_NAME

def compute_weight_statistics(model):
    stats = {}
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            weights = param.detach().cpu().numpy().flatten()
            stats[name] = {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'max': np.max(np.abs(weights))
            }
    return stats

def identify_super_weights(model, z_threshold=2.5):
    stats = compute_weight_statistics(model)
    super_weights = {}
    layer_summary = defaultdict(int)

    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            weights = param.detach().cpu().numpy().flatten()
            mean_val = stats[name]['mean']
            std_val = stats[name]['std'] + 1e-8  # avoid division by zero
            z_scores = np.abs((weights - mean_val) / std_val)
            indices = np.where(z_scores > z_threshold)[0]
            if len(indices) > 0:
                super_weights[name] = indices
                layer = name.split(".")[2] if "layers" in name else name.split(".")[0]
                layer_summary[layer] += len(indices)

    return super_weights, layer_summary

def plot_layerwise_superweights(layer_summary):
    if not layer_summary:
        print("No super weights found to plot.")
        return

    layers = list(layer_summary.keys())
    counts = [layer_summary[layer] for layer in layers]

    plt.figure(figsize=(12, 6))
    plt.bar(layers, counts)
    plt.xlabel("Layer")
    plt.ylabel("Super Weight Count")
    plt.title("Layer-wise Super Weight Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("layerwise_superweights.png")
    plt.close()
    print("✅ Saved plot: layerwise_superweights.png")

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    super_weights, layer_summary = identify_super_weights(model, z_threshold=2.5)

    print("Identified super weights (Z-score > 2.5):")
    for layer, indices in super_weights.items():
        print(f"Layer: {layer}, Count: {len(indices)}")

    print("\nLayer-wise Super Weight Summary:")
    for layer, count in layer_summary.items():
        print(f"{layer}: {count} super weights")

    plot_layerwise_superweights(layer_summary)
