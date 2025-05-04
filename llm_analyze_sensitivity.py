import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_NAME, TEST_PROMPT
from llm_super_weights import (
    compute_gradient_sensitivity,
    compute_hessian_sensitivity as compute_full_hessian_sensitivity,
    compute_integrated_gradients,
    identify_super_weights,
    visualize_sensitivity_maps
)

ssize = 10

def compute_hessian_diagonal_subset(loss_fn, param_data, subset_size=ssize, epsilon=1e-5):
    """
    Compute the diagonal of the Hessian for only a subset of parameters
    using a finite difference approximation.
    """
    flat_param = param_data.view(-1)
    diag = torch.zeros(subset_size, device=flat_param.device)
    base_loss = loss_fn(flat_param)
    for i in range(subset_size):
        orig = flat_param[i].item()
        flat_param[i] = orig + epsilon
        loss_plus = loss_fn(flat_param)
        flat_param[i] = orig - epsilon
        loss_minus = loss_fn(flat_param)
        flat_param[i] = orig  # Restore original value
        diag[i] = (loss_plus - 2 * base_loss + loss_minus) / (epsilon ** 2)
    return diag

def compute_hessian_sensitivity(model, input_text, device=torch.device("cpu"), subset_size=ssize):
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Focus on a smaller parameter: lm_head.weight
    if not hasattr(model, "lm_head"):
        raise ValueError("Model does not have an 'lm_head' attribute.")
    param = model.lm_head.weight
    param_data = param.data.clone().detach().requires_grad_(True)

    def loss_fn(param_vec):
        reshaped = param_vec.view_as(param)
        with torch.no_grad():
            model.lm_head.weight.copy_(reshaped)
        outputs = model(**inputs)
        logits = outputs.logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), inputs["input_ids"].view(-1))

    try:
        print("[DEBUG] Computing Hessian diagonal for first {} elements of lm_head.weight...".format(subset_size))
        diag = compute_hessian_diagonal_subset(loss_fn, param_data, subset_size=subset_size)
        sensitivity_score = diag.abs().sum().item()
        return {"lm_head.weight (first {} elements)".format(subset_size): sensitivity_score}
    except RuntimeError as e:
        print(f"Diagonal Hessian computation failed: {e}")
        print("Falling back to gradient-based sensitivity...")
        try:
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs.logits.mean()
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
            grad_sensitivity = {}
            for (name, _), grad in zip(model.named_parameters(), grads):
                if grad is not None:
                    grad_sensitivity[name] = grad.abs().sum().item()
            return grad_sensitivity
        except RuntimeError as e_grad:
            print(f"Gradient-based sensitivity failed: {e_grad}")
            return {}

def plot_sensitivity(sensitivity_scores, output_file="llm_diagram_sensitivity_plot.png"):
    if not sensitivity_scores:
        print("No sensitivity scores to plot.")
        return
    names = list(sensitivity_scores.keys())
    scores = [sensitivity_scores[name] for name in names]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(names)), scores)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.xlabel("Parameter")
    plt.ylabel("Sensitivity Score")
    plt.title("LLM Weight Sensitivity Scores (Subset Hessian Diagonal)")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def normalize_sensitivity_scores(sensitivity_data):
    """
    Normalize sensitivity scores to range [0, 1] for comparison.
    
    Args:
        sensitivity_data: Dictionary of layer -> list of (index, score) tuples
        
    Returns:
        Dictionary with normalized scores
    """
    all_scores = []
    for layer, weights in sensitivity_data.items():
        all_scores.extend([score for _, score in weights])
        
    if not all_scores:
        return sensitivity_data
        
    min_score = min(all_scores)
    max_score = max(all_scores)
    
    if max_score == min_score:
        return sensitivity_data  # Avoid division by zero
    
    normalized_data = {}
    for layer, weights in sensitivity_data.items():
        normalized_data[layer] = [(idx, (score - min_score) / (max_score - min_score)) 
                                 for idx, score in weights]
    
    return normalized_data

def compare_sensitivity_methods(model, tokenizer, prompt, output_dir, methods=None):
    """
    Compare different sensitivity analysis methods on the same model.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompt: Text prompt to evaluate
        output_dir: Directory to save results
        methods: List of methods to compare
        
    Returns:
        Dictionary with comparison results
    """
    if methods is None:
        methods = ["gradient", "z_score", "integrated"]
    
    print(f"\n🔍 Comparing {len(methods)} sensitivity analysis methods:")
    for method in methods:
        print(f"  - {method}")
    
    results = {}
    os.makedirs(output_dir, exist_ok=True)
    
    # Run each method and collect results
    for method in methods:
        print(f"\n⏳ Running {method} sensitivity analysis...")
        method_start = time.time()
        
        if method == "gradient":
            sensitivity_data = compute_gradient_sensitivity(model, tokenizer, prompt)
        elif method == "hessian":
            sensitivity_data = compute_full_hessian_sensitivity(model, tokenizer, prompt)
        elif method == "integrated":
            sensitivity_data = compute_integrated_gradients(model, tokenizer, prompt)
        elif method == "z_score":
            super_weights, layer_summary = identify_super_weights(model, z_threshold=2.5)
            sensitivity_data = {}
            for layer, indices in super_weights.items():
                sensitivity_data[layer] = [(idx, 1.0) for idx in indices]
        else:
            print(f"❓ Unknown method: {method}, skipping...")
            continue
        
        method_time = time.time() - method_start
        
        # Normalize scores for comparison
        normalized_data = normalize_sensitivity_scores(sensitivity_data)
        
        # Extract top weights across all layers
        all_weights = []
        for layer, weights in normalized_data.items():
            for idx, score in weights:
                all_weights.append((layer, idx, score))
        
        # Sort by score (highest first)
        all_weights.sort(key=lambda x: x[2], reverse=True)
        top_100 = all_weights[:100]
        
        # Save method results
        method_output_dir = os.path.join(output_dir, method)
        os.makedirs(method_output_dir, exist_ok=True)
        
        # Save data
        with open(os.path.join(method_output_dir, "sensitivity_data.json"), "w") as f:
            json.dump({
                "method": method,
                "execution_time": method_time,
                "top_weights": [
                    {"layer": layer, "index": int(idx), "score": float(score)}
                    for layer, idx, score in top_100
                ]
            }, f, indent=2)
        
        # Generate visualization
        visualize_sensitivity_maps(model, normalized_data, method_output_dir)
        
        # Track method summary
        results[method] = {
            "execution_time": method_time,
            "avg_top_score": np.mean([score for _, _, score in top_100]),
            "output_dir": method_output_dir
        }
    
    # Compare overlap between methods
    if len(methods) > 1:
        print("\n📊 Comparing overlap between sensitivity methods...")
        overlap_matrix = {}
        
        # Load each method's top 100 weights
        method_weights = {}
        for method in methods:
            data_file = os.path.join(output_dir, method, "sensitivity_data.json")
            if not os.path.exists(data_file):
                continue
                
            with open(data_file, "r") as f:
                data = json.load(f)
            
            # Extract (layer, index) pairs
            weight_set = set(
                (item["layer"], item["index"]) 
                for item in data["top_weights"]
            )
            method_weights[method] = weight_set
        
        # Compute Jaccard similarity between each pair
        for method1 in methods:
            if method1 not in method_weights:
                continue
                
            overlap_matrix[method1] = {}
            for method2 in methods:
                if method2 not in method_weights:
                    continue
                    
                weights1 = method_weights[method1]
                weights2 = method_weights[method2]
                
                # Jaccard similarity: |A ∩ B| / |A ∪ B|
                intersection = len(weights1.intersection(weights2))
                union = len(weights1.union(weights2))
                
                similarity = intersection / union if union > 0 else 0
                overlap_matrix[method1][method2] = similarity
        
        # Visualize similarity matrix
        plt.figure(figsize=(8, 6))
        matrix_methods = list(overlap_matrix.keys())
        matrix_values = np.array([
            [overlap_matrix[m1].get(m2, 0) for m2 in matrix_methods]
            for m1 in matrix_methods
        ])
        
        plt.imshow(matrix_values, vmin=0, vmax=1, cmap="viridis")
        plt.colorbar(label="Jaccard Similarity")
        
        plt.xticks(np.arange(len(matrix_methods)), matrix_methods, rotation=45)
        plt.yticks(np.arange(len(matrix_methods)), matrix_methods)
        
        # Add text annotations
        for i in range(len(matrix_methods)):
            for j in range(len(matrix_methods)):
                plt.text(j, i, f"{matrix_values[i, j]:.2f}", 
                        ha="center", va="center", 
                        color="white" if matrix_values[i, j] < 0.7 else "black")
        
        plt.title("Sensitivity Method Similarity")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "method_similarity.png"))
        plt.close()
        
        # Save similarity matrix
        with open(os.path.join(output_dir, "method_similarity.json"), "w") as f:
            json.dump(overlap_matrix, f, indent=2)
    
    # Create comparison summary report
    with open(os.path.join(output_dir, "comparison_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Sensitivity method comparison complete")
    print(f"📂 Results saved to: {output_dir}")
    
    return results

def main(model_name=None, output_dir=None, methods=None):
    """
    Run comparative analysis of sensitivity methods.
    
    Args:
        model_name: Name of the model to analyze
        output_dir: Directory to save results
        methods: List of methods to compare
    """
    # Use default model if not specified
    if model_name is None:
        model_name = MODEL_NAME
    
    # Create output directory
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/compare_sensitivity_{model_name.replace('/', '_')}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📂 Output directory: {output_dir}")
    
    # Set default methods if not specified
    if methods is None:
        methods = ["gradient", "z_score", "integrated"]
    elif isinstance(methods, str):
        # Allow comma-separated methods
        methods = methods.split(",")
    
    print(f"\n🔍 Running sensitivity method comparison for model: {model_name}")
    
    # Load model and tokenizer
    print("\n⏳ Loading model, please wait...")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    model.to(device)
    
    # Run the comparison
    results = compare_sensitivity_methods(
        model=model,
        tokenizer=tokenizer,
        prompt=TEST_PROMPT,
        output_dir=output_dir,
        methods=methods
    )
    
    # Create a summary visualization
    plt.figure(figsize=(10, 6))
    
    method_names = list(results.keys())
    execution_times = [results[method]["execution_time"] for method in method_names]
    
    plt.bar(method_names, execution_times)
    plt.ylabel("Execution Time (seconds)")
    plt.title("Sensitivity Method Performance Comparison")
    plt.xticks(rotation=45)
    
    for i, v in enumerate(execution_times):
        plt.text(i, v + 0.1, f"{v:.1f}s", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "execution_time_comparison.png"))
    
    print(f"\n🎉 Method comparison completed!")
    print(f"📂 Results available in: {output_dir}")
    
    return {
        "model_name": model_name,
        "methods": methods,
        "results": results,
        "output_dir": output_dir
    }

if __name__ == "__main__":
    main()
