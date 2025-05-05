import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import multiprocessing
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from config import MODEL_NAME, TEST_PROMPT, ANALYSIS_CONFIG
from torch.func import functional_call, vmap, vjp, jvp, jacrev

# Set up parallel processing capability
NUM_CPUS = multiprocessing.cpu_count()
# Use 75% of available CPUs but at least 2
PARALLEL_WORKERS = max(2, int(NUM_CPUS * 0.75))

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

def plot_layerwise_superweights(layer_summary, output_file=None):
    if not layer_summary:
        print("No super weights found to plot.")
        return
    
    if output_file is None:
        output_file = "layerwise_superweights.png"

    layers = list(layer_summary.keys())
    counts = [layer_summary[layer] for layer in layers]

    plt.figure(figsize=(12, 6))
    plt.bar(layers, counts)
    plt.xlabel("Layer")
    plt.ylabel("Super Weight Count")
    plt.title("Layer-wise Super Weight Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"✅ Saved plot: {output_file}")

def compute_gradient_sensitivity(model, tokenizer, prompt, top_k=100, use_parallel=True):
    """
    Compute sensitivity scores based on input-output gradients in parallel.
    """
    print(f"\n🔍 Computing gradient-based sensitivity scores (using {PARALLEL_WORKERS} workers)...")
    device = next(model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None
    
    outputs = model(**inputs)
    logits = outputs.logits
    
    target_ids = input_ids.clone()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = target_ids[:, 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss.backward()
    
    sensitivity_scores = {}
    all_scores = []
    
    # Modified section to use parallel processing for large models
    if use_parallel and len(list(model.named_parameters())) > 100:
        # Process parameters in parallel batches
        param_batches = []
        current_batch = []
        current_size = 0
        batch_size = 5000000  # ~20MB per batch
        
        # Group parameters into reasonably sized batches
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad and param.grad is not None:
                param_size = param.numel()
                if current_size + param_size > batch_size and current_batch:
                    param_batches.append(current_batch)
                    current_batch = []
                    current_size = 0
                current_batch.append((name, param))
                current_size += param_size
        
        if current_batch:
            param_batches.append(current_batch)
        
        # Process batches in parallel
        all_sensitivity_scores = {}
        all_scores_list = []
        
        def process_param_batch(batch):
            batch_scores = {}
            for name, param in batch:
                grad = param.grad.detach().cpu().abs()
                weight = param.detach().cpu().abs()
                sensitivity = (grad * weight).flatten()
                
                if len(sensitivity) > 0:
                    sorted_indices = torch.argsort(sensitivity, descending=True)
                    top_indices = sorted_indices[:min(top_k, len(sorted_indices))]
                    
                    batch_scores[name] = [(idx.item(), sensitivity[idx].item()) 
                                       for idx in top_indices]
            return batch_scores
            
        print(f"Processing {len(param_batches)} parameter batches in parallel...")
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = [executor.submit(process_param_batch, batch) for batch in param_batches]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                batch_result = future.result()
                all_sensitivity_scores.update(batch_result)
                for scores in batch_result.values():
                    all_scores_list.extend([score for _, score in scores])
        
        # Normalize scores
        max_score = max(all_scores_list) if all_scores_list else 1.0
        for name in all_sensitivity_scores:
            all_sensitivity_scores[name] = [(idx, score/max_score) 
                                   for idx, score in all_sensitivity_scores[name]]
        
        return all_sensitivity_scores
    else:
        # Use original non-parallel implementation for smaller models
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad and param.grad is not None:
                grad = param.grad.detach().abs()
                sensitivity = grad * param.detach().abs()
                flattened = sensitivity.flatten()
                
                if len(flattened) > 0:
                    sorted_indices = torch.argsort(flattened, descending=True)
                    top_indices = sorted_indices[:min(top_k, len(sorted_indices))]
                    
                    sensitivity_scores[name] = [(idx.item(), flattened[idx].item()) 
                                               for idx in top_indices]
                    
                    all_scores.extend([score for _, score in sensitivity_scores[name]])
        
        max_score = max(all_scores) if all_scores else 1.0
        for name in sensitivity_scores:
            sensitivity_scores[name] = [(idx, score/max_score) 
                                       for idx, score in sensitivity_scores[name]]
        
        return sensitivity_scores

def compute_hessian_sensitivity(model, tokenizer, prompt, layers_to_analyze=None, samples=10):
    print("\n🧮 Computing Hessian-based sensitivity (this may take a while)...")
    device = next(model.parameters()).device
    epsilon = 1e-4
    
    if layers_to_analyze is None:
        layers_to_analyze = []
        for name, _ in model.named_parameters():
            if "mlp" in name and "weight" in name and "down_proj" not in name:
                if "layer.0" in name or "layers.0" in name:
                    layers_to_analyze.append(name)
                    break
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    def compute_loss():
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    model.zero_grad()
    base_loss = compute_loss()
    base_loss.backward()
    
    base_grads = {}
    for name, param in model.named_parameters():
        if name in layers_to_analyze and param.grad is not None:
            base_grads[name] = param.grad.detach().clone()
    
    hessian_diag = {}
    
    for name in layers_to_analyze:
        print(f"  Analyzing layer: {name}")
        param = dict(model.named_parameters())[name]
        hessian_estimates = torch.zeros_like(param)
        
        for _ in range(samples):
            perturbation = torch.randn_like(param) * epsilon
            
            with torch.no_grad():
                param.add_(perturbation)
            
            model.zero_grad()
            new_loss = compute_loss()
            new_loss.backward()
            
            grad_diff = param.grad - base_grads[name]
            hessian_estimate = grad_diff / epsilon
            
            hessian_estimates.add_(hessian_estimate.abs() / samples)
            
            with torch.no_grad():
                param.sub_(perturbation)
        
        hessian_diag[name] = hessian_estimates.abs().detach()
    
    sensitivity_scores = {}
    all_scores = []
    
    for name, hessian in hessian_diag.items():
        param = dict(model.named_parameters())[name]
        sensitivity = (hessian * param.detach().abs()).flatten()
        
        sorted_indices = torch.argsort(sensitivity, descending=True)
        top_indices = sorted_indices[:min(100, len(sorted_indices))]
        
        sensitivity_scores[name] = [(idx.item(), sensitivity[idx].item()) 
                                   for idx in top_indices]
        
        all_scores.extend([score for _, score in sensitivity_scores[name]])
    
    max_score = max(all_scores) if all_scores else 1.0
    for name in sensitivity_scores:
        sensitivity_scores[name] = [(idx, score/max_score) 
                                   for idx, score in sensitivity_scores[name]]
    
    return sensitivity_scores

def compute_integrated_gradients(model, tokenizer, prompt, steps=50, top_k=100):
    print("\n🔄 Computing integrated gradients sensitivity...")
    device = next(model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    params_dict = dict(model.named_parameters())
    baseline_dict = {name: torch.zeros_like(param) for name, param in params_dict.items()}
    
    sensitivity_scores = {}
    accumulated_grads = {name: torch.zeros_like(param) for name, param in params_dict.items() 
                        if "weight" in name and param.requires_grad}
    
    for step in range(1, steps + 1):
        print(f"  Integration step {step}/{steps}")
        alpha = step / steps
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                baseline = baseline_dict[name]
                with torch.no_grad():
                    param.data.copy_(baseline + alpha * (params_dict[name] - baseline))
        
        model.zero_grad()
        outputs = model(**inputs)
        logits = outputs.logits
        
        target_ids = inputs["input_ids"].clone()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad and param.grad is not None:
                accumulated_grads[name].add_(param.grad.detach())
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            with torch.no_grad():
                param.data.copy_(params_dict[name].data)
    
    all_scores = []
    for name, accumulated_grad in accumulated_grads.items():
        attribution = accumulated_grad * (params_dict[name] - baseline_dict[name]) / steps
        flattened = attribution.abs().flatten()
        
        if len(flattened) > 0:
            sorted_indices = torch.argsort(flattened, descending=True)
            top_indices = sorted_indices[:min(top_k, len(sorted_indices))]
            
            sensitivity_scores[name] = [(idx.item(), flattened[idx].item()) 
                                      for idx in top_indices]
            
            all_scores.extend([score for _, score in sensitivity_scores[name]])
    
    max_score = max(all_scores) if all_scores else 1.0
    for name in sensitivity_scores:
        sensitivity_scores[name] = [(idx, score/max_score) 
                                   for idx, score in sensitivity_scores[name]]
    
    return sensitivity_scores

def ablation_sensitivity_test(model, tokenizer, sensitivity_data, prompt, output_dir):
    print("\n🧪 Testing super weight sensitivity via targeted perturbation...")
    device = next(model.parameters()).device
    
    def evaluate_perplexity():
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            return torch.exp(outputs.loss).item()
    
    baseline_ppl = evaluate_perplexity()
    print(f"  Baseline perplexity: {baseline_ppl:.2f}")
    
    params_dict = dict(model.named_parameters())
    backups = {}
    results = {
        "baseline_perplexity": baseline_ppl,
        "perturbation_results": []
    }
    
    perturbation_levels = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    weights_to_perturb = []
    for layer_name, weight_data in sensitivity_data.items():
        param_shape = params_dict[layer_name].shape
        for idx, score in weight_data:
            multi_idx = np.unravel_index(idx, param_shape)
            weights_to_perturb.append({
                'layer': layer_name,
                'indices': multi_idx,
                'score': score
            })
    
    weights_to_perturb.sort(key=lambda x: x['score'], reverse=True)
    
    for level in perturbation_levels:
        num_weights = int(len(weights_to_perturb) * level)
        
        backups = {}
        for item in weights_to_perturb[:num_weights]:
            layer = item['layer']
            indices = item['indices']
            
            if layer not in backups:
                backups[layer] = params_dict[layer].detach().clone()
            
            with torch.no_grad():
                params_dict[layer][indices] = 0.0
        
        perturbed_ppl = evaluate_perplexity()
        relative_change = (perturbed_ppl - baseline_ppl) / baseline_ppl * 100
        
        print(f"  Perturbing {num_weights} weights ({level*100:.1f}%): "
              f"New PPL = {perturbed_ppl:.2f} (Change: {relative_change:.2f}%)")
        
        results["perturbation_results"].append({
            "perturbation_level": level,
            "weights_perturbed": num_weights,
            "perplexity": perturbed_ppl,
            "relative_change_percent": relative_change
        })
        
        for layer, backup in backups.items():
            with torch.no_grad():
                params_dict[layer].copy_(backup)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "perturbation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    plt.figure(figsize=(10, 6))
    levels = [r["perturbation_level"] * 100 for r in results["perturbation_results"]]
    ppl_changes = [r["relative_change_percent"] for r in results["perturbation_results"]]
    
    plt.plot(levels, ppl_changes, marker='o', linewidth=2)
    plt.xlabel("Percentage of Super Weights Perturbed (%)")
    plt.ylabel("Perplexity Change (%)")
    plt.title(f"Impact of Targeted Super Weight Perturbation\nBaseline PPL: {baseline_ppl:.2f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "perturbation_impact.png"))
    plt.close()
    
    return results

def visualize_sensitivity_maps(model, sensitivity_data, output_dir):
    print("\n📊 Generating sensitivity visualization maps...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    layer_summary = defaultdict(float)
    layer_counts = defaultdict(int)
    
    for name, weight_data in sensitivity_data.items():
        if "layers." in name:
            parts = name.split(".")
            try:
                layer_idx = int(parts[2])
                layer_type = parts[3]
                component = parts[4] if len(parts) > 4 else ""
                layer_name = f"L{layer_idx}_{layer_type}_{component}"
            except:
                layer_name = name
        else:
            layer_name = name.split(".")[0]
            
        for _, score in weight_data:
            layer_summary[layer_name] += score
            layer_counts[layer_name] += 1
    
    for layer in layer_summary:
        if layer_counts[layer] > 0:
            layer_summary[layer] /= layer_counts[layer]
    
    layers = list(layer_summary.keys())
    
    def extract_layer_num(layer_name):
        if layer_name.startswith('L') and '_' in layer_name:
            try:
                return int(layer_name.split('_')[0][1:])
            except:
                return 999
        return 999
    
    layers.sort(key=extract_layer_num)
    
    plt.figure(figsize=(12, 8))
    scores = [layer_summary[layer] for layer in layers]
    
    plt.bar(range(len(layers)), scores, color='royalblue')
    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.xlabel("Model Layer")
    plt.ylabel("Average Sensitivity Score")
    plt.title("Layer-wise Sensitivity Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sensitivity_map.png"))
    plt.close()
    
    with open(os.path.join(output_dir, "layer_sensitivity.json"), "w") as f:
        json.dump({
            "layer_sensitivity": {k: float(v) for k, v in layer_summary.items()},
            "layer_counts": layer_counts
        }, f, indent=2)
    
    print(f"✅ Saved sensitivity map to {os.path.join(output_dir, 'sensitivity_map.png')}")
    
    return layer_summary

def extract_super_weights(model, sensitivity_data, threshold=0.5, max_weights=1000):
    all_weights = []
    for layer_name, weight_data in sensitivity_data.items():
        for idx, score in weight_data:
            if score >= threshold:
                all_weights.append((layer_name, idx, score))
    
    all_weights.sort(key=lambda x: x[2], reverse=True)
    
    all_weights = all_weights[:max_weights]
    
    super_weights = defaultdict(list)
    for layer_name, idx, score in all_weights:
        super_weights[layer_name].append({"index": idx, "score": float(score)})
    
    return dict(super_weights)

def save_sensitivity_results(sensitivity_data, output_dir, method_name):
    os.makedirs(output_dir, exist_ok=True)
    
    serializable_data = {}
    for layer_name, weight_data in sensitivity_data.items():
        serializable_data[layer_name] = [{"index": int(idx), "score": float(score)} 
                                        for idx, score in weight_data]
    
    output_file = os.path.join(output_dir, f"super_weights_{method_name}.json")
    with open(output_file, "w") as f:
        json.dump({
            "method": method_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "super_weights": serializable_data
        }, f, indent=2)
    
    print(f"✅ Saved sensitivity results to {output_file}")

def get_cached_sensitivity_results(model_name, method, output_dir):
    """
    Check if we already have sensitivity results cached for this model and method.
    
    Args:
        model_name: Name of the model
        method: Sensitivity method used
        output_dir: Output directory for this run
        
    Returns:
        Cached results or None if not found
    """
    # Extract simple model name for cache path
    if "/" in model_name:
        simple_name = model_name.split("/")[-1].lower()
    else:
        simple_name = model_name.lower()
    
    # Check in outputs directory for recent results
    from config import OUTPUT_DIR
    import glob
    import os
    
    # Look for sensitivity results in the output directory
    result_files = glob.glob(f"{OUTPUT_DIR}/super*_{simple_name}_*/super_weights_{method}.json")
    
    if not result_files:
        return None
    
    # Use the most recent results file
    result_files.sort(key=os.path.getmtime, reverse=True)
    latest_result = result_files[0]
    
    try:
        with open(latest_result, 'r') as f:
            import json
            data = json.load(f)
        
        print(f"✅ Found cached sensitivity results: {latest_result}")
        
        # Convert the cached format back to our expected format
        sensitivity_data = {}
        for layer_name, weights in data.get('super_weights', {}).items():
            sensitivity_data[layer_name] = [(item['index'], item['score']) 
                                          for item in weights]
        
        # Save a copy to the current output directory
        import shutil
        os.makedirs(output_dir, exist_ok=True)
        cache_info_file = os.path.join(output_dir, "cache_info.json")
        with open(cache_info_file, 'w') as f:
            json.dump({
                "cached_from": latest_result,
                "timestamp": data.get("timestamp", "unknown"),
                "reused_on": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        
        # Also copy the actual file
        target_file = os.path.join(output_dir, f"super_weights_{method}.json")
        shutil.copy2(latest_result, target_file)
        
        print(f"ℹ️ Using cached results to save computation time")
        return sensitivity_data
    except Exception as e:
        print(f"⚠️ Error loading cached results: {e}")
        return None

def main(model_name=None, output_dir=None, method="gradient", threshold=0.7, parallel=True, use_cache=True):
    """
    Identify and analyze super weights in a transformer model.
    
    Args:
        model_name: Name of the model to analyze
        output_dir: Directory to save results
        method: Sensitivity method to use (z_score, gradient, hessian, integrated)
        threshold: Sensitivity threshold for super weight classification
        parallel: Whether to use parallel processing for large models
        use_cache: Whether to use cached sensitivity results if available
    """
    # Use default model if not specified
    if model_name is None:
        model_name = MODEL_NAME
    else:
        # Get full model name if a short key was provided
        from config import get_model_by_key
        model_name = get_model_by_key(model_name)
    
    # Create output directory
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/super_weights_{model_name.replace('/', '_')}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📂 Output directory: {output_dir}")
    
    print(f"\n🔍 Analyzing super weights for model: {model_name}")
    print(f"📊 Using method: {method}")

    # Initialize variables
    execution_time = 0
    total_super_weights = 0
    model = None
    tokenizer = None
    
    # Check for cached results if caching is enabled
    cached_results = None
    if use_cache:
        cached_results = get_cached_sensitivity_results(model_name, method, output_dir)
    
    if cached_results is not None:
        sensitivity_data = cached_results
        print(f"\n🔄 Using cached sensitivity results")
        
        # Load the model just for visualization and analysis purposes
        print("\n⏳ Loading model for visualization...")
        
        try:
            # Try loading with optimized settings first
            try:
                from utils.model_loader import load_model_optimized
                model, tokenizer = load_model_optimized(model_name, load_8bit=False)
            except (ImportError, Exception) as e:
                print(f"⚠️ Optimized loader not available or failed: {e}")
                # Fall back to regular loading if the utility isn't available
                from config import MODEL_CONFIG
                model_config = MODEL_CONFIG.copy()
                
                # Remove quantization settings that might cause issues
                if "load_in_8bit" in model_config:
                    model_config.pop("load_in_8bit")
                if "quantization_config" in model_config:
                    model_config.pop("quantization_config")
                if "use_flash_attention" in model_config:
                    model_config.pop("use_flash_attention")
                if "use_memory_efficient_attention" in model_config:
                    model_config.pop("use_memory_efficient_attention")
                
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_config)
                tokenizer = AutoTokenizer.from_pretrained(model_name, **model_config)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"🖥️ Using device: {device}")
                model.to(device)
        except Exception as e:
            print(f"⚠️ Warning: Model loading failed with error: {e}")
            print("⚠️ Continuing with analysis using cached data, but visualizations may be limited.")
            # Create dummy model for structure analysis only
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name)
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_config(config)
            except:
                print("⚠️ Could not create model structure. Some functionality will be limited.")
    else:
        # Load model and tokenizer at full precision for analysis
        print("\n⏳ Loading model, please wait...")
        
        try:
            # Use model configuration from config.py
            from config import MODEL_CONFIG
            model_config = MODEL_CONFIG.copy()
            
            # Remove quantization settings that might cause issues
            if "load_in_8bit" in model_config:
                model_config.pop("load_in_8bit") 
            if "quantization_config" in model_config:
                model_config.pop("quantization_config")
                
            from transformers import AutoModelForCausalLM  # Ensure we have the import here
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_config)
            tokenizer = AutoTokenizer.from_pretrained(model_name, **model_config)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🖥️ Using device: {device}")
            model.to(device)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

        sensitivity_data = {}
        
        # Add performance timing
        start_time = time.time()
        
        if method == "z_score":
            super_weights, layer_summary = identify_super_weights(model, z_threshold=threshold*5)
            for layer, indices in super_weights.items():
                sensitivity_data[layer] = [(idx, 1.0) for idx in indices]
        
        elif method == "gradient":
            sensitivity_data = compute_gradient_sensitivity(
                model, tokenizer, TEST_PROMPT, top_k=1000, use_parallel=parallel)
        
        elif method == "hessian":
            sensitivity_data = compute_hessian_sensitivity(
                model, tokenizer, TEST_PROMPT)
        
        elif method == "integrated":
            sensitivity_data = compute_integrated_gradients(
                model, tokenizer, TEST_PROMPT)
        
        else:
            print(f"❌ Unknown method: {method}. Using gradient method instead.")
            sensitivity_data = compute_gradient_sensitivity(
                model, tokenizer, TEST_PROMPT)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # If we didn't use cache, save the results for future use
        save_sensitivity_results(sensitivity_data, output_dir, method)
    
    # Process the sensitivity data (whether from cache or newly computed)
    # Extract super weights
    super_weights = extract_super_weights(model, sensitivity_data, threshold=threshold)
    
    total_super_weights = sum(len(weights) for weights in super_weights.values())
    print(f"\n✅ Identified {total_super_weights} super weights across {len(super_weights)} layers")
    
    # Generate visualizations and perform analysis
    layer_sensitivity = visualize_sensitivity_maps(model, sensitivity_data, output_dir)
    
    results = ablation_sensitivity_test(
        model, tokenizer, sensitivity_data, TEST_PROMPT, output_dir)
    
    # Add performance metrics to output
    performance_data = {
        "analysis_method": method,
        "execution_time_seconds": execution_time,
        "parallel_processing": parallel,
        "num_workers": PARALLEL_WORKERS if parallel else 1,
        "model_size_parameters": sum(p.numel() for p in model.parameters()),
        "super_weights_found": total_super_weights,
        "date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(output_dir, "performance_metrics.json"), "w") as f:
        json.dump(performance_data, f, indent=2)
        
    if execution_time > 0:
        print(f"\n⏱️ Total execution time: {execution_time:.2f} seconds")
    
    print(f"\n🎉 Super weight analysis completed!")
    print(f"📂 Results available in: {output_dir}")
    
    return {
        "model_name": model_name,
        "method": method,
        "super_weights_count": total_super_weights,
        "layer_sensitivity": layer_sensitivity,
        "output_dir": output_dir
    }

if __name__ == "__main__":
    main()
