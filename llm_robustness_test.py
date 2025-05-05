import torch
import copy
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, get_model_by_key, get_model_config, get_model_paths, TEST_PROMPT

def apply_robustness_test(model_name, output_path, noise_type="gaussian", noise_level=0.01):
    """
    Applies noise to model weights to test robustness.
    
    Args:
        model_name: Name or path of the model
        output_path: Path to save the noisy model
        noise_type: Type of noise ('gaussian', 'uniform', 'targeted')
        noise_level: Standard deviation or magnitude of noise
    
    Returns:
        Tuple of (original_model, noisy_model, noise_stats)
    """
    print(f"\n⏳ Loading model: {model_name}")
    
    # Use appropriate model configuration from config.py
    model_config = get_model_config(model_name.split('/')[-1] if '/' in model_name else model_name)
    
    try:
        # Load model using the configuration
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **model_config)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    print(f"✅ Model loaded successfully")
    
    # Create a deep copy of the model for noise application
    noisy_model = copy.deepcopy(model)
    
    # Track noise statistics
    noise_stats = {
        "total_params": 0,
        "affected_params": 0,
        "noise_magnitude_avg": 0,
        "param_magnitude_avg": 0,
        "max_relative_change": 0,
        "max_relative_change_layer": "",
        "layer_stats": {}
    }
    
    print(f"\n🔄 Applying {noise_type} noise with level {noise_level}...")
    
    for name, param in noisy_model.named_parameters():
        if param.requires_grad:
            noise_stats["total_params"] += param.numel()
            
            # Track original parameter stats
            param_abs_avg = param.abs().mean().item()
            noise_stats["param_magnitude_avg"] += param_abs_avg
            
            # Generate noise based on the selected type
            if noise_type == "gaussian":
                noise = noise_level * torch.randn_like(param)
            elif noise_type == "uniform":
                noise = noise_level * (2 * torch.rand_like(param) - 1)
            elif noise_type == "targeted":
                # Only apply noise to specific parts (e.g., attention layers)
                if "attn" in name:
                    noise = noise_level * torch.randn_like(param)
                else:
                    noise = torch.zeros_like(param)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")
            
            # Apply noise
            param.data += noise
            
            # Track noise statistics
            noise_abs_avg = noise.abs().mean().item()
            noise_stats["noise_magnitude_avg"] += noise_abs_avg
            noise_stats["affected_params"] += param.numel()
            
            # Calculate relative change
            relative_change = noise_abs_avg / (param_abs_avg + 1e-10)
            if relative_change > noise_stats["max_relative_change"]:
                noise_stats["max_relative_change"] = relative_change
                noise_stats["max_relative_change_layer"] = name
            
            # Add per-layer stats
            layer_key = name.split('.')[0] if '.' in name else name
            if layer_key not in noise_stats["layer_stats"]:
                noise_stats["layer_stats"][layer_key] = {
                    "avg_noise": 0,
                    "avg_param": 0,
                    "param_count": 0
                }
            
            noise_stats["layer_stats"][layer_key]["avg_noise"] += noise_abs_avg * param.numel()
            noise_stats["layer_stats"][layer_key]["avg_param"] += param_abs_avg * param.numel()
            noise_stats["layer_stats"][layer_key]["param_count"] += param.numel()
    
    # Finalize the statistics
    noise_stats["noise_magnitude_avg"] /= max(1, len(noise_stats["layer_stats"]))
    noise_stats["param_magnitude_avg"] /= max(1, len(noise_stats["layer_stats"]))
    
    for layer in noise_stats["layer_stats"]:
        layer_stats = noise_stats["layer_stats"][layer]
        if layer_stats["param_count"] > 0:
            layer_stats["avg_noise"] /= layer_stats["param_count"]
            layer_stats["avg_param"] /= layer_stats["param_count"]
    
    # Save the noisy model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(noisy_model.state_dict(), output_path)
    print(f"✅ Noise added to model. Saved as {output_path}")
    
    return model, noisy_model, noise_stats

def evaluate_model_robustness(model, noisy_model, tokenizer, prompt=None):
    """
    Evaluates original and noisy models to measure robustness.
    
    Args:
        model: Original model
        noisy_model: Noisy model
        tokenizer: Tokenizer for the models
        prompt: Test prompt (if None, uses default)
    
    Returns:
        Dict of evaluation metrics
    """
    if prompt is None:
        prompt = TEST_PROMPT
    
    print(f"\n📊 Evaluating model robustness...")
    print(f"Test prompt: '{prompt}'")
    
    device = next(model.parameters()).device
    
    def evaluate_perplexity(model, prompt):
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            return torch.exp(torch.tensor(loss)).item()
    
    original_ppl = evaluate_perplexity(model, prompt)
    noisy_ppl = evaluate_perplexity(noisy_model, prompt)
    
    print(f"Original model perplexity: {original_ppl:.2f}")
    print(f"Noisy model perplexity: {noisy_ppl:.2f}")
    
    # Generate completions for comparison
    def generate_completion(model, prompt, max_new_tokens=50):
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.92,
                temperature=0.8
            )
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return completion[len(prompt):].strip()
    
    original_completion = generate_completion(model, prompt)
    noisy_completion = generate_completion(noisy_model, prompt)
    
    print(f"\n📝 Original model completion:\n{original_completion}")
    print(f"\n📝 Noisy model completion:\n{noisy_completion}")
    
    # Calculate similarity between outputs
    from difflib import SequenceMatcher
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    output_similarity = similarity(original_completion, noisy_completion)
    print(f"Output similarity: {output_similarity:.2f}")
    
    return {
        "original_perplexity": original_ppl,
        "noisy_perplexity": noisy_ppl,
        "perplexity_ratio": noisy_ppl / original_ppl,
        "output_similarity": output_similarity,
        "original_completion": original_completion,
        "noisy_completion": noisy_completion
    }

def run_noise_sweep(model_name, output_dir, noise_type="gaussian", noise_levels=None):
    """
    Run a sweep over multiple noise levels to measure robustness.
    
    Args:
        model_name: Model name or path
        output_dir: Directory to save results
        noise_type: Type of noise to apply
        noise_levels: List of noise levels to test (None for default range)
    """
    if noise_levels is None:
        noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    
    print(f"\n🧪 Running noise sweep with {noise_type} noise")
    
    model_config = get_model_config(model_name.split('/')[-1] if '/' in model_name else model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **model_config)
    device = next(model.parameters()).device
    
    sweep_results = {
        "model": model_name,
        "noise_type": noise_type,
        "results": []
    }
    
    for noise_level in noise_levels:
        print(f"\n🔄 Testing noise level: {noise_level}")
        
        # Create a noisy copy of the model
        noisy_model = copy.deepcopy(model)
        for name, param in noisy_model.named_parameters():
            if param.requires_grad:
                if noise_type == "gaussian":
                    noise = noise_level * torch.randn_like(param)
                elif noise_type == "uniform":
                    noise = noise_level * (2 * torch.rand_like(param) - 1)
                else:
                    raise ValueError(f"Unknown noise type: {noise_type}")
                param.data += noise
        
        # Evaluate this noise level
        def evaluate_perplexity(model, prompt=TEST_PROMPT):
            model.eval()
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                return torch.exp(torch.tensor(loss)).item()
        
        original_ppl = evaluate_perplexity(model)
        noisy_ppl = evaluate_perplexity(noisy_model)
        
        sweep_results["results"].append({
            "noise_level": noise_level,
            "original_perplexity": original_ppl,
            "noisy_perplexity": noisy_ppl,
            "perplexity_ratio": noisy_ppl / original_ppl
        })
        
        print(f"  PPL ratio at noise={noise_level}: {noisy_ppl / original_ppl:.2f}x")
    
    # Save the sweep results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "noise_sweep_results.json"), "w") as f:
        json.dump(sweep_results, f, indent=2)
    
    # Create a visualization
    plt.figure(figsize=(10, 6))
    
    x = [r["noise_level"] for r in sweep_results["results"]]
    y = [r["perplexity_ratio"] for r in sweep_results["results"]]
    
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2)
    plt.xscale('log')  # Use log scale for x-axis to better visualize different noise levels
    plt.yscale('log')  # Use log scale for better visualization of perplexity ratio
    
    plt.xlabel('Noise Level')
    plt.ylabel('Perplexity Ratio (Noisy / Original)')
    plt.title(f'Robustness Test: {noise_type.capitalize()} Noise vs. Perplexity Impact')
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at ratio = 2 for reference
    plt.axhline(y=2, color='r', linestyle='--', alpha=0.5)
    plt.text(x[0], 2.1, '2x worse', color='r', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "noise_sweep_plot.png"))
    
    print(f"✅ Noise sweep completed and results saved to {output_dir}")
    return sweep_results

def main(model_name=None, output_dir=None, noise_type="gaussian", noise_level=0.01, run_sweep=True, cpu_only=False):
    """
    Main function for running robustness tests compatible with run_analysis.py
    
    Args:
        model_name: Name of the model to analyze (or None for default)
        output_dir: Directory to save results (or None for auto-generated)
        noise_type: Type of noise to apply ('gaussian', 'uniform', 'targeted')
        noise_level: Noise level for the main test
        run_sweep: Whether to run a sweep over multiple noise levels
        cpu_only: Whether to force CPU-only mode
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
        output_dir = f"outputs/robustness_{model_name.replace('/', '_')}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📂 Output directory: {output_dir}")
    
    print(f"\n🔍 Running robustness analysis for model: {model_name}")
    print(f"📊 Using noise type: {noise_type}, noise level: {noise_level}")
    
    # Set CPU-only mode if requested
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["USE_CPU_ONLY"] = "1"
        print("⚠️ Using CPU-only mode for analysis")
    
    # Get model paths
    model_paths = get_model_paths(model_name)
    output_path = model_paths["noisy"]
    
    # Apply noise and evaluate
    try:
        start_time = time.time()
        
        model, noisy_model, noise_stats = apply_robustness_test(
            model_name, output_path, noise_type, noise_level
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        eval_results = evaluate_model_robustness(model, noisy_model, tokenizer)
        
        # Run a sweep if requested
        sweep_results = None
        if run_sweep:
            sweep_results = run_noise_sweep(model_name, output_dir, noise_type)
        
        # Save detailed results
        with open(os.path.join(output_dir, "robustness_results.json"), "w") as f:
            json.dump({
                "model_name": model_name,
                "noise_type": noise_type,
                "noise_level": noise_level,
                "noise_stats": noise_stats,
                "evaluation": eval_results,
                "execution_time": time.time() - start_time
            }, f, indent=2)
        
        # Save a human-readable report
        with open(os.path.join(output_dir, "robustness_report.txt"), "w") as f:
            f.write(f"ROBUSTNESS ANALYSIS REPORT\n")
            f.write(f"========================\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"TEST CONFIGURATION\n")
            f.write(f"------------------\n")
            f.write(f"Noise type: {noise_type}\n")
            f.write(f"Noise level: {noise_level}\n")
            f.write(f"Prompt: '{TEST_PROMPT}'\n\n")
            
            f.write(f"RESULTS\n")
            f.write(f"-------\n")
            f.write(f"Original model perplexity: {eval_results['original_perplexity']:.2f}\n")
            f.write(f"Noisy model perplexity: {eval_results['noisy_perplexity']:.2f}\n")
            f.write(f"Perplexity ratio: {eval_results['perplexity_ratio']:.2f}x\n")
            f.write(f"Output similarity: {eval_results['output_similarity']:.2f}\n\n")
            
            f.write(f"NOISE STATISTICS\n")
            f.write(f"---------------\n")
            f.write(f"Total parameters: {noise_stats['total_params']:,}\n")
            f.write(f"Affected parameters: {noise_stats['affected_params']:,}\n")
            f.write(f"Average parameter magnitude: {noise_stats['param_magnitude_avg']:.6f}\n")
            f.write(f"Average noise magnitude: {noise_stats['noise_magnitude_avg']:.6f}\n")
            f.write(f"Max relative change: {noise_stats['max_relative_change']:.6f}\n")
            f.write(f"Max change layer: {noise_stats['max_relative_change_layer']}\n\n")
            
            f.write(f"TEXT GENERATION COMPARISON\n")
            f.write(f"------------------------\n")
            f.write(f"Prompt: '{TEST_PROMPT}'\n\n")
            f.write(f"Original model completion:\n{eval_results['original_completion']}\n\n")
            f.write(f"Noisy model completion:\n{eval_results['noisy_completion']}\n")
        
        print(f"\n✅ Robustness analysis completed successfully!")
        print(f"📂 Results saved to {output_dir}")
        
        return {
            "model_name": model_name,
            "noise_type": noise_type,
            "perplexity_ratio": eval_results["perplexity_ratio"],
            "output_dir": output_dir
        }
        
    except Exception as e:
        print(f"❌ Error during robustness analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
