import torch
import math
import os
import json
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, TEST_PROMPT, get_model_by_key, get_model_config, is_model_cached

def evaluate_perplexity(model, tokenizer, prompt):
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    return math.exp(loss)

def layer_ablation_experiment(model, tokenizer, prompt, layers_to_ablate):
    """
    Zero out weights for each layer in layers_to_ablate, measure perplexity.
    """
    baseline = evaluate_perplexity(model, tokenizer, prompt)
    results = {}
    
    param_dict = dict(model.named_parameters())
    for layer_name in layers_to_ablate:
        if layer_name not in param_dict:
            print(f"Layer '{layer_name}' not found in model; skipping.")
            continue
        
        param = param_dict[layer_name]
        backup = param.detach().clone()
        with torch.no_grad():
            param.zero_()
        new_ppl = evaluate_perplexity(model, tokenizer, prompt)
        results[layer_name] = new_ppl
        with torch.no_grad():
            param.copy_(backup)
    
    return baseline, results

def save_results_to_file(output_dir, model_name, baseline, ablation_results):
    """Save analysis results to files in the output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results as JSON
    result_data = {
        "model_name": model_name,
        "baseline_perplexity": baseline,
        "ablation_results": {k: float(v) for k, v in ablation_results.items()}
    }
    
    with open(os.path.join(output_dir, "ablation_results.json"), "w") as f:
        json.dump(result_data, f, indent=2)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.bar(
        ['Baseline'] + list(ablation_results.keys()), 
        [baseline] + list(ablation_results.values())
    )
    plt.ylabel('Perplexity (lower is better)')
    plt.title(f'Layer Ablation Results for {model_name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_results.png"))
    
    # Save a report as text
    with open(os.path.join(output_dir, "analysis_report.txt"), "w") as f:
        f.write(f"SENSITIVITY ANALYSIS REPORT\n")
        f.write(f"========================\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Date: {os.popen('date').read().strip()}\n\n")
        f.write(f"Baseline perplexity: {baseline:.2f}\n\n")
        f.write(f"LAYER ABLATION RESULTS:\n")
        for layer, ppl in ablation_results.items():
            f.write(f"- Layer: {layer}\n")
            f.write(f"  Perplexity after ablation: {ppl:.2f}\n")
            f.write(f"  Impact: {ppl/baseline:.2f}x increase in perplexity\n\n")
    
    print(f"\nResults saved to {output_dir}")
    print(f"- Full report: {os.path.join(output_dir, 'analysis_report.txt')}")
    print(f"- Visualization: {os.path.join(output_dir, 'ablation_results.png')}")

def main(model_name=None, output_dir=None, cpu_only=False):
    # Check if model specified in environment or use default
    model_key = os.environ.get("MODEL_NAME", None)
    if model_key:
        # Try to use model_key as a key in AVAILABLE_MODELS
        from config import get_model_by_key
        model_name = get_model_by_key(model_key)
    else:
        model_name = MODEL_NAME
    
    print(f"\n📋 SENSITIVITY ANALYSIS")
    print(f"=======================")
    print(f"🔑 Model key: {model_key}")
    print(f"🔄 Full model path: {model_name}")
    
    # Check if the model exists locally
    cached, cache_type, cache_path = is_model_cached(model_name)
    if cached:
        print(f"\n💾 Using locally cached {cache_type} model: {cache_path}")
    
    test_prompts = [TEST_PROMPT]

    print(f"\n⏳ Loading model, please wait...")
    
    # Use model-specific configuration with local cache
    from config import DEFAULT_MODEL_KEY
    model_config = get_model_config(model_key if model_key else DEFAULT_MODEL_KEY)
    
    # Check for CPU-only mode from either function parameter or environment variable
    cpu_only = cpu_only or os.environ.get("USE_CPU_ONLY", "0") == "1"
    
    if cpu_only:
        print("⚠️ Using CPU-only mode (no CUDA acceleration)")
        # Remove GPU-specific options for CPU-only mode
        for key in ["device_map", "torch_dtype", "use_flash_attention", 
                   "use_flash_attention_2", "use_flash_attn", "use_memory_efficient_attention"]:
            if key in model_config:
                del model_config[key]
        
        # Force CPU device
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {device}")
        
        if device.type == "cuda":
            # Print GPU information
            gpu_props = torch.cuda.get_device_properties(device)
            print(f"🔥 GPU: {gpu_props.name} with {gpu_props.total_memory / (1024**3):.2f} GB memory")
    
    # Load model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **model_config)
        model.to(device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠️ Try setting up a Hugging Face token if the model requires authentication.")
        return

    # Decide layer names depending on model architecture
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower:
        print(f"\n🦙 Detected Llama model architecture")
        layers_to_ablate = ["model.layers.0.mlp.gate_proj.weight"]
    elif "gpt-j" in model_name_lower:
        print(f"\n🤖 Detected GPT-J model architecture")
        layers_to_ablate = ["transformer.h.0.mlp.fc_in.weight"]
    elif "gpt-neo" in model_name_lower:
        print(f"\n🤖 Detected GPT-Neo model architecture") 
        layers_to_ablate = ["transformer.h.0.mlp.c_fc.weight"]
    else:
        # Generic approach
        print(f"\n⚠️ Unknown model architecture, using generic approach")
        # Print the first few parameter names to help with debugging
        param_names = list(model.named_parameters())[:5]
        print(f"Sample parameters: {[name for name, _ in param_names]}")
        layers_to_ablate = [name for name, _ in param_names if "mlp" in name and "weight" in name][:1]
        if not layers_to_ablate:
            # Fallback to the first weight parameter
            layers_to_ablate = [name for name, _ in param_names if "weight" in name][:1]
    
    print(f"\n🔍 Testing sensitivity of layers: {layers_to_ablate}")

    # 1) Layer ablation experiment
    ablation_prompt = test_prompts[0]
    print(f"\n📝 Test prompt: '{ablation_prompt}'")
    baseline_ablation, ablation_results = layer_ablation_experiment(model, tokenizer, ablation_prompt, layers_to_ablate)
    print(f"\n📊 Baseline perplexity: {baseline_ablation:.2f}")
    print("📉 Layer Ablation Results:")
    for layer, ppl in ablation_results.items():
        print(f"Layer: {layer} -> Perplexity after ablation: {ppl:.2f} (Impact: {ppl/baseline_ablation:.2f}x)")

    # Save results if output directory is provided
    if output_dir:
        output_dir = os.path.abspath(output_dir)
        print(f"\n💾 Saving results to: {output_dir}")
        save_results_to_file(output_dir, model_name, baseline_ablation, ablation_results)
    else:
        print("\n⚠️ No output directory specified. Results will not be saved.")

    # Offer to create a data file with the model weights
    print("\n🧠 Model successfully analyzed.")
    print("ℹ️ In future runs, the model will be loaded from local cache")

if __name__ == "__main__":
    main()
