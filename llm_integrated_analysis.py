import os
os.environ["TORCH_USE_FLASH_ATTENTION"] = "0"
os.environ["TORCH_USE_EFFICIENT_ATTENTION"] = "0"
os.environ["LLAMA_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["PYTORCH_USE_SDPA"] = "0"

import torch
import time
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from llm_analyze_sensitivity import compute_hessian_sensitivity, plot_sensitivity
from llm_super_weights import identify_super_weights, plot_layerwise_superweights
from config import MODEL_NAME, TEST_PROMPT, get_model_by_key
from tqdm import tqdm

def run_integrated_analysis(model_name=None, input_text=TEST_PROMPT, output_dir=None):
    # Use default model if not specified
    if model_name is None:
        model_name = MODEL_NAME
    else:
        # Convert key to full model name if it's a short key
        model_name = get_model_by_key(model_name)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n📂 Output directory: {output_dir}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # 1) Load config and attempt to disable flash/efficient attention
    print(f"\n⏳ Loading model configuration for: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, "use_flash_attn"):
        config.use_flash_attn = False
    if hasattr(config, "rope_scaling"):
        config.rope_scaling = None
    
    # 2) Load model with updated config
    print(f"⏳ Loading model, please wait...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        config=config,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()
    
    print(f"\n🔍 Running integrated analysis on prompt: '{input_text}'")
    
    print(f"📊 Computing Hessian-based sensitivity scores...")
    try:
        sensitivity_scores = compute_hessian_sensitivity(model, input_text, device=device)
    except RuntimeError as e:
        print(f"❌ Error during integrated analysis on device {device}: {e}")
        # Attempt fallback to CPU
        if device.type == "cuda":
            print("⚠️ Falling back to CPU for integrated analysis...")
            model.to("cpu")
            try:
                sensitivity_scores = compute_hessian_sensitivity(model, input_text, device=torch.device("cpu"))
            except RuntimeError as e2:
                print(f"❌ Still failed on CPU: {e2}")
                print("⚠️ Skipping Hessian analysis.")
                sensitivity_scores = {}
        else:
            print("⚠️ Skipping Hessian analysis (no fallback).")
            sensitivity_scores = {}
    
    print("\n🔍 Identifying super weights (Z-score > 2.5)...")
    super_weights, layer_summary = identify_super_weights(model, z_threshold=2.5)
    
    # Save outputs to the specified directory
    if output_dir:
        # Save sensitivity visualization
        if sensitivity_scores:
            sensitivity_plot_path = os.path.join(output_dir, "hessian_sensitivity.png")
            plot_sensitivity(sensitivity_scores, output_file=sensitivity_plot_path)
            print(f"✅ Saved Hessian sensitivity plot to: {sensitivity_plot_path}")
        
        # Save super weights visualization
        superweights_plot_path = os.path.join(output_dir, "superweights_distribution.png")
        plot_layerwise_superweights(layer_summary, output_file=superweights_plot_path)
        print(f"✅ Saved super weights distribution plot to: {superweights_plot_path}")
    
    print("\n📈 Integrated Analysis of LLM Weight Importance:")
    for name in tqdm(sensitivity_scores, desc="Processing parameters", unit="param"):
        print(f"Parameter: {name}")
        print(f"  Hessian Sensitivity Score: {sensitivity_scores[name]:.4f}")
        if name in super_weights:
            print(f"  Super Weight Outlier Indices: {super_weights[name]}")
        else:
            print("  No super weight outliers detected.")
        print()
    
    return {
        "model_name": model_name,
        "device": str(device),
        "sensitivity_scores": sensitivity_scores,
        "super_weights": {k: list(v) for k, v in super_weights.items()},
        "output_dir": output_dir
    }

def main(model_name=None, output_dir=None, prompt=None):
    """
    Run integrated analysis combining multiple analysis methods.
    
    Args:
        model_name: Name of the model to analyze
        output_dir: Directory to save results
        prompt: Optional custom prompt to use for analysis
    
    Returns:
        Dictionary with analysis results
    """
    start_time = time.time()
    
    # Use default model if not specified
    if model_name is None:
        model_name = MODEL_NAME
    
    # Create output directory
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/integrated_{model_name.replace('/', '_')}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📂 Output directory: {output_dir}")
    
    # Use default or custom prompt
    input_text = prompt if prompt is not None else TEST_PROMPT
    
    print(f"\n🚀 Running integrated analysis for model: {model_name}")
    
    # Run the integrated analysis
    results = run_integrated_analysis(
        model_name=model_name,
        input_text=input_text,
        output_dir=output_dir
    )
    
    # Calculate execution time
    execution_time = time.time() - start_time
    print(f"\n✅ Integrated analysis completed in {execution_time:.1f} seconds")
    print(f"📂 Results saved to: {output_dir}")
    
    # Add execution time to results
    results["execution_time"] = execution_time
    
    return results

if __name__ == "__main__":
    main()