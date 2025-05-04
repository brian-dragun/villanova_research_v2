import os
os.environ["TORCH_USE_FLASH_ATTENTION"] = "0"
os.environ["TORCH_USE_EFFICIENT_ATTENTION"] = "0"
os.environ["LLAMA_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["PYTORCH_USE_SDPA"] = "0"

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from llm_analyze_sensitivity import compute_hessian_sensitivity, plot_sensitivity
from llm_super_weights import identify_super_weights, plot_layerwise_superweights
from config import MODEL_NAME, TEST_PROMPT
from tqdm import tqdm

def run_integrated_analysis(input_text=TEST_PROMPT):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1) Load config and attempt to disable flash/efficient attention
    config = AutoConfig.from_pretrained(MODEL_NAME)
    if hasattr(config, "use_flash_attn"):
        config.use_flash_attn = False
    if hasattr(config, "rope_scaling"):
        config.rope_scaling = None
    
    # 2) Load model with updated config
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        config=config,
        trust_remote_code=True
    )
    model.to(device)
    model.eval()
    
    print(f"Computing Hessian-based sensitivity scores on device: {device}")
    try:
        sensitivity_scores = compute_hessian_sensitivity(model, input_text, device=device)
    except RuntimeError as e:
        print(f"Error during integrated analysis on device {device}: {e}")
        # Attempt fallback to CPU
        if device.type == "cuda":
            print("Falling back to CPU for integrated analysis...")
            model.to("cpu")
            try:
                sensitivity_scores = compute_hessian_sensitivity(model, input_text, device=torch.device("cpu"))
            except RuntimeError as e2:
                print(f"Still failed on CPU: {e2}")
                print("Skipping Hessian analysis.")
                sensitivity_scores = {}
        else:
            print("Skipping Hessian analysis (no fallback).")
            sensitivity_scores = {}
    
    print("Identifying super weights (Z-score > 2.5)...")
    #super_weights = identify_super_weights(model, z_threshold=2.5)
    super_weights, layer_summary = identify_super_weights(model, z_threshold=2.5)
    plot_layerwise_superweights(layer_summary)
    
    print("\nIntegrated Analysis of LLM Weight Importance:")
    for name in tqdm(sensitivity_scores, desc="Processing parameters", unit="param"):
        print(f"Parameter: {name}")
        print(f"  Hessian Sensitivity Score: {sensitivity_scores[name]:.4f}")
        if name in super_weights:
            print(f"  Super Weight Outlier Indices: {super_weights[name]}")
        else:
            print("  No super weight outliers detected.")
        print()
    
    if sensitivity_scores:
        plot_sensitivity(sensitivity_scores)

if __name__ == "__main__":
    run_integrated_analysis()