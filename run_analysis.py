#!/usr/bin/env python3
"""
Run Analysis Script

This script makes it easy to run different types of analyses on various models.
"""

import argparse
import os
import sys
import importlib
from datetime import datetime

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import AVAILABLE_MODELS, OUTPUT_DIR

def list_available_models():
    """List all available models with their details."""
    print("\n🔍 Available Models:")
    print("-" * 80)
    print(f"{'Key':<15} | {'Name':<35} | {'Auth Required':<12} | Details")
    print("-" * 80)
    
    for key, details in AVAILABLE_MODELS.items():
        print(f"{key:<15} | {details['name']:<35} | {'Yes' if details['requires_auth'] else 'No':<12} | {details['details']}")
    
    print("-" * 80)

def list_available_analyses():
    """List all available analysis types."""
    analyses = {
        "sensitivity": {
            "module": "llm_weight_sensitivity_analysis",
            "description": "Analyze weight sensitivity in the model"
        },
        "pruning": {
            "module": "llm_prune_model",
            "description": "Prune model weights based on sensitivity analysis"
        },
        "robustness": {
            "module": "llm_robustness_test",
            "description": "Test model robustness under various conditions"
        },
        "adversarial": {
            "module": "llm_adversarial_test",
            "description": "Run adversarial attacks against the model"
        },
        "bit-level": {
            "module": "llm_bit_level_and_ablation_analysis",
            "description": "Perform bit-level and ablation analysis"
        },
        "integrated": {
            "module": "llm_integrated_analysis",
            "description": "Run comprehensive integrated analysis on model"
        },
        "evaluate": {
            "module": "llm_evaluate_models",
            "description": "Evaluate model performance on benchmark tasks"
        },
        "super-weights": {
            "module": "llm_super_weights",
            "description": "Perform super-weights analysis"
        },
        "compare-sensitivity": {
            "module": "llm_analyze_sensitivity",
            "description": "Compare sensitivity across models"
        },
        "targeted-perturbation": {
            "module": "llm_super_weights",
            "description": "Run targeted perturbation analysis"
        }
    }
    
    print("\n📊 Available Analysis Types:")
    print("-" * 80)
    print(f"{'Type':<15} | {'Module':<35} | Description")
    print("-" * 80)
    
    for key, details in analyses.items():
        print(f"{key:<15} | {details['module']:<35} | {details['description']}")
    
    print("-" * 80)
    
    return analyses

def main():
    parser = argparse.ArgumentParser(description='Run analyses on language models')
    parser.add_argument('--model', type=str, default='gpt-neo-125m',
                        help='Model key or full Hugging Face model identifier')
    parser.add_argument('--analysis', type=str, default='sensitivity',
                        help='Type of analysis to run')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (defaults to outputs/ANALYSIS_MODEL_DATE)')
    parser.add_argument('--list-models', action='store_true',
                        help='List available model shortcuts')
    parser.add_argument('--list-analyses', action='store_true',
                        help='List available analysis types')
    parser.add_argument('--skip-finetune', action='store_true',
                        help='Skip fine-tuning if model directory does not exist')
    parser.add_argument('--method', type=str, default=None,
                        help='Method to use for analysis (e.g., gradient, z_score, hessian, integrated)')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated list of methods for comparison analyses')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Sensitivity threshold for super weight identification (0.0-1.0)')
    parser.add_argument('--use-cache', action='store_true',
                        help='Use cached results when available to speed up analysis')
    parser.add_argument('--parallel', action='store_true', default=True,
                        help='Use parallel processing for large models')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
        
    if args.list_analyses:
        analyses = list_available_analyses()
        return
    
    # Determine the module to import based on analysis type
    analyses = {
        "sensitivity": "llm_weight_sensitivity_analysis",
        "pruning": "llm_prune_model",
        "robustness": "llm_robustness_test",
        "adversarial": "llm_adversarial_test",
        "bit-level": "llm_bit_level_and_ablation_analysis",
        "integrated": "llm_integrated_analysis",
        "evaluate": "llm_evaluate_models",
        "super-weights": "llm_super_weights",
        "compare-sensitivity": "llm_analyze_sensitivity",
        "targeted-perturbation": "llm_super_weights"
    }
    
    if args.analysis not in analyses:
        print(f"\n❌ Unknown analysis type: {args.analysis}")
        list_available_analyses()
        return
    
    module_name = analyses[args.analysis]
    
    # Create output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model.replace("/", "_")
        args.output_dir = os.path.join(OUTPUT_DIR, f"{args.analysis}_{model_name}_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n📂 Output directory: {args.output_dir}")
    
    # Set model and skip_finetune in local environment
    os.environ["MODEL_NAME"] = args.model
    os.environ["SKIP_FINETUNE"] = "1" if args.skip_finetune else "0"
    os.environ["OUTPUT_DIR"] = args.output_dir
    
    print(f"\n🚀 Running {args.analysis} analysis on model: {args.model}")
    print(f"📊 Using module: {module_name}")
    
    try:
        # Import and run the analysis module
        analysis_module = importlib.import_module(module_name)
        
        # Set environment variables for the analysis module to use
        os.environ["MODEL_NAME"] = args.model
        os.environ["OUTPUT_DIR"] = args.output_dir
        
        # Check if the module has a main function with the expected parameters
        if hasattr(analysis_module, "main"):
            # Try to inspect the function signature to determine parameters
            import inspect
            sig = inspect.signature(analysis_module.main)
            param_names = list(sig.parameters.keys())
            
            # Prepare keyword arguments based on what's available
            kwargs = {}
            if "model_name" in param_names:
                kwargs["model_name"] = args.model
            if "output_dir" in param_names:
                kwargs["output_dir"] = args.output_dir
            if "method" in param_names and args.method is not None:
                kwargs["method"] = args.method
            if "methods" in param_names and args.methods is not None:
                kwargs["methods"] = args.methods
            if "threshold" in param_names and args.threshold is not None:
                kwargs["threshold"] = args.threshold
            if "use_cache" in param_names:
                kwargs["use_cache"] = args.use_cache
            if "parallel" in param_names:
                kwargs["parallel"] = args.parallel
            
            # Call the main function with appropriate parameters
            if kwargs:
                analysis_module.main(**kwargs)
            else:
                # Just call the main function without parameters
                analysis_module.main()
        elif hasattr(analysis_module, "run_analysis"):
            # Prepare kwargs for run_analysis
            kwargs = {
                "model_name": args.model,
                "output_dir": args.output_dir
            }
            
            # Add optional parameters if they're provided
            if args.method is not None:
                kwargs["method"] = args.method
            if args.methods is not None:
                kwargs["methods"] = args.methods
            if args.threshold is not None:
                kwargs["threshold"] = args.threshold
            kwargs["use_cache"] = args.use_cache
            kwargs["parallel"] = args.parallel
                
            # Call run_analysis with the kwargs
            analysis_module.run_analysis(**kwargs)
        else:
            print(f"\n⚠️ Could not find main() or run_analysis() in {module_name}")
            print("Running the module directly instead...")
            
        print(f"\n✅ Analysis completed successfully!")
        print(f"📂 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error running analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
