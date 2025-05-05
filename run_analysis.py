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
        "selective-pruning": {
            "module": "selective_weight_pruning",
            "description": "Advanced pruning with fine-grained control over which weights to remove"
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
    print(f"{'Type':<20} | {'Module':<35} | Description")
    print("-" * 80)
    
    for key, details in analyses.items():
        print(f"{key:<20} | {details['module']:<35} | {details['description']}")
    
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
    parser.add_argument('--interactive', action='store_true',
                        help='Use interactive mode for selective pruning')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Force CPU-only mode for model loading and inference')
    
    # Additional arguments for selective weight pruning
    parser.add_argument('--prune-method', type=str, default='zero',
                        choices=['zero', 'mean', 'small_noise'],
                        help='Pruning method to apply (for pruning analyses)')
    parser.add_argument('--layers', type=str, default='all',
                        help='Comma-separated list of layers to target (for pruning analyses)')
    parser.add_argument('--component', type=str, default=None,
                        help='Specific component to target (query, key, value, etc. for pruning)')
    parser.add_argument('--percentile', type=float, default=100,
                        help='Top percentile to select for pruning (0-100)')
    parser.add_argument('--max-per-layer', type=int, default=1000,
                        help='Maximum weights to prune per layer')
    parser.add_argument('--sensitivity-file', type=str, default=None,
                        help='Path to pre-computed sensitivity data (for pruning)')
    
    # Arguments for robustness testing
    parser.add_argument('--noise-type', type=str, default='gaussian',
                        choices=['gaussian', 'uniform', 'targeted'],
                        help='Type of noise to use for robustness testing')
    parser.add_argument('--noise-level', type=float, default=0.01,
                        help='Noise level for robustness testing (0.0-1.0)')
    parser.add_argument('--attack-type', type=str, default='gradient',
                        choices=['gradient', 'random', 'token_swap', 'char_level'],
                        help='Type of attack for adversarial testing')
    
    # Arguments for bit-level analysis
    parser.add_argument('--bits', type=str, default=None,
                        help='Comma-separated list of bit positions to flip (for bit-level analysis)')
    parser.add_argument('--quantize', action='store_true',
                        help='Perform quantization analysis')
    parser.add_argument('--quantize-bits', type=int, default=8,
                        help='Number of bits to use for quantization analysis')
    
    # Arguments for model evaluation
    parser.add_argument('--eval-tasks', type=str, default='all',
                        help='Comma-separated list of evaluation tasks (for evaluate)')
    parser.add_argument('--compare-with', type=str, default=None,
                        help='Model to compare with (for comparative analyses)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom prompt for testing')
    parser.add_argument('--generate-samples', type=int, default=5,
                        help='Number of text samples to generate when evaluating')
    parser.add_argument('--max-tokens', type=int, default=100,
                        help='Maximum tokens to generate when testing')
    
    # Arguments for super weights analysis
    parser.add_argument('--z-threshold', type=float, default=2.5,
                        help='Z-score threshold for super weight identification')
    parser.add_argument('--top-k', type=int, default=1000,
                        help='Top-K weights to consider in sensitivity analysis')
    parser.add_argument('--perturb-ratio', type=float, default=0.01,
                        help='Ratio of weights to perturb in targeted perturbation')
    
    # Arguments for integrated analysis
    parser.add_argument('--run-all', action='store_true',
                        help='Run all analysis types in sequence')
    parser.add_argument('--save-weights', action='store_true',
                        help='Save modified weights after analysis')
    parser.add_argument('--compute-metrics', action='store_true',
                        help='Compute additional metrics during analysis')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations for results')
    
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
        "selective-pruning": "selective_weight_pruning",
        "robustness": "llm_robustness_test",
        "adversarial": "llm_adversarial_test",
        "bit-level": "llm_bit_level_and_ablation_analysis",
        "integrated": "llm_integrated_analysis",
        "evaluate": "llm_evaluate_models",
        "super-weights": "llm_super_weights",
        "compare-sensitivity": "llm_analyze_sensitivity",
        "targeted-perturbation": "llm_super_weights",
        "run-model": "run_model"
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
    
    # Set CPU-only mode if requested
    if args.cpu_only:
        print("\n⚠️ Using CPU-only mode for model loading and inference")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["USE_CPU_ONLY"] = "1"
    
    print(f"\n🚀 Running {args.analysis} analysis on model: {args.model}")
    print(f"📊 Using module: {module_name}")
    
    try:
        # Import and run the analysis module
        analysis_module = importlib.import_module(module_name)
        
        # Set environment variables for the analysis module to use
        os.environ["MODEL_NAME"] = args.model
        os.environ["OUTPUT_DIR"] = args.output_dir
        
        # Special handling for selective pruning if interactive mode is enabled
        if args.analysis == "selective-pruning" and args.interactive:
            print("\n🔍 Running selective pruning in interactive mode...")
            if hasattr(analysis_module, "run_selective_pruning"):
                # Create an args-like object with the interactive flag
                from types import SimpleNamespace
                pruning_args = SimpleNamespace(
                    model=args.model,
                    method=args.method or "gradient",
                    threshold=args.threshold,
                    percentile=args.percentile,
                    prune_method=args.prune_method,
                    layers=args.layers,
                    component=args.component,
                    max_per_layer=args.max_per_layer,
                    sensitivity_file=args.sensitivity_file,
                    no_parallel=not args.parallel,
                    evaluate=True,
                    interactive=True
                )
                analysis_module.run_selective_pruning(pruning_args)
            else:
                print("\n⚠️ Interactive mode not available for this module")
                sys.exit(1)
            return
        
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
            
            # Add all the new parameters we've added
            param_mapping = {
                "prune_method": args.prune_method,
                "layers": args.layers,
                "component": args.component,
                "percentile": args.percentile,
                "max_per_layer": args.max_per_layer,
                "sensitivity_file": args.sensitivity_file,
                "noise_type": args.noise_type,
                "noise_level": args.noise_level,
                "attack_type": args.attack_type,
                "bits": args.bits,
                "quantize": args.quantize,
                "quantize_bits": args.quantize_bits,
                "eval_tasks": args.eval_tasks,
                "compare_with": args.compare_with,
                "prompt": args.prompt,
                "generate_samples": args.generate_samples,
                "max_tokens": args.max_tokens,
                "z_threshold": args.z_threshold,
                "top_k": args.top_k,
                "perturb_ratio": args.perturb_ratio,
                "run_all": args.run_all,
                "save_weights": args.save_weights,
                "compute_metrics": args.compute_metrics,
                "visualize": args.visualize,
                "interactive": args.interactive,
                "cpu_only": args.cpu_only
            }
            
            # Add parameters that exist in the function signature
            for param, value in param_mapping.items():
                if param in param_names:
                    kwargs[param] = value
            
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
            param_mapping = {
                "method": args.method,
                "methods": args.methods,
                "threshold": args.threshold,
                "use_cache": args.use_cache,
                "parallel": args.parallel,
                "prune_method": args.prune_method,
                "layers": args.layers,
                "component": args.component,
                "percentile": args.percentile,
                "max_per_layer": args.max_per_layer,
                "sensitivity_file": args.sensitivity_file,
                "noise_type": args.noise_type,
                "noise_level": args.noise_level,
                "attack_type": args.attack_type,
                "bits": args.bits,
                "quantize": args.quantize,
                "quantize_bits": args.quantize_bits,
                "eval_tasks": args.eval_tasks,
                "compare_with": args.compare_with,
                "prompt": args.prompt,
                "generate_samples": args.generate_samples,
                "max_tokens": args.max_tokens,
                "z_threshold": args.z_threshold,
                "top_k": args.top_k,
                "perturb_ratio": args.perturb_ratio,
                "run_all": args.run_all,
                "save_weights": args.save_weights,
                "compute_metrics": args.compute_metrics,
                "visualize": args.visualize,
                "interactive": args.interactive,
                "cpu_only": args.cpu_only
            }
            
            # Add all parameters with non-None values (except use_cache and parallel which were handled separately)
            for param, value in param_mapping.items():
                if value is not None and param not in ["use_cache", "parallel"]:
                    kwargs[param] = value
                
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
