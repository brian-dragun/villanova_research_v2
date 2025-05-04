#!/usr/bin/env python3
"""
Iteration Helper Script

This script provides an interactive CLI to help researchers iterate through
different models, analyses, and parameters in their LLM research.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import json
from datetime import datetime

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import AVAILABLE_MODELS, OUTPUT_DIR, DATA_DIR

# Available analysis types
ANALYSIS_TYPES = {
    "sensitivity": {
        "script": "llm_analyze_sensitivity.py",
        "description": "Analyze weight sensitivity of the model"
    },
    "robustness": {
        "script": "llm_robustness_test.py",
        "description": "Test model robustness against adversarial inputs"
    },
    "adversarial": {
        "script": "llm_adversarial_test.py",
        "description": "Run adversarial attacks on the model"
    },
    "pruning": {
        "script": "llm_prune_model.py",
        "description": "Prune the model based on sensitivity analysis"
    },
    "evaluate": {
        "script": "llm_evaluate_models.py",
        "description": "Evaluate model performance on benchmark tasks"
    },
    "ablation": {
        "script": "llm_bit_level_and_ablation_analysis.py",
        "description": "Perform bit-level and ablation analysis"
    },
    "integrated": {
        "script": "llm_integrated_analysis.py",
        "description": "Run integrated analysis combining multiple metrics"
    },
}

class IterationLogger:
    """Logs iteration history and settings"""
    
    def __init__(self):
        self.log_dir = OUTPUT_DIR / "iteration_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = self.log_dir / f"iteration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.iterations = []
    
    def add_iteration(self, iteration_data):
        """Add an iteration to the log"""
        iteration_data["timestamp"] = datetime.now().isoformat()
        self.iterations.append(iteration_data)
        self._save_log()
    
    def _save_log(self):
        """Save the log to disk"""
        with open(self.log_file, 'w') as f:
            json.dump({"iterations": self.iterations}, f, indent=2)
    
    def get_last_settings(self):
        """Get the settings from the last iteration"""
        if not self.iterations:
            return None
        return self.iterations[-1]


def display_menu(title, options):
    """Display a menu and return the user's selection"""
    print(f"\n{title}")
    print("=" * len(title))
    
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {option}")
    
    print(f"{len(options) + 1}. Exit")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (number): "))
            if 1 <= choice <= len(options) + 1:
                return choice
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")


def select_model():
    """Let the user select a model"""
    models = list(AVAILABLE_MODELS.keys())
    model_names = [f"{model} - {AVAILABLE_MODELS[model]['details']}" for model in models]
    
    choice = display_menu("Select a Model", model_names)
    
    if choice == len(models) + 1:
        return None
    
    return models[choice - 1]


def select_analysis_type():
    """Let the user select an analysis type"""
    analysis_types = list(ANALYSIS_TYPES.keys())
    analysis_descriptions = [f"{a_type} - {ANALYSIS_TYPES[a_type]['description']}" for a_type in analysis_types]
    
    choice = display_menu("Select Analysis Type", analysis_descriptions)
    
    if choice == len(analysis_types) + 1:
        return None
    
    return analysis_types[choice - 1]


def get_analysis_params(analysis_type):
    """Get parameters for the selected analysis type"""
    print(f"\nParameters for {analysis_type} analysis:")
    params = {}
    
    # Common parameters
    params["output_dir"] = input("Output directory name [default=auto]: ") or "auto"
    
    # Analysis-specific parameters
    if analysis_type == "sensitivity":
        params["layers"] = input("Layers to analyze (comma-separated, leave empty for all): ")
        params["num_samples"] = input("Number of samples [default=100]: ") or "100"
    
    elif analysis_type == "robustness":
        params["test_type"] = input("Test type (noise/semantic/structural) [default=all]: ") or "all"
        params["intensity"] = input("Test intensity (0-1) [default=0.1]: ") or "0.1"
    
    elif analysis_type == "pruning":
        params["method"] = input("Pruning method (magnitude/sensitivity) [default=sensitivity]: ") or "sensitivity"
        params["ratio"] = input("Pruning ratio (0-1) [default=0.3]: ") or "0.3"
    
    elif analysis_type == "evaluate":
        params["benchmarks"] = input("Benchmarks to run (comma-separated, leave empty for default): ")
    
    return params


def run_analysis(model, analysis_type, params):
    """Run the selected analysis with the given parameters"""
    script_path = ANALYSIS_TYPES[analysis_type]["script"]
    
    # Build command
    cmd = [sys.executable, script_path, "--model", model]
    
    # Add parameters
    for key, value in params.items():
        if key == "output_dir" and value == "auto":
            # Auto-generate output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            value = f"{model}_{analysis_type}_{timestamp}"
            params["output_dir"] = value  # Update the params for logging
        
        if value:  # Only add non-empty parameters
            cmd.extend([f"--{key.replace('_', '-')}", value])
    
    print(f"\n📊 Running {analysis_type} analysis on {model}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✅ Analysis completed successfully")
            print("\nOutput summary:")
            # Print the last few lines of output which usually contain the summary
            output_lines = result.stdout.strip().split('\n')
            print('\n'.join(output_lines[-15:]))
        else:
            print(f"\n❌ Analysis failed with error code {result.returncode}")
            print("Error output:")
            print(result.stderr)
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"\n❌ Error running analysis: {e}")
        return False


def continue_iteration_prompt():
    """Ask the user if they want to continue iterating"""
    print("\n" + "=" * 60)
    print("Continue to iterate?")
    print("=" * 60)
    options = [
        "Yes, with same model but different analysis",
        "Yes, with different model but same analysis type",
        "Yes, with completely new settings",
        "No, I'm done"
    ]
    
    choice = display_menu("Iteration Options", options)
    
    if choice == 1:
        return "same_model"
    elif choice == 2:
        return "same_analysis"
    elif choice == 3:
        return "new"
    else:
        return "exit"


def main():
    parser = argparse.ArgumentParser(description='Iterate through LLM research experiments')
    parser.add_argument('--continue-last', action='store_true', 
                        help='Continue from the last iteration settings')
    
    args = parser.parse_args()
    
    logger = IterationLogger()
    last_settings = None
    
    if args.continue_last:
        last_settings = logger.get_last_settings()
        if last_settings:
            print(f"\n🔄 Continuing from last iteration: {last_settings}")
        else:
            print("\n⚠️ No previous iterations found. Starting fresh.")
    
    iteration_count = 0
    current_model = None
    current_analysis_type = None
    
    # Main iteration loop
    while True:
        iteration_count += 1
        print(f"\n{'=' * 80}")
        print(f"🔍 Research Iteration #{iteration_count}")
        print(f"{'=' * 80}")
        
        # Determine model selection based on previous iteration
        if current_model is None or not (last_settings and iteration_count == 1):
            current_model = select_model()
            if current_model is None:
                break
        
        # Determine analysis type based on previous iteration
        if current_analysis_type is None or not (last_settings and iteration_count == 1):
            current_analysis_type = select_analysis_type()
            if current_analysis_type is None:
                break
        
        # Get parameters for the analysis
        params = get_analysis_params(current_analysis_type)
        
        # Log the iteration
        iteration_data = {
            "iteration": iteration_count,
            "model": current_model,
            "analysis_type": current_analysis_type,
            "parameters": params
        }
        
        # Run the analysis
        success = run_analysis(current_model, current_analysis_type, params)
        iteration_data["success"] = success
        
        # Log the results
        logger.add_iteration(iteration_data)
        
        # Ask if the user wants to continue
        next_action = continue_iteration_prompt()
        
        if next_action == "exit":
            break
        elif next_action == "same_model":
            current_analysis_type = None  # Reset just the analysis type
        elif next_action == "same_analysis":
            current_model = None  # Reset just the model
        else:  # next_action == "new"
            current_model = None
            current_analysis_type = None
    
    print("\n🏁 Iteration session complete!")
    print(f"Session log saved to: {logger.log_file}")


if __name__ == "__main__":
    main()