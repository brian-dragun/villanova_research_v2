#!/usr/bin/env python3
"""
Generate Report - Extreme Sensitivity Analysis (ESA) for Large Language Models

This script generates comprehensive reports from the results of various LLM sensitivity analyses.
It uses the ResultsAnalyzer class to process analysis outputs and create detailed reports
while keeping models separate for accurate testing and comparison.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import importlib.util
import importlib
import subprocess

# Get absolute paths for reliable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Print Python environment information
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")

# Check if necessary packages are available and install them if needed
required_packages = ['numpy', 'pandas', 'matplotlib', 'seaborn']
missing_packages = []

for package in required_packages:
    try:
        importlib.import_module(package)
        print(f"✓ {package} is already installed")
    except ImportError:
        missing_packages.append(package)
        print(f"✗ {package} is not installed")

if missing_packages:
    print(f"\nAttempting to install missing packages: {', '.join(missing_packages)}")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("Packages installed successfully. Reloading modules...")
        for package in missing_packages:
            importlib.import_module(package)
            print(f"Successfully loaded {package} after installation")
    except Exception as e:
        print(f"Error installing packages: {e}")
        print("\nYou may need to manually install the missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)

# Try to load all required modules directly
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("Successfully imported all visualization dependencies")
except ImportError as e:
    print(f"Error importing visualization modules: {e}")
    print("This may indicate an environment configuration issue.")
    sys.exit(1)

# Import ResultsAnalyzer using importlib.util for more reliable importing
results_analyzer_path = os.path.join(current_dir, 'reporting', 'results_analyzer.py')
if os.path.exists(results_analyzer_path):
    print(f"Found results_analyzer.py at: {results_analyzer_path}")
    try:
        spec = importlib.util.spec_from_file_location("results_analyzer", results_analyzer_path)
        results_analyzer_module = importlib.util.module_from_spec(spec)
        sys.modules["results_analyzer"] = results_analyzer_module
        spec.loader.exec_module(results_analyzer_module)
        ResultsAnalyzer = results_analyzer_module.ResultsAnalyzer
        print("Successfully imported ResultsAnalyzer class")
    except Exception as e:
        print(f"Error importing ResultsAnalyzer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print(f"Error: Could not find results_analyzer.py at: {results_analyzer_path}")
    sys.exit(1)

# Default models if not found in config
DEFAULT_MODELS = [
    "gpt-neo-125m", "gpt-neo-1.3B", "gpt-neo-2.7B", "gpt-j-6B", "gpt-neox-20B", 
    "opt-125m", "opt-1.3b", "opt-2.7b", "opt-6.7b", "opt-13b",
    "bloom-560m", "bloom-1b1", "bloom-1b7", "bloom-3b", "bloom-7b1"
]

# Try to import models from config, fall back to defaults if not available
try:
    config_path = os.path.join(current_dir, 'config.py')
    if os.path.exists(config_path):
        print(f"Found config.py at: {config_path}")
        try:
            spec = importlib.util.spec_from_file_location("config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Check if MODELS exists in the config
            if hasattr(config_module, 'MODELS'):
                MODELS = config_module.MODELS
                print(f"Loaded models from config: {len(MODELS)} models found")
            else:
                print("Config file doesn't contain MODELS variable, using defaults")
                MODELS = DEFAULT_MODELS
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default models")
            MODELS = DEFAULT_MODELS
    else:
        print("Config file not found, using default models")
        MODELS = DEFAULT_MODELS
except Exception:
    print("Error accessing config, using default models")
    MODELS = DEFAULT_MODELS


def generate_complete_report(outputs_dir, report_dir=None, formats=None, models=None, include_viz=True):
    """
    Generate comprehensive reports for specified models using the ResultsAnalyzer.
    
    Args:
        outputs_dir (str): Directory containing analysis outputs
        report_dir (str, optional): Directory to save generated reports
        formats (list, optional): Output formats to generate ("text", "json", "html")
        models (list, optional): List of models to analyze (all if None)
        include_viz (bool): Whether to include visualizations
    
    Returns:
        dict: Dictionary with paths to generated reports
    """
    # Default formats if not specified
    if formats is None:
        formats = ["html", "text"]
        
    # Create ResultsAnalyzer instance
    analyzer = ResultsAnalyzer(outputs_dir, report_dir)
    
    # Dictionary to store generated report paths
    generated_reports = {}
    
    # If models are specified, generate reports only for those models
    if models:
        for model in models:
            for output_format in formats:
                report = analyzer.generate_model_report(model, output_format)
                extension = "html" if output_format == "html" else "json" if output_format == "json" else "txt"
                
                # Create report directory if it doesn't exist
                report_path = os.path.join(report_dir or os.path.join(outputs_dir, "reports"), 
                                        f"{model}_report.{extension}")
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                
                with open(report_path, 'w') as f:
                    f.write(report)
                
                if model not in generated_reports:
                    generated_reports[model] = {}
                generated_reports[model][output_format] = report_path
        
        print(f"Generated reports for {len(models)} models: {', '.join(models)}")
    else:
        # Generate reports for all models
        analyzer.generate_reports(formats[0] if formats else "html", include_viz)
        
        # Add summary report to the generated reports
        for output_format in formats:
            summary_extension = "html" if output_format == "html" else "json" if output_format == "json" else "txt"
            summary_path = os.path.join(analyzer.report_dir, f"summary_report.{summary_extension}")
            
            if "summary" not in generated_reports:
                generated_reports["summary"] = {}
            generated_reports["summary"][output_format] = summary_path
                
        print(f"Generated reports for all models in {outputs_dir}")
    
    return generated_reports


def main():
    """
    Main function to parse arguments and generate reports.
    """
    parser = argparse.ArgumentParser(
        description="Generate comprehensive reports from ESA analysis results"
    )
    
    parser.add_argument(
        "--outputs_dir", 
        type=str, 
        default="./outputs",
        help="Directory containing analysis outputs"
    )
    
    parser.add_argument(
        "--report_dir", 
        type=str, 
        default=None,
        help="Directory to save generated reports (defaults to outputs_dir/reports)"
    )
    
    parser.add_argument(
        "--formats", 
        nargs="+", 
        choices=["text", "json", "html"], 
        default=["html"],
        help="Output formats for reports"
    )
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=None,
        help="Generate reports only for specified models"
    )
    
    parser.add_argument(
        "--no_viz", 
        action="store_true",
        help="Disable generation of visualizations"
    )
    
    args = parser.parse_args()
    
    # Normalize paths
    outputs_dir = os.path.abspath(args.outputs_dir)
    report_dir = os.path.abspath(args.report_dir) if args.report_dir else None
    
    # Generate reports
    start_time = datetime.now()
    print(f"Generating reports at {start_time.strftime('%Y-%m-%d %H:%M:%S')}...")
    
    reports = generate_complete_report(
        outputs_dir=outputs_dir,
        report_dir=report_dir,
        formats=args.formats,
        models=args.models,
        include_viz=not args.no_viz
    )
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"Report generation completed in {duration.total_seconds():.2f} seconds")
    print("Generated reports:")
    
    for model, model_reports in reports.items():
        for format_type, path in model_reports.items():
            print(f"- {model} ({format_type}): {path}")


if __name__ == "__main__":
    main()