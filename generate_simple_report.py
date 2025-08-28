#!/usr/bin/env python3
"""
Generate Simple Report - Extreme Sensitivity Analysis (ESA) for Large Language Models

This is a simplified version of the report generator that avoids some of the more complex
analysis features that might cause errors in the ResultsAnalyzer class.
"""

import os
import sys
import json
import glob
import re
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import base64
import io
from collections import defaultdict

# Define a custom color palette for visualizations
CUSTOM_CMAP = LinearSegmentedColormap.from_list('custom_cmap', ['#2c3e50', '#3498db', '#2ecc71', '#f1c40f', '#e74c3c'])
ANALYSIS_COLORS = {
    'sensitivity': '#3498db',
    'super-weights': '#2ecc71', 
    'integrated': '#9b59b6',
    'compare-sensitivity': '#e74c3c',
    'selective-pruning': '#f1c40f',
    'selective_pruning': '#f1c40f',
    'evaluate': '#1abc9c'
}

def analyze_outputs_directory(outputs_dir):
    """
    Analyze the outputs directory and organize results by model and analysis type.
    
    Args:
        outputs_dir (str): Directory containing analysis outputs
        
    Returns:
        dict: Results organized by model and analysis type
    """
    outputs_dir = Path(outputs_dir)
    results = {}
    
    # Define analysis types
    analysis_types = {
        "sensitivity": "Basic Sensitivity Analysis",
        "super-weights": "Super Weights Identification", 
        "integrated": "Integrated Gradients Analysis",
        "compare-sensitivity": "Comparative Sensitivity Analysis",
        "selective-pruning": "Selective Weight Pruning",
        "selective_pruning": "Selective Weight Pruning",
        "evaluate": "Model Evaluation"
    }
    
    # Pattern to match directory names
    pattern = re.compile(r'^([a-zA-Z-_]+)_([a-zA-Z0-9-]+)_(\d+)_(\d+)$')
    
    for dir_path in outputs_dir.iterdir():
        if not dir_path.is_dir() or dir_path.name.startswith('_') or dir_path.name == "reports":
            continue
            
        match = pattern.match(dir_path.name)
        if match:
            analysis_type, model_name, date, time = match.groups()
            timestamp = f"{date}_{time}"
            
            if analysis_type in analysis_types:
                if model_name not in results:
                    results[model_name] = {}
                
                if analysis_type not in results[model_name]:
                    results[model_name][analysis_type] = []
                
                results[model_name][analysis_type].append({
                    "path": dir_path,
                    "timestamp": timestamp,
                    "datetime": datetime.strptime(f"{date} {time}", "%Y%m%d %H%M%S")
                })
    
    # Sort results by datetime for each model and analysis type
    for model in results:
        for analysis_type in results[model]:
            results[model][analysis_type].sort(key=lambda x: x["datetime"], reverse=True)
    
    return results, analysis_types

def load_json_file(file_path):
    """
    Load and parse a JSON file.
    
    Args:
        file_path (Path): Path to JSON file
        
    Returns:
        dict: Parsed JSON data or empty dict if error
    """
    try:
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            print(f"File not found: {file_path}")
            return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}")
        return {}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def extract_metrics_from_results(model_results):
    """
    Extract key metrics from analysis results for a model.
    
    Args:
        model_results (dict): Results for a model organized by analysis type
        
    Returns:
        dict: Dictionary of extracted metrics
    """
    metrics = defaultdict(dict)
    
    # Extract sensitivity metrics
    if 'sensitivity' in model_results and model_results['sensitivity']:
        result_path = model_results['sensitivity'][0]['path']
        layer_sensitivity_path = result_path / "layer_sensitivity.json"
        
        if layer_sensitivity_path.exists():
            data = load_json_file(layer_sensitivity_path)
            if "layer_sensitivities" in data:
                # Calculate average sensitivity
                sensitivities = [float(val) for val in data["layer_sensitivities"].values()]
                metrics["sensitivity"]["avg_sensitivity"] = np.mean(sensitivities) if sensitivities else 0
                metrics["sensitivity"]["max_sensitivity"] = max(sensitivities) if sensitivities else 0
                metrics["sensitivity"]["min_sensitivity"] = min(sensitivities) if sensitivities else 0
                
                # Find top 3 most sensitive layers
                sorted_layers = sorted(data["layer_sensitivities"].items(), key=lambda x: float(x[1]), reverse=True)
                metrics["sensitivity"]["top_layers"] = sorted_layers[:3] if len(sorted_layers) >= 3 else sorted_layers
    
    # Extract pruning metrics
    if any(pt in model_results for pt in ['selective-pruning', 'selective_pruning']):
        prune_type = 'selective-pruning' if 'selective-pruning' in model_results else 'selective_pruning'
        if model_results[prune_type]:
            result_path = model_results[prune_type][0]['path']
            pruning_results_path = result_path / "pruning_results.json"
            
            if pruning_results_path.exists():
                data = load_json_file(pruning_results_path)
                if "accuracy_before" in data and "accuracy_after" in data:
                    metrics["pruning"]["accuracy_before"] = data["accuracy_before"]
                    metrics["pruning"]["accuracy_after"] = data["accuracy_after"]
                    metrics["pruning"]["accuracy_change"] = data["accuracy_after"] - data["accuracy_before"]
                    metrics["pruning"]["pruning_percentage"] = data.get("pruning_percentage", 0)
    
    # Extract evaluation metrics
    if 'evaluate' in model_results and model_results['evaluate']:
        result_path = model_results['evaluate'][0]['path']
        eval_results_path = result_path / "evaluation_results.json"
        
        if eval_results_path.exists():
            data = load_json_file(eval_results_path)
            if "metrics" in data:
                for metric, value in data["metrics"].items():
                    if isinstance(value, (int, float)):
                        metrics["evaluation"][metric] = value
    
    return metrics

def generate_comparative_charts(all_models_metrics, report_dir):
    """
    Generate comparative charts for models.
    
    Args:
        all_models_metrics (dict): Metrics for all models
        report_dir (Path): Directory to save charts
        
    Returns:
        dict: Paths to generated charts
    """
    charts = {}
    vis_dir = report_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True, parents=True)
    
    # Only proceed if we have metrics for multiple models
    if len(all_models_metrics) <= 1:
        return charts
    
    # Sensitivity comparison chart
    models_with_sensitivity = [model for model, metrics in all_models_metrics.items() 
                              if "sensitivity" in metrics and "avg_sensitivity" in metrics["sensitivity"]]
    
    if len(models_with_sensitivity) > 1:
        plt.figure(figsize=(10, 6))
        avg_sensitivities = [all_models_metrics[model]["sensitivity"]["avg_sensitivity"] 
                            for model in models_with_sensitivity]
        
        # Create bar chart
        bars = plt.bar(models_with_sensitivity, avg_sensitivities, color=ANALYSIS_COLORS['sensitivity'])
        plt.title("Average Layer Sensitivity Comparison", fontsize=14)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Average Sensitivity", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Save chart
        chart_path = vis_dir / "sensitivity_comparison.png"
        plt.savefig(chart_path)
        plt.close()
        charts["sensitivity_comparison"] = chart_path
    
    # Pruning impact chart
    models_with_pruning = [model for model, metrics in all_models_metrics.items() 
                          if "pruning" in metrics and "accuracy_change" in metrics["pruning"]]
    
    if len(models_with_pruning) > 1:
        plt.figure(figsize=(10, 6))
        accuracy_changes = [all_models_metrics[model]["pruning"]["accuracy_change"] 
                           for model in models_with_pruning]
        
        # Create bar chart with colors based on positive/negative changes
        bars = plt.bar(models_with_pruning, accuracy_changes, 
                      color=[ANALYSIS_COLORS['selective-pruning'] if ac >= 0 else '#e74c3c' 
                            for ac in accuracy_changes])
        
        plt.title("Pruning Impact on Accuracy", fontsize=14)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Accuracy Change After Pruning", fontsize=12)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + 0.001 if height >= 0 else height - 0.005,
                    f'{height:.4f}', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=10)
        
        # Save chart
        chart_path = vis_dir / "pruning_impact_comparison.png"
        plt.savefig(chart_path)
        plt.close()
        charts["pruning_impact_comparison"] = chart_path
    
    # Evaluation metrics comparison
    all_eval_metrics = set()
    for model, metrics in all_models_metrics.items():
        if "evaluation" in metrics:
            all_eval_metrics.update(metrics["evaluation"].keys())
    
    for metric in all_eval_metrics:
        models_with_metric = [model for model, metrics in all_models_metrics.items() 
                            if "evaluation" in metrics and metric in metrics["evaluation"]]
        
        if len(models_with_metric) > 1:
            plt.figure(figsize=(10, 6))
            metric_values = [all_models_metrics[model]["evaluation"][metric] 
                           for model in models_with_metric]
            
            # Create bar chart
            bars = plt.bar(models_with_metric, metric_values, color=ANALYSIS_COLORS['evaluate'])
            plt.title(f"{metric.replace('_', ' ').title()} Comparison", fontsize=14)
            plt.xlabel("Model", fontsize=12)
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=10)
            
            # Save chart
            safe_metric_name = metric.replace(' ', '_').lower()
            chart_path = vis_dir / f"{safe_metric_name}_comparison.png"
            plt.savefig(chart_path)
            plt.close()
            charts[f"{safe_metric_name}_comparison"] = chart_path
    
    return charts

def plot_to_base64(plt_figure):
    """
    Convert matplotlib figure to base64 encoded string for embedding in HTML.
    
    Args:
        plt_figure: Matplotlib figure
        
    Returns:
        str: Base64 encoded string of the image
    """
    buf = io.BytesIO()
    plt_figure.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def generate_layer_heatmap(model, model_results):
    """
    Generate a heatmap of layer sensitivities if data is available.
    
    Args:
        model (str): Model name
        model_results (dict): Results for this model
        
    Returns:
        str: Base64 encoded image or None if not available
    """
    if 'sensitivity' not in model_results or not model_results['sensitivity']:
        return None
        
    result_path = model_results['sensitivity'][0]['path']
    layer_sensitivity_path = result_path / "layer_sensitivity.json"
    
    if not layer_sensitivity_path.exists():
        return None
        
    data = load_json_file(layer_sensitivity_path)
    if "layer_sensitivities" not in data:
        return None
    
    # Organize layer data
    layer_data = [(layer, float(sensitivity)) for layer, sensitivity in data["layer_sensitivities"].items()]
    
    # Sort by layer name/number if possible
    try:
        layer_data.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    except ValueError:
        layer_data.sort(key=lambda x: x[0])
    
    layers, sensitivities = zip(*layer_data)
    
    # Create heatmap figure
    plt.figure(figsize=(12, 8))
    
    # If we have a lot of layers, reshape the data into a more square-like grid
    num_layers = len(layers)
    
    if num_layers > 20:
        # Calculate dimensions for a more square grid
        grid_size = int(np.ceil(np.sqrt(num_layers)))
        
        # Create 2D array for heatmap, fill with NaN for empty spots
        heatmap_data = np.full((grid_size, grid_size), np.nan)
        
        # Fill in the actual data
        for i, sens in enumerate(sensitivities):
            row, col = i // grid_size, i % grid_size
            heatmap_data[row, col] = sens
        
        # Generate custom labels for axes
        x_labels = [f"{i}" if i < num_layers % grid_size or i < num_layers else "" 
                   for i in range(grid_size)]
        y_labels = [f"{i*grid_size}" if i*grid_size < num_layers else "" 
                   for i in range(grid_size)]
        
        # Create heatmap
        ax = sns.heatmap(heatmap_data, cmap=CUSTOM_CMAP, annot=False, 
                        xticklabels=x_labels, yticklabels=y_labels,
                        cbar_kws={'label': 'Sensitivity'})
        
        plt.title(f"Layer Sensitivity Heatmap for {model}", fontsize=14)
        plt.xlabel("Layer Index (mod grid size)", fontsize=12)
        plt.ylabel("Layer Index (floor div grid size)", fontsize=12)
    
    else:
        # For fewer layers, create a simple 1D heatmap
        heatmap_data = np.array(sensitivities).reshape(1, -1)
        ax = sns.heatmap(heatmap_data, cmap=CUSTOM_CMAP, annot=True, fmt=".3f",
                        xticklabels=layers, yticklabels=["Sensitivity"],
                        cbar_kws={'label': 'Sensitivity'})
        
        plt.title(f"Layer Sensitivity for {model}", fontsize=14)
        plt.xlabel("Layer", fontsize=12)
        plt.xticks(rotation=90)
    
    plt.tight_layout()
    
    # Convert to base64
    img_str = plot_to_base64(plt)
    plt.close()
    
    return img_str

def generate_text_report(model, model_results, analysis_types, all_models_metrics=None):
    """
    Generate a text report for a specific model.
    
    Args:
        model (str): Model name
        model_results (dict): Results for this model
        analysis_types (dict): Mapping of analysis type codes to names
        all_models_metrics (dict, optional): Metrics for all models for comparison
        
    Returns:
        str: Text report
    """
    report = f"ANALYSIS REPORT FOR MODEL: {model}\n"
    report += "=" * 50 + "\n\n"
    
    # Add analysis sections
    for analysis_type, results_list in model_results.items():
        if not results_list:
            continue
            
        # Get the latest result
        latest_result = results_list[0]
        result_path = latest_result["path"]
        
        report += f"{analysis_types.get(analysis_type, analysis_type.upper())}\n"
        report += "-" * 30 + "\n"
        
        # List all files in the directory
        report += "Available result files:\n"
        for file_path in result_path.glob("*"):
            if file_path.is_file():
                report += f"- {file_path.name} ({file_path.stat().st_size} bytes)\n"
        
        # Try to load some common result files
        for filename in ["analysis_report.txt", "evaluation_results.json", "layer_sensitivity.json",
                         "pruning_results.json", "super_weights_gradient.json"]:
            file_path = result_path / filename
            if file_path.exists() and file_path.is_file():
                if file_path.suffix == ".txt":
                    try:
                        with open(file_path, 'r') as f:
                            report += f"\nContents of {filename}:\n"
                            report += "-" * 20 + "\n"
                            report += f.read()
                            report += "\n"
                    except Exception as e:
                        report += f"\nError reading {filename}: {e}\n"
                elif file_path.suffix == ".json":
                    try:
                        data = load_json_file(file_path)
                        report += f"\nSummary of {filename}:\n"
                        report += "-" * 20 + "\n"
                        
                        if isinstance(data, dict):
                            for key, value in list(data.items())[:5]:  # Show first 5 items only
                                report += f"  {key}: {str(value)[:100]}...(truncated)\n"
                            if len(data) > 5:
                                report += f"  ... and {len(data) - 5} more items\n"
                    except Exception as e:
                        report += f"\nError processing {filename}: {e}\n"
        
        report += "\n\n"
    
    # Add comparative section if metrics for multiple models are provided
    if all_models_metrics and len(all_models_metrics) > 1:
        report += "COMPARATIVE ANALYSIS\n"
        report += "-" * 30 + "\n"
        
        # Compare sensitivity metrics if available
        if "sensitivity" in all_models_metrics.get(model, {}):
            report += "Sensitivity Comparison:\n"
            
            # Sort models by average sensitivity
            models_by_sensitivity = sorted(
                [(m, metrics["sensitivity"].get("avg_sensitivity", 0)) 
                for m, metrics in all_models_metrics.items()
                if "sensitivity" in metrics and "avg_sensitivity" in metrics["sensitivity"]],
                key=lambda x: x[1],
                reverse=True
            )
            
            for i, (other_model, avg_sens) in enumerate(models_by_sensitivity):
                report += f"  {i+1}. {other_model}: {avg_sens:.4f}"
                if other_model == model:
                    report += " (this model)"
                report += "\n"
        
        # Compare pruning metrics if available
        if "pruning" in all_models_metrics.get(model, {}):
            report += "\nPruning Impact Comparison:\n"
            
            # Sort models by accuracy change
            models_by_pruning = sorted(
                [(m, metrics["pruning"].get("accuracy_change", 0)) 
                for m, metrics in all_models_metrics.items()
                if "pruning" in metrics and "accuracy_change" in metrics["pruning"]],
                key=lambda x: x[1],
                reverse=True
            )
            
            for i, (other_model, acc_change) in enumerate(models_by_pruning):
                report += f"  {i+1}. {other_model}: {acc_change:.4f}"
                if other_model == model:
                    report += " (this model)"
                report += "\n"
        
        # Compare evaluation metrics if available
        if "evaluation" in all_models_metrics.get(model, {}):
            all_eval_metrics = set()
            for m, metrics in all_models_metrics.items():
                if "evaluation" in metrics:
                    all_eval_metrics.update(metrics["evaluation"].keys())
            
            if all_eval_metrics:
                report += "\nEvaluation Metrics Comparison:\n"
                
                for metric in sorted(all_eval_metrics):
                    report += f"  {metric.replace('_', ' ').title()}:\n"
                    
                    # Sort models by this metric
                    models_by_metric = sorted(
                        [(m, metrics["evaluation"].get(metric, float('-inf'))) 
                        for m, metrics in all_models_metrics.items()
                        if "evaluation" in metrics and metric in metrics["evaluation"]],
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    for i, (other_model, value) in enumerate(models_by_metric):
                        if value != float('-inf'):
                            report += f"    {i+1}. {other_model}: {value:.4f}"
                            if other_model == model:
                                report += " (this model)"
                            report += "\n"
    
    return report

def generate_html_report(model, model_results, analysis_types, all_models_metrics=None, charts=None):
    """
    Generate an HTML report for a specific model.
    
    Args:
        model (str): Model name
        model_results (dict): Results for this model
        analysis_types (dict): Mapping of analysis type codes to names
        all_models_metrics (dict, optional): Metrics for all models for comparison
        charts (dict, optional): Dictionary of generated chart paths
        
    Returns:
        str: HTML report
    """
    html = f"""
    <html>
    <head>
        <title>Analysis Report for {model}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
            pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            .file-list {{ margin-left: 20px; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .metric-card {{ background-color: #f8f9fa; border: 1px solid #ddd; 
                           border-radius: 5px; padding: 15px; margin-bottom: 15px;
                           display: inline-block; width: 30%; margin-right: 2%; text-align: center; }}
            .metric-card h3 {{ margin-top: 0; color: #7f8c8d; }}
            .metric-card .value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-comparison {{ margin-top: 20px; }}
            .toggle-section {{ cursor: pointer; }}
            .toggle-section::before {{ content: '▶ '; }}
            .toggle-section.active::before {{ content: '▼ '; }}
            .hidden {{ display: none; }}
            .chart-container {{ margin: 20px 0; }}
            .tabs {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
            .tabs button {{ background-color: inherit; float: left; border: none; outline: none;
                         cursor: pointer; padding: 14px 16px; transition: 0.3s; }}
            .tabs button:hover {{ background-color: #ddd; }}
            .tabs button.active {{ background-color: #3498db; color: white; }}
            .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; 
                        border-top: none; animation: fadeEffect 1s; }}
            @keyframes fadeEffect {{ from {{opacity: 0;}} to {{opacity: 1;}} }}
        </style>
        <script>
            function toggleSection(id) {{
                const section = document.getElementById(id);
                const header = document.querySelector('[onclick="toggleSection(\\'' + id + '\\')"]');
                if (section.classList.contains('hidden')) {{
                    section.classList.remove('hidden');
                    header.classList.add('active');
                }} else {{
                    section.classList.add('hidden');
                    header.classList.remove('active');
                }}
            }}
            
            function openTab(evt, tabName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].style.display = "none";
                }}
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }}
            
            // Default to opening the first tab
            window.onload = function() {{
                document.getElementsByClassName("tablinks")[0].click();
            }}
        </script>
    </head>
    <body>
        <h1>Analysis Report for Model: {model}</h1>
    """
    
    # Generate layer sensitivity heatmap if data available
    layer_heatmap = generate_layer_heatmap(model, model_results)
    
    # Extract metrics if available
    model_metrics = all_models_metrics.get(model, {}) if all_models_metrics else {}
    
    # Add summary metrics dashboard
    if model_metrics:
        html += '<div class="section"><h2>Summary Metrics</h2><div class="metrics-dashboard">'
        
        if "sensitivity" in model_metrics:
            sensitivity = model_metrics["sensitivity"]
            html += f"""
                <div class="metric-card">
                    <h3>Average Sensitivity</h3>
                    <div class="value">{sensitivity.get("avg_sensitivity", "N/A"):.4f}</div>
                </div>
            """
            
            if "max_sensitivity" in sensitivity:
                html += f"""
                    <div class="metric-card">
                        <h3>Maximum Sensitivity</h3>
                        <div class="value">{sensitivity.get("max_sensitivity", "N/A"):.4f}</div>
                    </div>
                """
        
        if "pruning" in model_metrics:
            pruning = model_metrics["pruning"]
            change = pruning.get("accuracy_change", 0)
            color = "green" if change >= 0 else "red"
            
            html += f"""
                <div class="metric-card">
                    <h3>Pruning Impact</h3>
                    <div class="value" style="color: {color};">
                        {change:.4f}
                    </div>
                </div>
            """
        
        if "evaluation" in model_metrics:
            eval_metrics = model_metrics["evaluation"]
            for metric, value in list(eval_metrics.items())[:2]:  # Show only first 2 eval metrics in dashboard
                html += f"""
                    <div class="metric-card">
                        <h3>{metric.replace('_', ' ').title()}</h3>
                        <div class="value">{value:.4f}</div>
                    </div>
                """
        
        html += '</div></div>'  # Close metrics-dashboard and section
    
    # Add tabs for different analysis types
    html += '<div class="section"><h2>Analysis Results</h2>'
    html += '<div class="tabs">'
    
    # Create tab buttons
    for i, (analysis_type, results_list) in enumerate(model_results.items()):
        if results_list:
            analysis_name = analysis_types.get(analysis_type, analysis_type.title())
            html += f'<button class="tablinks" onclick="openTab(event, \'{analysis_type}\')">{analysis_name}</button>'
    
    html += '</div>'  # Close tabs
    
    # Create tab content
    for analysis_type, results_list in model_results.items():
        if not results_list:
            continue
            
        # Get the latest result
        latest_result = results_list[0]
        result_path = latest_result["path"]
        
        html += f'<div id="{analysis_type}" class="tabcontent">'
        html += f'<h3>{analysis_types.get(analysis_type, analysis_type.title())}</h3>'
        html += f'<p><strong>Analysis date:</strong> {latest_result["datetime"].strftime("%Y-%m-%d %H:%M:%S")}</p>'
        html += f'<p><strong>Results directory:</strong> {result_path}</p>'
        
        # Add layer heatmap if this is sensitivity analysis and we have a heatmap
        if analysis_type == 'sensitivity' and layer_heatmap:
            html += '<h3>Layer Sensitivity Heatmap</h3>'
            html += f'<img src="data:image/png;base64,{layer_heatmap}" alt="Layer Sensitivity Heatmap">'
        
        # Add collapsible file list section
        html += '<h3 class="toggle-section" onclick="toggleSection(\'file-list-{}\')">Available Result Files</h3>'.format(analysis_type)
        html += '<div id="file-list-{}" class="hidden">'.format(analysis_type)
        html += '<ul class="file-list">'
        
        # List all files in the directory
        for file_path in result_path.glob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                size_str = f"{file_size / 1024:.1f} KB" if file_size >= 1024 else f"{file_size} bytes"
                html += f'<li>{file_path.name} ({size_str})</li>'
        
        html += '</ul></div>'  # Close file list
        
        # Include any images found in the directory
        image_files = [f for f in result_path.glob("*.png") or result_path.glob("*.jpg")]
        if image_files:
            html += '<h3 class="toggle-section active" onclick="toggleSection(\'visualizations-{}\')">Visualizations</h3>'.format(analysis_type)
            html += '<div id="visualizations-{}" class="chart-container">'.format(analysis_type)
            
            for img_path in image_files:
                img_rel_path = os.path.relpath(img_path, start=result_path.parent.parent)
                html += f'<div><img src="../../{img_rel_path}" alt="{img_path.name}"><p>{img_path.name}</p></div>'
            
            html += '</div>'  # Close visualizations
        
        # Add content from common result files
        for filename in ["analysis_report.txt", "evaluation_results.json", "layer_sensitivity.json",
                         "pruning_results.json", "super_weights_gradient.json"]:
            file_path = result_path / filename
            if file_path.exists() and file_path.is_file():
                section_id = f"{analysis_type}-{filename.replace('.', '-')}"
                html += f'<h3 class="toggle-section" onclick="toggleSection(\'{section_id}\')">{filename}</h3>'
                html += f'<div id="{section_id}" class="hidden">'
                
                if file_path.suffix == ".txt":
                    try:
                        with open(file_path, 'r') as f:
                            html += f"<pre>{f.read()}</pre>"
                    except Exception as e:
                        html += f"<p>Error reading {filename}: {e}</p>"
                elif file_path.suffix == ".json":
                    try:
                        data = load_json_file(file_path)
                        
                        if isinstance(data, dict):
                            html += "<table><tr><th>Key</th><th>Value</th></tr>"
                            for key, value in list(data.items())[:20]:  # Show first 20 items only
                                safe_value = str(value)[:150] + "..." if len(str(value)) > 150 else str(value)
                                html += f"<tr><td>{key}</td><td>{safe_value}</td></tr>"
                            html += "</table>"
                            if len(data) > 20:
                                html += f"<p>... and {len(data) - 20} more items</p>"
                    except Exception as e:
                        html += f"<p>Error processing {filename}: {e}</p>"
                
                html += '</div>'  # Close section
        
        html += '</div>'  # Close tab content
    
    html += '</div>'  # Close section
    
    # Add comparative section if metrics for multiple models are provided
    if all_models_metrics and len(all_models_metrics) > 1:
        html += '<div class="section"><h2>Comparative Analysis</h2>'
        
        # Reference charts if available
        if charts:
            for chart_name, chart_path in charts.items():
                chart_rel_path = os.path.relpath(chart_path, start=Path(os.path.dirname(chart_path)).parent)
                title = chart_name.replace('_', ' ').title().replace('Comparison', 'Comparison Across Models')
                html += f'<div class="chart-container"><h3>{title}</h3>'
                html += f'<img src="{chart_rel_path}" alt="{title}">'
                html += '</div>'
        
        # Compare sensitivity metrics if available
        if "sensitivity" in all_models_metrics.get(model, {}):
            html += '<div class="metric-comparison"><h3>Model Ranking by Average Layer Sensitivity</h3>'
            html += '<table><tr><th>Rank</th><th>Model</th><th>Average Sensitivity</th></tr>'
            
            # Sort models by average sensitivity
            models_by_sensitivity = sorted(
                [(m, metrics["sensitivity"].get("avg_sensitivity", 0)) 
                for m, metrics in all_models_metrics.items()
                if "sensitivity" in metrics and "avg_sensitivity" in metrics["sensitivity"]],
                key=lambda x: x[1],
                reverse=True
            )
            
            for i, (other_model, avg_sens) in enumerate(models_by_sensitivity):
                row_class = ' class="highlight"' if other_model == model else ''
                model_name = f"{other_model} (this model)" if other_model == model else other_model
                html += f'<tr{row_class}><td>{i+1}</td><td>{model_name}</td><td>{avg_sens:.4f}</td></tr>'
            
            html += '</table></div>'
        
        # Compare pruning metrics if available
        if "pruning" in all_models_metrics.get(model, {}):
            html += '<div class="metric-comparison"><h3>Model Ranking by Pruning Impact</h3>'
            html += '<table><tr><th>Rank</th><th>Model</th><th>Accuracy Change</th><th>Pruning %</th></tr>'
            
            # Sort models by accuracy change
            models_by_pruning = sorted(
                [(m, metrics["pruning"].get("accuracy_change", 0), metrics["pruning"].get("pruning_percentage", "N/A")) 
                for m, metrics in all_models_metrics.items()
                if "pruning" in metrics and "accuracy_change" in metrics["pruning"]],
                key=lambda x: x[1],
                reverse=True
            )
            
            for i, (other_model, acc_change, prune_pct) in enumerate(models_by_pruning):
                row_class = ' class="highlight"' if other_model == model else ''
                model_name = f"{other_model} (this model)" if other_model == model else other_model
                html += f'<tr{row_class}><td>{i+1}</td><td>{model_name}</td>'
                html += f'<td>{acc_change:.4f}</td><td>{prune_pct}</td></tr>'
            
            html += '</table></div>'
        
        html += '</div>'  # Close comparative section
    
    # Add JavaScript for interactivity
    html += """
    </body>
    </html>
    """
    
    return html

def generate_summary_report(results, analysis_types, report_dir, format="html", all_models_metrics=None, charts=None):
    """
    Generate a summary report comparing all models.
    
    Args:
        results (dict): Results for all models
        analysis_types (dict): Mapping of analysis type codes to names
        report_dir (Path): Directory to save the report
        format (str): Output format (html or text)
        all_models_metrics (dict, optional): Metrics for all models
        charts (dict, optional): Dictionary of generated chart paths
        
    Returns:
        str: Path to the generated report
    """
    models = list(results.keys())
    
    if format == "html":
        html = f"""
        <html>
        <head>
            <title>ESA Analysis Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .chart-container {{ margin: 20px 0; }}
                .dashboard {{ display: flex; flex-wrap: wrap; justify-content: space-between; margin: 20px 0; }}
                .stat-box {{ background-color: #f8f9fa; border-radius: 5px; padding: 15px; 
                           margin-bottom: 20px; width: 48%; text-align: center; }}
                .stat-box h3 {{ margin-top: 0; color: #7f8c8d; }}
                .stat-box .big-number {{ font-size: 36px; font-weight: bold; color: #2c3e50; }}
                .highlight {{ background-color: #e8f4f8; }}
            </style>
        </head>
        <body>
            <h1>ESA Analysis Summary Report</h1>
            <p><strong>Generated at:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        """
        
        # Add summary statistics dashboard
        html += '<div class="section"><h2>Summary Statistics</h2><div class="dashboard">'
        
        # Calculate total analyses performed
        total_analyses = sum(len(model_results) for model_results in results.values())
        html += f"""
            <div class="stat-box">
                <h3>Models Analyzed</h3>
                <div class="big-number">{len(models)}</div>
            </div>
            <div class="stat-box">
                <h3>Total Analyses Performed</h3>
                <div class="big-number">{total_analyses}</div>
            </div>
        """
        
        # Count analysis types
        analysis_counts = defaultdict(int)
        for model_results in results.values():
            for analysis_type in model_results:
                analysis_counts[analysis_type] += 1
        
        # Find most common analysis
        if analysis_counts:
            most_common = max(analysis_counts.items(), key=lambda x: x[1])
            html += f"""
                <div class="stat-box">
                    <h3>Most Common Analysis Type</h3>
                    <div class="big-number">{analysis_types.get(most_common[0], most_common[0])}</div>
                    <p>Performed on {most_common[1]} models</p>
                </div>
            """
        
        # If metrics are available, add more stats
        if all_models_metrics:
            # Find model with highest average sensitivity
            sensitivity_models = [(m, metrics["sensitivity"].get("avg_sensitivity", 0))
                                 for m, metrics in all_models_metrics.items() 
                                 if "sensitivity" in metrics and "avg_sensitivity" in metrics["sensitivity"]]
            
            if sensitivity_models:
                most_sensitive = max(sensitivity_models, key=lambda x: x[1])
                html += f"""
                    <div class="stat-box">
                        <h3>Most Sensitive Model</h3>
                        <div class="big-number">{most_sensitive[0]}</div>
                        <p>Average Sensitivity: {most_sensitive[1]:.4f}</p>
                    </div>
                """
            
            # Find model with best pruning outcome
            pruning_models = [(m, metrics["pruning"].get("accuracy_change", float('-inf')))
                            for m, metrics in all_models_metrics.items() 
                            if "pruning" in metrics and "accuracy_change" in metrics["pruning"]]
            
            if pruning_models:
                best_pruned = max(pruning_models, key=lambda x: x[1])
                if best_pruned[1] != float('-inf'):
                    html += f"""
                        <div class="stat-box">
                            <h3>Best Pruning Outcome</h3>
                            <div class="big-number">{best_pruned[0]}</div>
                            <p>Accuracy Change: {best_pruned[1]:.4f}</p>
                        </div>
                    """
        
        html += '</div></div>'  # Close dashboard and section
        
        # Add comparative charts if available
        if charts:
            html += '<div class="section"><h2>Comparative Analysis</h2>'
            
            for chart_name, chart_path in charts.items():
                chart_rel_path = os.path.relpath(chart_path, start=report_dir)
                title = chart_name.replace('_', ' ').title()
                html += f'<div class="chart-container"><h3>{title}</h3>'
                html += f'<img src="{chart_rel_path}" alt="{title}">'
                html += '</div>'
            
            html += '</div>'  # Close section
            
        # Add models section
        html += """
            <div class="section">
                <h2>Models Analyzed</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Analysis Types</th>
                        <th>Report</th>
                    </tr>
        """
        
        for model in models:
            analysis_types_available = ", ".join([analysis_types.get(at, at) 
                                                 for at in results[model].keys()])
            html += f"""
                    <tr>
                        <td>{model}</td>
                        <td>{analysis_types_available}</td>
                        <td><a href="{model}_report.html">View Report</a></td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Save the summary report
        report_path = report_dir / "summary_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        return str(report_path)
    
    else:  # Text format
        text = "ESA ANALYSIS SUMMARY REPORT\n"
        text += "=" * 50 + "\n"
        text += f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        text += "SUMMARY STATISTICS\n"
        text += "-" * 20 + "\n"
        text += f"Models Analyzed: {len(models)}\n"
        
        total_analyses = sum(len(model_results) for model_results in results.values())
        text += f"Total Analyses Performed: {total_analyses}\n"
        
        # Count analysis types
        analysis_counts = defaultdict(int)
        for model_results in results.values():
            for analysis_type in model_results:
                analysis_counts[analysis_type] += 1
        
        if analysis_counts:
            text += "\nAnalysis Types Performed:\n"
            for analysis_type, count in sorted(analysis_counts.items(), key=lambda x: x[1], reverse=True):
                text += f"  {analysis_types.get(analysis_type, analysis_type)}: {count} models\n"
        
        # Add comparative metrics if available
        if all_models_metrics and len(all_models_metrics) > 1:
            text += "\nCOMPARATIVE METRICS\n"
            text += "-" * 20 + "\n"
            
            # Sensitivity comparison
            sensitivity_models = [(m, metrics["sensitivity"].get("avg_sensitivity", 0))
                                for m, metrics in all_models_metrics.items() 
                                if "sensitivity" in metrics and "avg_sensitivity" in metrics["sensitivity"]]
            
            if sensitivity_models:
                text += "Average Layer Sensitivity Ranking:\n"
                for i, (model, sensitivity) in enumerate(sorted(sensitivity_models, key=lambda x: x[1], reverse=True)):
                    text += f"  {i+1}. {model}: {sensitivity:.4f}\n"
            
            # Pruning comparison
            pruning_models = [(m, metrics["pruning"].get("accuracy_change", float('-inf')))
                            for m, metrics in all_models_metrics.items() 
                            if "pruning" in metrics and "accuracy_change" in metrics["pruning"]]
            
            if pruning_models:
                valid_models = [(m, ac) for m, ac in pruning_models if ac != float('-inf')]
                if valid_models:
                    text += "\nPruning Impact Ranking:\n"
                    for i, (model, acc_change) in enumerate(sorted(valid_models, key=lambda x: x[1], reverse=True)):
                        text += f"  {i+1}. {model}: {acc_change:.4f}\n"
        
        text += "\nMODELS ANALYZED\n"
        text += "-" * 20 + "\n"
        for model in models:
            analysis_types_available = ", ".join([analysis_types.get(at, at) 
                                                for at in results[model].keys()])
            text += f"- {model}: {analysis_types_available}\n"
        
        text += "\nANALYSIS REPORTS\n"
        text += "-" * 20 + "\n"
        for model in models:
            text += f"- {model}: {model}_report.txt\n"
        
        # Save the summary report
        report_path = report_dir / "summary_report.txt"
        with open(report_path, 'w') as f:
            f.write(text)
        
        return str(report_path)

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
        choices=["text", "html"], 
        default=["html", "text"],
        help="Output formats for reports"
    )
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=None,
        help="Generate reports only for specified models"
    )
    
    parser.add_argument(
        "--no_charts", 
        action="store_true",
        help="Disable generation of comparison charts"
    )
    
    args = parser.parse_args()
    
    # Normalize paths
    outputs_dir = Path(os.path.abspath(args.outputs_dir))
    report_dir = Path(os.path.abspath(args.report_dir)) if args.report_dir else outputs_dir / "reports"
    report_dir.mkdir(exist_ok=True, parents=True)
    
    # Analyze outputs directory
    print(f"Analyzing outputs directory: {outputs_dir}")
    results, analysis_types = analyze_outputs_directory(outputs_dir)
    
    if not results:
        print(f"No analysis results found in {outputs_dir}")
        return
    
    # Filter models if specified
    models = args.models if args.models else list(results.keys())
    models = [m for m in models if m in results]
    
    if not models:
        print("No matching models found")
        return
    
    print(f"Generating reports for {len(models)} models: {', '.join(models)}")
    
    # Extract metrics for all models for comparative analysis
    all_models_metrics = {}
    for model in models:
        model_metrics = extract_metrics_from_results(results[model])
        if model_metrics:
            all_models_metrics[model] = model_metrics
    
    # Generate comparative charts if needed and not disabled
    charts = {}
    if not args.no_charts and len(models) > 1:
        charts = generate_comparative_charts(all_models_metrics, report_dir)
    
    # Generate reports
    start_time = datetime.now()
    print(f"Generating reports at {start_time.strftime('%Y-%m-%d %H:%M:%S')}...")
    
    generated_reports = {}
    
    for model in models:
        for output_format in args.formats:
            if output_format == "html":
                report = generate_html_report(model, results[model], analysis_types, 
                                             all_models_metrics, charts)
                report_path = report_dir / f"{model}_report.html"
            else:  # text
                report = generate_text_report(model, results[model], analysis_types,
                                            all_models_metrics)
                report_path = report_dir / f"{model}_report.txt"
            
            with open(report_path, 'w') as f:
                f.write(report)
            
            if model not in generated_reports:
                generated_reports[model] = {}
            generated_reports[model][output_format] = str(report_path)
    
    # Generate summary report
    for output_format in args.formats:
        summary_path = generate_summary_report(
            {model: results[model] for model in models},
            analysis_types,
            report_dir,
            output_format,
            all_models_metrics,
            charts
        )
        if "summary" not in generated_reports:
            generated_reports["summary"] = {}
        generated_reports["summary"][output_format] = summary_path
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"Report generation completed in {duration.total_seconds():.2f} seconds")
    print("Generated reports:")
    
    for model, model_reports in generated_reports.items():
        for format_type, path in model_reports.items():
            print(f"- {model} ({format_type}): {path}")


if __name__ == "__main__":
    main()