#!/usr/bin/env python3
"""
Results Analyzer - Extreme Sensitivity Analysis (ESA) for Large Language Models

This module analyzes the results in the output folders from various LLM sensitivity analyses
and creates comprehensive reports of the discoveries, ensuring that models are kept separate
for accurate testing and comparison.
"""

import os
import json
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any, Optional, Union

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import project modules
try:
    from config import MODELS
except ImportError:
    MODELS = ["gpt-neo-125m", "gpt-neo-1.3B", "gpt-neo-2.7B", "gpt-j-6B", "gpt-neox-20B", 
              "opt-125m", "opt-1.3b", "opt-2.7b", "opt-6.7b", "opt-13b",
              "bloom-560m", "bloom-1b1", "bloom-1b7", "bloom-3b", "bloom-7b1"]

class ResultsAnalyzer:
    """
    A class to analyze and report on the results of Extreme Sensitivity Analysis for LLMs.
    """
    
    ANALYSIS_TYPES = {
        "sensitivity": "Basic Sensitivity Analysis",
        "super-weights": "Super Weights Identification", 
        "integrated": "Integrated Gradients Analysis",
        "compare-sensitivity": "Comparative Sensitivity Analysis",
        "selective-pruning": "Selective Weight Pruning",
        "selective_pruning": "Selective Weight Pruning",
        "evaluate": "Model Evaluation"
    }
    
    def __init__(self, outputs_dir: str, report_dir: str = None):
        """
        Initialize the ResultsAnalyzer with paths to outputs and report directories.
        
        Args:
            outputs_dir: Path to the directory containing analysis outputs
            report_dir: Path to save generated reports (defaults to outputs_dir/reports)
        """
        self.outputs_dir = Path(outputs_dir)
        self.report_dir = Path(report_dir) if report_dir else self.outputs_dir / "reports"
        self.report_dir.mkdir(exist_ok=True, parents=True)
        
        # Dictionary to store results by model and analysis type
        self.results = defaultdict(lambda: defaultdict(list))
        
        # Scan outputs directory
        self.scan_outputs()
    
    def scan_outputs(self) -> None:
        """
        Scan the outputs directory for results, organizing them by model and analysis type.
        """
        pattern = re.compile(r'^([a-zA-Z-_]+)_([a-zA-Z0-9-]+)_(\d+)_(\d+)$')
        
        for dir_path in self.outputs_dir.iterdir():
            if not dir_path.is_dir() or dir_path.name.startswith('_') or dir_path.name == "reports":
                continue
                
            match = pattern.match(dir_path.name)
            if match:
                analysis_type, model_name, date, time = match.groups()
                timestamp = f"{date}_{time}"
                
                if analysis_type in self.ANALYSIS_TYPES:
                    self.results[model_name][analysis_type].append({
                        "path": dir_path,
                        "timestamp": timestamp,
                        "datetime": datetime.strptime(f"{date} {time}", "%Y%m%d %H%M%S")
                    })
        
        # Sort results by datetime for each model and analysis type
        for model in self.results:
            for analysis_type in self.results[model]:
                self.results[model][analysis_type].sort(key=lambda x: x["datetime"], reverse=True)
    
    def load_json_data(self, file_path: Path) -> dict:
        """
        Load JSON data from a file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary containing the JSON data
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_path}")
            return {}
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return {}
    
    def analyze_sensitivity_results(self, model: str, timestamp: str = None) -> dict:
        """
        Analyze basic sensitivity analysis results for a specific model.
        
        Args:
            model: Model name
            timestamp: Specific timestamp to analyze (uses latest if None)
            
        Returns:
            Dictionary with analysis results
        """
        analysis_type = "sensitivity"
        results = self._get_results_data(model, analysis_type, timestamp)
        if not results:
            return {}
            
        analysis = {}
        
        # Load layer sensitivity data if available
        layer_sensitivity_path = results["path"] / "layer_sensitivity.json"
        if layer_sensitivity_path.exists():
            layer_data = self.load_json_data(layer_sensitivity_path)
            
            # Find most sensitive layers
            if "layer_sensitivities" in layer_data:
                layer_sens = layer_data["layer_sensitivities"]
                sorted_layers = sorted(layer_sens.items(), key=lambda x: float(x[1]), reverse=True)
                analysis["most_sensitive_layers"] = [
                    {"layer": layer, "sensitivity": float(sens)} 
                    for layer, sens in sorted_layers[:5]
                ]
                analysis["avg_layer_sensitivity"] = np.mean([float(sens) for sens in layer_sens.values()])
        
        return analysis
    
    def analyze_super_weights_results(self, model: str, timestamp: str = None) -> dict:
        """
        Analyze super weights identification results for a specific model.
        
        Args:
            model: Model name
            timestamp: Specific timestamp to analyze (uses latest if None)
            
        Returns:
            Dictionary with analysis results
        """
        analysis_type = "super-weights"
        results = self._get_results_data(model, analysis_type, timestamp)
        if not results:
            return {}
            
        analysis = {}
        
        # Load super weights data if available
        super_weights_path = results["path"] / "super_weights_gradient.json"
        if super_weights_path.exists():
            sw_data = self.load_json_data(super_weights_path)
            
            if "super_weights" in sw_data:
                analysis["super_weight_count"] = len(sw_data["super_weights"])
                analysis["super_weight_stats"] = {
                    "top_5": sw_data["super_weights"][:5] if len(sw_data["super_weights"]) >= 5 else sw_data["super_weights"],
                    "distribution": self._calculate_distribution_by_layer(sw_data["super_weights"])
                }
        
        return analysis
    
    def analyze_pruning_results(self, model: str, timestamp: str = None) -> dict:
        """
        Analyze selective pruning results for a specific model.
        
        Args:
            model: Model name
            timestamp: Specific timestamp to analyze (uses latest if None)
            
        Returns:
            Dictionary with analysis results
        """
        # Check both naming conventions
        for analysis_type in ["selective-pruning", "selective_pruning"]:
            results = self._get_results_data(model, analysis_type, timestamp)
            if results:
                break
                
        if not results:
            return {}
            
        analysis = {}
        
        # Load pruning results data
        pruning_results_path = results["path"] / "pruning_results.json"
        if pruning_results_path.exists():
            prune_data = self.load_json_data(pruning_results_path)
            
            if "accuracy_before" in prune_data and "accuracy_after" in prune_data:
                analysis["pruning_impact"] = {
                    "accuracy_before": prune_data["accuracy_before"],
                    "accuracy_after": prune_data["accuracy_after"],
                    "accuracy_change": prune_data["accuracy_after"] - prune_data["accuracy_before"],
                    "pruning_percentage": prune_data.get("pruning_percentage", "N/A"),
                    "pruning_method": prune_data.get("method", "N/A")
                }
        
        return analysis
    
    def analyze_integrated_results(self, model: str, timestamp: str = None) -> dict:
        """
        Analyze integrated gradients analysis results for a specific model.
        
        Args:
            model: Model name
            timestamp: Specific timestamp to analyze (uses latest if None)
            
        Returns:
            Dictionary with analysis results
        """
        analysis_type = "integrated"
        results = self._get_results_data(model, analysis_type, timestamp)
        if not results:
            return {}
            
        analysis = {}
        
        # Load integrated gradients data
        integrated_path = results["path"] / "integrated_gradients.json"
        if integrated_path.exists():
            ig_data = self.load_json_data(integrated_path)
            
            if "attributions" in ig_data:
                # Extract top attributions
                if isinstance(ig_data["attributions"], dict):
                    # Convert dict to list of (key, value) pairs and sort
                    sorted_attrs = sorted(ig_data["attributions"].items(), 
                                          key=lambda x: abs(float(x[1])), 
                                          reverse=True)
                    analysis["top_attributions"] = [
                        {"parameter": param, "attribution": float(attr)} 
                        for param, attr in sorted_attrs[:10]
                    ]
                
                # Compute statistics
                if "attribution_statistics" in ig_data:
                    analysis["attribution_stats"] = ig_data["attribution_statistics"]
        
        return analysis
    
    def analyze_evaluation_results(self, model: str, timestamp: str = None) -> dict:
        """
        Analyze model evaluation results for a specific model.
        
        Args:
            model: Model name
            timestamp: Specific timestamp to analyze (uses latest if None)
            
        Returns:
            Dictionary with analysis results
        """
        analysis_type = "evaluate"
        results = self._get_results_data(model, analysis_type, timestamp)
        if not results:
            return {}
            
        analysis = {}
        
        # Load evaluation results
        eval_path = results["path"] / "evaluation_results.json"
        if eval_path.exists():
            eval_data = self.load_json_data(eval_path)
            
            if "metrics" in eval_data:
                analysis["evaluation_metrics"] = eval_data["metrics"]
            
            if "task_results" in eval_data:
                analysis["task_performance"] = eval_data["task_results"]
        
        return analysis
    
    def analyze_comparative_results(self, model: str, timestamp: str = None) -> dict:
        """
        Analyze comparative sensitivity results for a specific model.
        
        Args:
            model: Model name
            timestamp: Specific timestamp to analyze (uses latest if None)
            
        Returns:
            Dictionary with analysis results
        """
        analysis_type = "compare-sensitivity"
        results = self._get_results_data(model, analysis_type, timestamp)
        if not results:
            return {}
            
        analysis = {}
        
        # Load comparison data
        comparison_path = results["path"] / "sensitivity_comparison.json"
        if comparison_path.exists():
            comp_data = self.load_json_data(comparison_path)
            
            if "method_comparison" in comp_data:
                analysis["method_comparison"] = comp_data["method_comparison"]
                
                # Find best performing method
                if isinstance(comp_data["method_comparison"], dict):
                    best_method = max(comp_data["method_comparison"].items(), 
                                      key=lambda x: x[1].get("f1_score", 0) 
                                      if isinstance(x[1], dict) else 0)
                    analysis["best_method"] = {
                        "name": best_method[0],
                        "metrics": best_method[1]
                    }
        
        return analysis
    
    def _get_results_data(self, model: str, analysis_type: str, timestamp: str = None) -> dict:
        """
        Get results data for a specific model, analysis type and timestamp.
        
        Args:
            model: Model name
            analysis_type: Analysis type
            timestamp: Specific timestamp (uses latest if None)
            
        Returns:
            Dictionary with results data or empty dict if not found
        """
        if model not in self.results or analysis_type not in self.results[model]:
            return {}
            
        results_list = self.results[model][analysis_type]
        
        if not results_list:
            return {}
            
        if timestamp:
            for result in results_list:
                if result["timestamp"] == timestamp:
                    return result
            return {}
        else:
            # Return the latest result
            return results_list[0]
    
    def _calculate_distribution_by_layer(self, super_weights: List[dict]) -> dict:
        """
        Calculate the distribution of super weights across layers.
        
        Args:
            super_weights: List of super weights
            
        Returns:
            Dictionary with layer distribution
        """
        layer_counts = defaultdict(int)
        total = len(super_weights)
        
        for sw in super_weights:
            if isinstance(sw, dict) and "layer" in sw:
                layer = sw["layer"]
                layer_counts[layer] += 1
        
        return {
            layer: {
                "count": count,
                "percentage": (count / total) * 100 if total > 0 else 0
            } 
            for layer, count in layer_counts.items()
        }
    
    def generate_model_report(self, model: str, output_format: str = "text") -> str:
        """
        Generate a comprehensive report for a specific model.
        
        Args:
            model: Model name
            output_format: Output format ("text", "json", or "html")
            
        Returns:
            Report in the specified format
        """
        # Collect results from all analysis types
        model_results = {
            "model": model,
            "sensitivity": self.analyze_sensitivity_results(model),
            "super_weights": self.analyze_super_weights_results(model),
            "pruning": self.analyze_pruning_results(model),
            "integrated": self.analyze_integrated_results(model),
            "evaluation": self.analyze_evaluation_results(model),
            "comparative": self.analyze_comparative_results(model)
        }
        
        if output_format == "json":
            return json.dumps(model_results, indent=2)
        
        elif output_format == "html":
            # Generate basic HTML report
            html = f"<html><head><title>Report for {model}</title>"
            html += "<style>body{font-family:Arial;margin:40px;} .section{margin-bottom:30px;} "
            html += "table{border-collapse:collapse;width:100%;} "
            html += "th,td{border:1px solid #ddd;padding:8px;text-align:left;}"
            html += "th{background-color:#f2f2f2;}</style></head><body>"
            html += f"<h1>Analysis Report for Model: {model}</h1>"
            
            # Add sections for each analysis type
            html += self._generate_html_section("Basic Sensitivity Analysis", model_results["sensitivity"])
            html += self._generate_html_section("Super Weights Identification", model_results["super_weights"])
            html += self._generate_html_section("Pruning Results", model_results["pruning"])
            html += self._generate_html_section("Integrated Gradients Analysis", model_results["integrated"])
            html += self._generate_html_section("Model Evaluation", model_results["evaluation"])
            html += self._generate_html_section("Comparative Analysis", model_results["comparative"])
            
            html += "</body></html>"
            return html
        
        else:  # Default to text format
            report = f"ANALYSIS REPORT FOR MODEL: {model}\n"
            report += "=" * 50 + "\n\n"
            
            # Basic Sensitivity Analysis
            if model_results["sensitivity"]:
                report += "BASIC SENSITIVITY ANALYSIS\n"
                report += "-" * 30 + "\n"
                
                if "most_sensitive_layers" in model_results["sensitivity"]:
                    report += "Most Sensitive Layers:\n"
                    for idx, layer in enumerate(model_results["sensitivity"]["most_sensitive_layers"]):
                        report += f"  {idx+1}. {layer['layer']}: {layer['sensitivity']:.6f}\n"
                
                if "avg_layer_sensitivity" in model_results["sensitivity"]:
                    report += f"Average Layer Sensitivity: {model_results['sensitivity']['avg_layer_sensitivity']:.6f}\n"
                
                report += "\n"
            
            # Super Weights Identification
            if model_results["super_weights"]:
                report += "SUPER WEIGHTS IDENTIFICATION\n"
                report += "-" * 30 + "\n"
                
                if "super_weight_count" in model_results["super_weights"]:
                    report += f"Number of Super Weights Identified: {model_results['super_weights']['super_weight_count']}\n\n"
                
                if "super_weight_stats" in model_results["super_weights"]:
                    stats = model_results["super_weights"]["super_weight_stats"]
                    
                    if "top_5" in stats:
                        report += "Top 5 Super Weights:\n"
                        for idx, sw in enumerate(stats["top_5"]):
                            if isinstance(sw, dict):
                                layer = sw.get("layer", "unknown")
                                param = sw.get("parameter", "unknown")
                                sensitivity = sw.get("sensitivity", "N/A")
                                report += f"  {idx+1}. Layer: {layer}, Parameter: {param}, Sensitivity: {sensitivity}\n"
                    
                    if "distribution" in stats:
                        report += "\nDistribution by Layer:\n"
                        for layer, info in stats["distribution"].items():
                            report += f"  {layer}: {info['count']} weights ({info['percentage']:.2f}%)\n"
                
                report += "\n"
            
            # Pruning Results
            if model_results["pruning"]:
                report += "SELECTIVE PRUNING RESULTS\n"
                report += "-" * 30 + "\n"
                
                if "pruning_impact" in model_results["pruning"]:
                    impact = model_results["pruning"]["pruning_impact"]
                    report += f"Pruning Method: {impact.get('pruning_method', 'N/A')}\n"
                    report += f"Pruning Percentage: {impact.get('pruning_percentage', 'N/A')}\n"
                    report += f"Accuracy Before: {impact.get('accuracy_before', 'N/A')}\n"
                    report += f"Accuracy After: {impact.get('accuracy_after', 'N/A')}\n"
                    
                    if isinstance(impact.get('accuracy_change'), (int, float)):
                        change = impact['accuracy_change']
                        direction = "improvement" if change >= 0 else "degradation"
                        report += f"Accuracy Change: {abs(change):.4f} ({direction})\n"
                
                report += "\n"
            
            # Integrated Gradients
            if model_results["integrated"]:
                report += "INTEGRATED GRADIENTS ANALYSIS\n"
                report += "-" * 30 + "\n"
                
                if "top_attributions" in model_results["integrated"]:
                    report += "Top 10 Parameter Attributions:\n"
                    for idx, attr in enumerate(model_results["integrated"]["top_attributions"]):
                        report += f"  {idx+1}. {attr['parameter']}: {attr['attribution']:.6f}\n"
                
                if "attribution_stats" in model_results["integrated"]:
                    stats = model_results["integrated"]["attribution_stats"]
                    report += "\nAttribution Statistics:\n"
                    for stat, value in stats.items():
                        report += f"  {stat}: {value}\n"
                
                report += "\n"
            
            # Model Evaluation
            if model_results["evaluation"]:
                report += "MODEL EVALUATION\n"
                report += "-" * 30 + "\n"
                
                if "evaluation_metrics" in model_results["evaluation"]:
                    metrics = model_results["evaluation"]["evaluation_metrics"]
                    report += "Evaluation Metrics:\n"
                    for metric, value in metrics.items():
                        report += f"  {metric}: {value}\n"
                
                report += "\n"
            
            # Comparative Analysis
            if model_results["comparative"]:
                report += "COMPARATIVE SENSITIVITY ANALYSIS\n"
                report += "-" * 30 + "\n"
                
                if "method_comparison" in model_results["comparative"]:
                    report += "Method Comparison:\n"
                    methods = model_results["comparative"]["method_comparison"]
                    for method, metrics in methods.items():
                        report += f"  {method}:\n"
                        if isinstance(metrics, dict):
                            for metric, value in metrics.items():
                                report += f"    {metric}: {value}\n"
                
                if "best_method" in model_results["comparative"]:
                    best = model_results["comparative"]["best_method"]
                    report += f"\nBest Performing Method: {best['name']}\n"
                
                report += "\n"
            
            return report
    
    def _generate_html_section(self, title: str, data: dict) -> str:
        """
        Generate HTML for a section of the report.
        
        Args:
            title: Section title
            data: Section data
            
        Returns:
            HTML string for the section
        """
        if not data:
            return ""
            
        html = f'<div class="section"><h2>{title}</h2>'
        
        # Handle specific section types
        if title == "Basic Sensitivity Analysis":
            if "most_sensitive_layers" in data:
                html += "<h3>Most Sensitive Layers</h3>"
                html += "<table><tr><th>Rank</th><th>Layer</th><th>Sensitivity</th></tr>"
                for idx, layer in enumerate(data["most_sensitive_layers"]):
                    html += f"<tr><td>{idx+1}</td><td>{layer['layer']}</td>"
                    html += f"<td>{layer['sensitivity']:.6f}</td></tr>"
                html += "</table>"
            
        elif title == "Super Weights Identification":
            if "super_weight_stats" in data and "top_5" in data["super_weight_stats"]:
                html += "<h3>Top Super Weights</h3>"
                html += "<table><tr><th>Rank</th><th>Layer</th><th>Parameter</th><th>Sensitivity</th></tr>"
                for idx, sw in enumerate(data["super_weight_stats"]["top_5"]):
                    if isinstance(sw, dict):
                        html += f"<tr><td>{idx+1}</td><td>{sw.get('layer', 'unknown')}</td>"
                        html += f"<td>{sw.get('parameter', 'unknown')}</td>"
                        html += f"<td>{sw.get('sensitivity', 'N/A')}</td></tr>"
                html += "</table>"
                
        elif title == "Pruning Results" and "pruning_impact" in data:
            impact = data["pruning_impact"]
            html += f"<p><strong>Pruning Method:</strong> {impact.get('pruning_method', 'N/A')}</p>"
            html += f"<p><strong>Pruning Percentage:</strong> {impact.get('pruning_percentage', 'N/A')}</p>"
            html += "<table><tr><th>Metric</th><th>Before</th><th>After</th><th>Change</th></tr>"
            html += f"<tr><td>Accuracy</td><td>{impact.get('accuracy_before', 'N/A')}</td>"
            html += f"<td>{impact.get('accuracy_after', 'N/A')}</td>"
            
            change = impact.get('accuracy_change')
            if isinstance(change, (int, float)):
                direction = "+" if change >= 0 else ""
                html += f"<td>{direction}{change:.4f}</td></tr>"
            else:
                html += "<td>N/A</td></tr>"
            html += "</table>"
            
        # Add more custom HTML generation for other sections as needed
        
        html += "</div>"
        return html
    
    def generate_reports(self, output_format: str = "text", include_visualizations: bool = True) -> None:
        """
        Generate reports for all models and save them to the report directory.
        
        Args:
            output_format: Output format ("text", "json", or "html")
            include_visualizations: Whether to include visualizations in the report
        """
        # Create aggregated metrics for all models to enable comparisons
        aggregated_metrics = defaultdict(dict)
        
        # Generate individual model reports
        for model in self.results:
            report = self.generate_model_report(model, output_format)
            
            # Save the report
            extension = "html" if output_format == "html" else "json" if output_format == "json" else "txt"
            report_path = self.report_dir / f"{model}_report.{extension}"
            
            with open(report_path, 'w') as f:
                f.write(report)
                
            print(f"Report for model {model} saved to {report_path}")
            
            # Collect metrics for comparison across models
            for analysis_type in ["sensitivity", "super_weights", "pruning", "evaluation"]:
                analysis_func = getattr(self, f"analyze_{analysis_type}_results")
                results = analysis_func(model)
                
                if analysis_type == "sensitivity" and "avg_layer_sensitivity" in results:
                    aggregated_metrics["avg_layer_sensitivity"][model] = results["avg_layer_sensitivity"]
                    
                if analysis_type == "pruning" and "pruning_impact" in results:
                    if "accuracy_change" in results["pruning_impact"]:
                        aggregated_metrics["pruning_impact"][model] = results["pruning_impact"]["accuracy_change"]
                        
                if analysis_type == "evaluation" and "evaluation_metrics" in results:
                    metrics = results["evaluation_metrics"]
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            if metric not in aggregated_metrics:
                                aggregated_metrics[metric] = {}
                            aggregated_metrics[metric][model] = value
        
        # Generate comparative visualizations if required
        if include_visualizations:
            self._generate_comparative_visualizations(aggregated_metrics)
            
        # Generate summary report comparing all models
        self._generate_summary_report(aggregated_metrics, output_format)
    
    def _generate_comparative_visualizations(self, aggregated_metrics: dict) -> None:
        """
        Generate comparative visualizations for metrics across models.
        
        Args:
            aggregated_metrics: Dictionary of metrics aggregated across models
        """
        visualizations_dir = self.report_dir / "visualizations"
        visualizations_dir.mkdir(exist_ok=True)
        
        for metric, model_values in aggregated_metrics.items():
            if not model_values:
                continue
                
            plt.figure(figsize=(12, 6))
            
            # Sort models by value for better visualization
            sorted_items = sorted(model_values.items(), key=lambda x: x[1])
            models, values = zip(*sorted_items)
            
            # Create bar chart
            plt.bar(models, values)
            plt.title(f"Comparison of {metric} across models")
            plt.xlabel("Model")
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            # Save visualization
            viz_path = visualizations_dir / f"{metric}_comparison.png"
            plt.savefig(viz_path)
            plt.close()
            
            print(f"Comparative visualization for {metric} saved to {viz_path}")
    
    def _generate_summary_report(self, aggregated_metrics: dict, output_format: str = "text") -> None:
        """
        Generate a summary report comparing metrics across all models.
        
        Args:
            aggregated_metrics: Dictionary of metrics aggregated across models
            output_format: Output format ("text", "json", or "html")
        """
        if output_format == "json":
            summary = {
                "title": "ESA Analysis Summary Report",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metrics_comparison": aggregated_metrics,
                "models_analyzed": list(self.results.keys())
            }
            
            summary_path = self.report_dir / "summary_report.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
        elif output_format == "html":
            html = "<html><head><title>ESA Analysis Summary Report</title>"
            html += "<style>body{font-family:Arial;margin:40px;} .section{margin-bottom:30px;} "
            html += "table{border-collapse:collapse;width:100%;} "
            html += "th,td{border:1px solid #ddd;padding:8px;text-align:left;}"
            html += "th{background-color:#f2f2f2;}</style></head><body>"
            html += "<h1>ESA Analysis Summary Report</h1>"
            html += f"<p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
            
            html += f"<h2>Models Analyzed</h2>"
            html += "<ul>"
            for model in self.results:
                html += f"<li>{model}</li>"
            html += "</ul>"
            
            html += "<h2>Metrics Comparison</h2>"
            for metric, model_values in aggregated_metrics.items():
                if not model_values:
                    continue
                    
                html += f"<h3>{metric}</h3>"
                html += "<table><tr><th>Model</th><th>Value</th></tr>"
                
                # Sort models by value
                sorted_items = sorted(model_values.items(), key=lambda x: x[1], reverse=True)
                for model, value in sorted_items:
                    html += f"<tr><td>{model}</td><td>{value}</td></tr>"
                
                html += "</table>"
                
                # Include image reference if visualization exists
                viz_path = f"visualizations/{metric}_comparison.png"
                html += f'<p><img src="{viz_path}" alt="{metric} comparison" width="800"></p>'
            
            html += "</body></html>"
            
            summary_path = self.report_dir / "summary_report.html"
            with open(summary_path, 'w') as f:
                f.write(html)
                
        else:  # Default to text format
            summary = "ESA ANALYSIS SUMMARY REPORT\n"
            summary += "=" * 50 + "\n"
            summary += f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            summary += "MODELS ANALYZED\n"
            summary += "-" * 20 + "\n"
            for model in self.results:
                summary += f"- {model}\n"
            
            summary += "\nMETRICS COMPARISON\n"
            summary += "-" * 20 + "\n"
            
            for metric, model_values in aggregated_metrics.items():
                if not model_values:
                    continue
                    
                summary += f"\n{metric.upper()}:\n"
                
                # Sort models by value
                sorted_items = sorted(model_values.items(), key=lambda x: x[1], reverse=True)
                for i, (model, value) in enumerate(sorted_items):
                    summary += f"  {i+1}. {model}: {value}\n"
            
            summary_path = self.report_dir / "summary_report.txt"
            with open(summary_path, 'w') as f:
                f.write(summary)
        
        print(f"Summary report saved to {summary_path}")


def main():
    """
    Main function to run the results analyzer.
    """
    parser = argparse.ArgumentParser(description="Analyze LLM sensitivity analysis results")
    parser.add_argument("--outputs_dir", type=str, default="../outputs",
                      help="Directory containing analysis outputs")
    parser.add_argument("--report_dir", type=str, default=None,
                      help="Directory to save generated reports")
    parser.add_argument("--format", type=str, choices=["text", "json", "html"], default="html",
                      help="Output format for reports")
    parser.add_argument("--model", type=str, default=None,
                      help="Generate report for specific model only")
    parser.add_argument("--no-viz", action="store_true",
                      help="Disable generation of visualizations")
    
    args = parser.parse_args()
    
    # Resolve relative paths
    if not os.path.isabs(args.outputs_dir):
        args.outputs_dir = os.path.join(os.path.dirname(__file__), args.outputs_dir)
    
    if args.report_dir and not os.path.isabs(args.report_dir):
        args.report_dir = os.path.join(os.path.dirname(__file__), args.report_dir)
    
    # Create analyzer and generate reports
    analyzer = ResultsAnalyzer(args.outputs_dir, args.report_dir)
    
    if args.model:
        # Generate report for specific model
        report = analyzer.generate_model_report(args.model, args.format)
        extension = "html" if args.format == "html" else "json" if args.format == "json" else "txt"
        report_path = os.path.join(args.report_dir or os.path.join(args.outputs_dir, "reports"), 
                                 f"{args.model}_report.{extension}")
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
            
        print(f"Report for model {args.model} saved to {report_path}")
    else:
        # Generate reports for all models
        analyzer.generate_reports(args.format, not args.no_viz)


if __name__ == "__main__":
    main()