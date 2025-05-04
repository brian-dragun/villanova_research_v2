"""
Visualization Module

This module provides functions for plotting and visualizing sensitivity analysis results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from ..core.utils import debug_print, ensure_dir
from ..config import OUTPUT_DIR, VISUALIZATION_CONFIG

def set_plot_style():
    """Set consistent plot styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = VISUALIZATION_CONFIG["figsize"]
    plt.rcParams['figure.dpi'] = VISUALIZATION_CONFIG["dpi"]

def plot_weight_distributions(layer_stats, output_dir=OUTPUT_DIR):
    """
    Plot weight value distributions for model layers.
    
    Args:
        layer_stats: Dictionary of layer statistics
        output_dir: Directory to save plots
    """
    ensure_dir(output_dir)
    set_plot_style()
    
    # Calculate plot grid dimensions
    n_layers = len(layer_stats)
    grid_size = int(np.ceil(np.sqrt(n_layers)))
    
    # Create multi-panel figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, (name, stats) in enumerate(layer_stats.items()):
        if i >= len(axes):
            break
            
        # Create synthetic data based on statistics for visualization
        if isinstance(stats, dict) and 'mean' in stats and 'std' in stats:
            mean = stats['mean']
            std = stats['std']
            # Generate synthetic distribution for visualization
            synthetic_data = np.random.normal(mean, std, 1000)
            
            ax = axes[i]
            sns.histplot(synthetic_data, kde=True, ax=ax)
            
            # Add statistics as text
            stats_text = f"μ={mean:.3f}\nσ={std:.3f}"
            ax.text(0.95, 0.95, stats_text, 
                   transform=ax.transAxes, 
                   verticalalignment='top', 
                   horizontalalignment='right',
                   bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
            
            # Set title to shortened layer name
            short_name = name.split('.')[-2] if '.' in name else name
            ax.set_title(short_name, fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=6)
    
    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'weight_distributions.png')
    plt.savefig(output_path)
    plt.close()
    
    debug_print(f"Weight distributions plot saved to {output_path}")
    
    return output_path

def plot_sensitivity_map(sensitivity_data, output_dir=OUTPUT_DIR):
    """
    Plot a heatmap of weight sensitivity across model layers.
    
    Args:
        sensitivity_data: Dictionary mapping layer names to sensitivity metrics
        output_dir: Directory to save plots
    """
    ensure_dir(output_dir)
    set_plot_style()
    
    # Extract data for heatmap
    layer_names = []
    mean_sensitivities = []
    max_sensitivities = []
    
    for name, metrics in sensitivity_data.items():
        if isinstance(metrics, dict) and 'mean' in metrics:
            layer_names.append(name.split('.')[-2] if '.' in name else name)
            mean_sensitivities.append(metrics['mean'])
            max_sensitivities.append(metrics.get('max', 0))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot mean sensitivity
    sns.barplot(x=layer_names, y=mean_sensitivities, ax=ax1)
    ax1.set_title('Mean Weight Sensitivity by Layer')
    ax1.set_ylabel('Mean Sensitivity')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    
    # Plot max sensitivity
    sns.barplot(x=layer_names, y=max_sensitivities, ax=ax2, color='orange')
    ax2.set_title('Max Weight Sensitivity by Layer')
    ax2.set_ylabel('Max Sensitivity')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'sensitivity_by_layer.png')
    plt.savefig(output_path)
    plt.close()
    
    debug_print(f"Sensitivity map saved to {output_path}")
    
    return output_path

def plot_adversarial_comparison(adversarial_results, output_dir=OUTPUT_DIR):
    """
    Plot comparison of original vs adversarial outputs.
    
    Args:
        adversarial_results: Dictionary with adversarial testing results
        output_dir: Directory to save plots
    """
    ensure_dir(output_dir)
    set_plot_style()
    
    if not isinstance(adversarial_results, dict):
        debug_print("Invalid adversarial results format")
        return None
    
    # Extract similarity metrics
    metrics = []
    labels = []
    
    if 'fgsm_similarity' in adversarial_results:
        metrics.append(adversarial_results['fgsm_similarity'])
        labels.append('FGSM')
        
    if 'pgd_similarity' in adversarial_results:
        metrics.append(adversarial_results['pgd_similarity'])
        labels.append('PGD')
    
    if not metrics:
        debug_print("No similarity metrics found in results")
        return None
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, metrics)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    ax.set_ylabel('Similarity to Original Output')
    ax.set_title(f'Adversarial Attack Resilience (ε={adversarial_results.get("epsilon", "N/A")})')
    ax.set_ylim(0, 1.1)  # Set y-axis from 0 to 1.1 for visibility
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'adversarial_comparison.png')
    plt.savefig(output_path)
    plt.close()
    
    debug_print(f"Adversarial comparison plot saved to {output_path}")
    
    return output_path

def plot_ablation_impact(ablation_results, top_n=10, output_dir=OUTPUT_DIR):
    """
    Plot impact of ablating different model components.
    
    Args:
        ablation_results: Dictionary with ablation analysis results
        top_n: Number of top impactful components to show
        output_dir: Directory to save plots
    """
    ensure_dir(output_dir)
    set_plot_style()
    
    if not isinstance(ablation_results, dict) or 'ablation_by_layer' not in ablation_results:
        debug_print("Invalid ablation results format")
        return None
    
    ablation_data = ablation_results['ablation_by_layer']
    
    # Extract impact values and sort
    impacts = []
    layer_names = []
    
    for name, results in ablation_data.items():
        if isinstance(results, dict) and 'impact' in results:
            impacts.append(results['impact'])
            layer_names.append(name.split('.')[-2] if '.' in name else name)
    
    # Sort by impact and take top N
    sorted_indices = np.argsort(impacts)[-top_n:]
    top_impacts = [impacts[i] for i in sorted_indices]
    top_names = [layer_names[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(top_names, top_impacts)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               f'{width:.2%}', ha='left', va='center')
    
    ax.set_xlabel('Performance Impact (higher = more important)')
    ax.set_title('Top Components by Ablation Impact')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ablation_impact.png')
    plt.savefig(output_path)
    plt.close()
    
    debug_print(f"Ablation impact plot saved to {output_path}")
    
    return output_path

def create_summary_dashboard(all_results, output_dir=OUTPUT_DIR):
    """
    Create a comprehensive dashboard of all sensitivity analysis results.
    
    Args:
        all_results: Dictionary with results from all analysis types
        output_dir: Directory to save the dashboard
    """
    ensure_dir(output_dir)
    set_plot_style()
    
    # Create a large figure for the dashboard
    fig = plt.figure(figsize=(20, 16))
    grid = plt.GridSpec(3, 3, figure=fig, wspace=0.3, hspace=0.4)
    
    # Add title
    fig.suptitle("LLM Sensitivity Analysis Dashboard", fontsize=20, y=0.98)
    
    # Plot 1: Weight Sensitivity Map (top left)
    ax1 = fig.add_subplot(grid[0, 0])
    if 'sensitivity' in all_results and 'sensitivity_by_param' in all_results['sensitivity']:
        data = all_results['sensitivity']['sensitivity_by_param']
        layers = list(data.keys())[:10]  # Take first 10 layers
        values = [data[l].get('mean', 0) for l in layers]
        ax1.bar(range(len(layers)), values)
        ax1.set_title("Weight Sensitivity")
        ax1.set_xticks(range(len(layers)))
        ax1.set_xticklabels([l.split('.')[-2] for l in layers], rotation=90)
    else:
        ax1.text(0.5, 0.5, "No sensitivity data available", ha='center')
    
    # Plot 2: Adversarial Resilience (top center)
    ax2 = fig.add_subplot(grid[0, 1])
    if 'adversarial' in all_results:
        data = all_results['adversarial']
        metrics = []
        labels = []
        if 'fgsm_similarity' in data:
            metrics.append(data['fgsm_similarity'])
            labels.append('FGSM')
        if 'pgd_similarity' in data:
            metrics.append(data['pgd_similarity'])
            labels.append('PGD')
        
        if metrics:
            ax2.bar(labels, metrics)
            ax2.set_title("Adversarial Resilience")
            ax2.set_ylim(0, 1)
        else:
            ax2.text(0.5, 0.5, "No adversarial data available", ha='center')
    else:
        ax2.text(0.5, 0.5, "No adversarial data available", ha='center')
    
    # Plot 3: Bit Sensitivity (top right)
    ax3 = fig.add_subplot(grid[0, 2])
    if ('bit_sensitivity' in all_results and 
        'bit_sensitivity_by_layer' in all_results['bit_sensitivity']):
        
        data = all_results['bit_sensitivity']['bit_sensitivity_by_layer']
        layers = list(data.keys())[:5]  # Take first 5 layers
        
        x = np.arange(len(layers))
        width = 0.2
        
        # Plot different bit type impacts
        sign_impacts = [data[l]['sign_impact'] for l in layers]
        exp_impacts = [data[l]['exponent_impact'] for l in layers]
        mant_impacts = [data[l]['mantissa_impact'] for l in layers]
        
        ax3.bar(x - width, sign_impacts, width, label='Sign')
        ax3.bar(x, exp_impacts, width, label='Exponent')
        ax3.bar(x + width, mant_impacts, width, label='Mantissa')
        
        ax3.set_title("Bit-level Sensitivity")
        ax3.set_xticks(x)
        ax3.set_xticklabels([l.split('.')[-2] for l in layers], rotation=90)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No bit-level data available", ha='center')
    
    # Plot 4: Ablation Impact (middle left spanning 2 columns)
    ax4 = fig.add_subplot(grid[1, :2])
    if ('ablation' in all_results and 
        'ablation_by_layer' in all_results['ablation']):
        
        data = all_results['ablation']['ablation_by_layer']
        items = [(k, v.get('impact', 0)) for k, v in data.items() 
                if isinstance(v, dict) and 'impact' in v]
        
        # Sort by impact and take top 8
        items.sort(key=lambda x: x[1], reverse=True)
        top_items = items[:8]
        
        names = [item[0].split('.')[-2] for item in top_items]
        impacts = [item[1] for item in top_items]
        
        y_pos = np.arange(len(names))
        ax4.barh(y_pos, impacts)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(names)
        ax4.set_title("Component Importance (Ablation Impact)")
    else:
        ax4.text(0.5, 0.5, "No ablation data available", ha='center')
    
    # Plot 5: Super Weights (middle right)
    ax5 = fig.add_subplot(grid[1, 2])
    if ('super_weights' in all_results and 
        'most_sensitive_bits' in all_results['super_weights']):
        
        data = all_results['super_weights']['most_sensitive_bits']
        if data:
            impacts = [item['impact'] for item in data]
            labels = [f"{item['layer'].split('.')[-2]}:{item['bit_position']}" 
                    for item in data]
            
            # Plot as horizontal bars
            y_pos = np.arange(len(labels))
            ax5.barh(y_pos, impacts)
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(labels)
            ax5.set_title("Top Sensitive Bits")
        else:
            ax5.text(0.5, 0.5, "No super weight data available", ha='center')
    else:
        ax5.text(0.5, 0.5, "No super weight data available", ha='center')
    
    # Plot 6: Noise Robustness (bottom spanning all columns)
    ax6 = fig.add_subplot(grid[2, :])
    if 'noise_robustness' in all_results:
        data = all_results['noise_robustness']
        # This would be a more complex plot showing how performance varies with noise
        # For now, we'll just show a placeholder
        ax6.text(0.5, 0.5, 
                "Noise Robustness Analysis\n(Detailed data visualization would go here)", 
                ha='center')
    else:
        ax6.text(0.5, 0.5, "No noise robustness data available", ha='center')
    
    # Save the dashboard
    output_path = os.path.join(output_dir, 'sensitivity_dashboard.png')
    plt.savefig(output_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
    plt.close()
    
    debug_print(f"Summary dashboard saved to {output_path}")
    
    return output_path

def run_robust_analysis_display(all_results=None, output_dir=OUTPUT_DIR):
    """
    Main function to generate all visualizations.
    
    Args:
        all_results: Dictionary with results from all analyses
        output_dir: Directory to save visualizations
    """
    ensure_dir(output_dir)
    
    # If no results provided, generate dummy data for demonstration
    if not all_results:
        debug_print("No results provided, using synthetic data for demo")
        all_results = {
            'sensitivity': {
                'sensitivity_by_param': {f'layer.{i}': {'mean': np.random.random()} 
                                        for i in range(10)}
            },
            'adversarial': {
                'fgsm_similarity': np.random.random() * 0.5 + 0.5,
                'pgd_similarity': np.random.random() * 0.4 + 0.5,
                'epsilon': 0.05
            }
        }
    
    # Generate individual plots
    debug_print("Generating sensitivity visualization plots")
    if 'sensitivity' in all_results and 'sensitivity_by_param' in all_results['sensitivity']:
        plot_sensitivity_map(all_results['sensitivity']['sensitivity_by_param'], output_dir)
    
    if 'adversarial' in all_results:
        plot_adversarial_comparison(all_results['adversarial'], output_dir)
    
    if 'ablation' in all_results and 'ablation_by_layer' in all_results['ablation']:
        plot_ablation_impact(all_results['ablation'], output_dir=output_dir)
    
    # Generate comprehensive dashboard
    create_summary_dashboard(all_results, output_dir)
    
    return {
        'output_dir': output_dir,
        'plots_generated': os.listdir(output_dir)
    }