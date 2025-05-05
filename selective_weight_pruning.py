#!/usr/bin/env python
"""
Selective Weight Pruning Tool

This script allows for targeted pruning of specific weights in transformer models 
based on their importance scores. You can use custom criteria to select which weights
to remove and analyze the impact on model performance.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import argparse
from tqdm.auto import tqdm
from collections import defaultdict, Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME, TEST_PROMPT, ANALYSIS_CONFIG, MODEL_CONFIG, get_model_by_key
from llm_super_weights import compute_gradient_sensitivity, extract_super_weights
from llm_evaluate_models import evaluate_generation_quality


class SelectiveWeightPruner:
    """
    Tool for selectively pruning specific weights in neural network models
    based on importance criteria.
    """
    
    def __init__(self, model, tokenizer, output_dir):
        """
        Initialize the pruner.
        
        Args:
            model: The PyTorch model to prune
            tokenizer: The tokenizer for the model
            output_dir: Directory to save results
        """
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.device = next(model.parameters()).device
        self.params_dict = dict(model.named_parameters())
        self.original_params = {name: param.detach().clone() 
                               for name, param in self.params_dict.items()}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store weight priorities
        self.sensitivity_data = None
        
        # Keep track of pruned weights for restoration
        self.pruned_weights = {}
    
    def compute_sensitivity(self, method="gradient", prompt=TEST_PROMPT, use_parallel=True):
        """
        Compute weight sensitivity to identify important weights.
        
        Args:
            method: Method to use ('gradient', 'z_score', 'hessian', 'integrated')
            prompt: Input prompt for sensitivity computation
            use_parallel: Whether to use parallel processing
            
        Returns:
            Dictionary of sensitivity data by layer
        """
        print(f"\n🧮 Computing sensitivity using {method} method...")
        
        if method == "gradient":
            self.sensitivity_data = compute_gradient_sensitivity(
                self.model, self.tokenizer, prompt, 
                top_k=1000, use_parallel=use_parallel
            )
        else:
            # Import the appropriate function
            if method == "integrated":
                from llm_super_weights import compute_integrated_gradients
                self.sensitivity_data = compute_integrated_gradients(
                    self.model, self.tokenizer, prompt)
            elif method == "hessian":
                from llm_super_weights import compute_hessian_sensitivity
                self.sensitivity_data = compute_hessian_sensitivity(
                    self.model, self.tokenizer, prompt)
            elif method == "z_score":
                from llm_super_weights import identify_super_weights
                super_weights, _ = identify_super_weights(self.model, z_threshold=3.5)
                self.sensitivity_data = {
                    name: [(idx, 1.0) for idx in indices]
                    for name, indices in super_weights.items()
                }
            else:
                raise ValueError(f"Unknown method: {method}")
                
        print(f"✅ Identified sensitive weights in {len(self.sensitivity_data)} layers")
        return self.sensitivity_data
    
    def load_sensitivity_data(self, file_path):
        """
        Load previously computed sensitivity data.
        
        Args:
            file_path: Path to the JSON file with sensitivity data
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.sensitivity_data = {}
        for name, weights in data.get('super_weights', {}).items():
            self.sensitivity_data[name] = [(item['index'], item['score']) 
                                       for item in weights]
        
        print(f"✅ Loaded sensitivity data from {file_path}")
        return self.sensitivity_data
    
    def select_weights_to_prune(self, criteria):
        """
        Select weights to prune based on specified criteria.
        
        Args:
            criteria: Dictionary with criteria for weight selection
                - threshold: Sensitivity threshold (higher = more important)
                - layers: List of layers to focus on, or 'all'
                - percentile: Top percentile to prune (0-100)
                - max_per_layer: Maximum weights to prune per layer
                - component: Specific component to target ('query', 'key', 'value', etc.)
                - custom_fn: Custom function for weight selection
        
        Returns:
            Dictionary mapping layer names to lists of weight indices to prune
        """
        if self.sensitivity_data is None:
            raise ValueError("Must compute or load sensitivity data first")
            
        threshold = criteria.get('threshold', 0.5)
        target_layers = criteria.get('layers', 'all')
        percentile = criteria.get('percentile', 100)
        max_per_layer = criteria.get('max_per_layer', float('inf'))
        component = criteria.get('component', None)
        custom_fn = criteria.get('custom_fn', None)
        
        pruning_targets = defaultdict(list)
        
        # Collect all weights and their scores
        all_weights = []
        for layer_name, weights in self.sensitivity_data.items():
            # Skip layers not in target_layers if specified
            if target_layers != 'all' and not any(l in layer_name for l in target_layers):
                continue
                
            # Apply component filter if specified
            if component and component not in layer_name:
                continue
                
            for idx, score in weights:
                if score >= threshold:
                    all_weights.append((layer_name, idx, score))
        
        # Sort all weights by sensitivity score (higher score = more important)
        all_weights.sort(key=lambda x: x[2], reverse=True)
        
        # Apply percentile filter (higher percentile = more important weights)
        if percentile < 100:
            cutoff = int(len(all_weights) * (percentile / 100))
            all_weights = all_weights[:cutoff]
        
        # Apply custom filtering function if provided
        if custom_fn is not None:
            all_weights = custom_fn(all_weights)
        
        # Group by layer and apply max_per_layer limit
        layer_counts = defaultdict(int)
        for layer_name, idx, score in all_weights:
            if layer_counts[layer_name] < max_per_layer:
                pruning_targets[layer_name].append((idx, score))
                layer_counts[layer_name] += 1
        
        # Report on selected weights
        total_weights = sum(len(indices) for indices in pruning_targets.values())
        print(f"✅ Selected {total_weights} weights for pruning across {len(pruning_targets)} layers")
        
        # Save selection details
        selection_file = os.path.join(self.output_dir, "pruning_selection.json")
        with open(selection_file, 'w') as f:
            json.dump({
                'criteria': {k: str(v) if callable(v) else v for k, v in criteria.items()},
                'selected_weights': {
                    layer: [{'index': int(idx), 'score': float(score)} 
                           for idx, score in indices]
                    for layer, indices in pruning_targets.items()
                },
                'total_selected': total_weights
            }, f, indent=2)
        
        return pruning_targets
    
    def apply_pruning(self, pruning_targets, method="zero", save_checkpoint=True):
        """
        Apply pruning to selected weights.
        
        Args:
            pruning_targets: Dictionary mapping layer names to lists of weight indices
            method: Pruning method ('zero', 'mean', 'small_noise')
            save_checkpoint: Whether to save the pruned model checkpoint
            
        Returns:
            Summary of pruning operations
        """
        print(f"\n✂️ Applying {method} pruning to selected weights...")
        
        self.pruned_weights = {}
        for layer_name, indices in pruning_targets.items():
            if layer_name not in self.params_dict:
                print(f"⚠️ Warning: Layer {layer_name} not found in model")
                continue
                
            # Get the parameter tensor
            param = self.params_dict[layer_name]
            param_shape = param.shape
            
            # Keep track of original values for potential restoration
            if layer_name not in self.pruned_weights:
                self.pruned_weights[layer_name] = {}
            
            # Apply pruning to each selected weight
            with torch.no_grad():
                for idx, _ in indices:
                    # Convert flat index to tensor indices
                    tensor_idx = np.unravel_index(idx, param_shape)
                    
                    # Store original value
                    self.pruned_weights[layer_name][idx] = param[tensor_idx].item()
                    
                    # Apply the selected pruning method
                    if method == "zero":
                        param[tensor_idx] = 0.0
                    elif method == "mean":
                        param[tensor_idx] = param.mean().item()
                    elif method == "small_noise":
                        std = param.std().item() * 0.01
                        param[tensor_idx] = torch.randn(1).item() * std
        
        # Save pruned model if requested
        if save_checkpoint:
            checkpoint_path = os.path.join(self.output_dir, f"pruned_model_{int(time.time())}.pt")
            self.save_model_weights(checkpoint_path)
            print(f"✅ Saved pruned model checkpoint to {checkpoint_path}")
        
        # Summarize pruning by layer type
        layer_summary = defaultdict(int)
        for layer_name, indices in pruning_targets.items():
            if "query" in layer_name:
                component = "attention_query"
            elif "key" in layer_name:
                component = "attention_key"
            elif "value" in layer_name:
                component = "attention_value"
            elif "out_proj" in layer_name:
                component = "attention_output"
            elif "mlp" in layer_name or "ffn" in layer_name:
                component = "ffn"
            elif "embedding" in layer_name:
                component = "embedding"
            elif "head" in layer_name or "lm_head" in layer_name:
                component = "output_head"
            else:
                component = "other"
                
            layer_summary[component] += len(indices)
        
        print("\n📊 Pruning distribution by component:")
        for component, count in layer_summary.items():
            print(f"  • {component}: {count} weights pruned")
            
        return {
            "pruning_method": method,
            "total_pruned": sum(len(indices) for indices in pruning_targets.values()),
            "layer_distribution": dict(layer_summary)
        }
    
    def restore_weights(self):
        """
        Restore all pruned weights to their original values.
        """
        if not self.pruned_weights:
            print("No weights to restore")
            return
            
        print(f"\n🔄 Restoring {sum(len(w) for w in self.pruned_weights.values())} pruned weights...")
        
        with torch.no_grad():
            for layer_name, indices in self.pruned_weights.items():
                param = self.params_dict.get(layer_name)
                if param is None:
                    continue
                    
                param_shape = param.shape
                for idx, value in indices.items():
                    tensor_idx = np.unravel_index(idx, param_shape)
                    param[tensor_idx] = value
        
        print("✅ Weights restored to original values")
    
    def reset_model(self):
        """
        Reset the model to its original state.
        """
        print(f"\n🔄 Resetting model to original state...")
        
        with torch.no_grad():
            for name, param in self.params_dict.items():
                if name in self.original_params:
                    param.copy_(self.original_params[name])
        
        print("✅ Model reset to original state")
    
    def save_model_weights(self, path):
        """
        Save current model weights to file.
        
        Args:
            path: Path to save weights
        """
        torch.save({
            name: param.detach().cpu() 
            for name, param in self.model.named_parameters()
        }, path)
    
    def evaluate_model(self, prompts=None):
        """
        Evaluate model performance after pruning.
        
        Args:
            prompts: List of prompts to evaluate on
            
        Returns:
            Dictionary with evaluation metrics
        """
        if prompts is None:
            prompts = [TEST_PROMPT]
            
        if not isinstance(prompts, list):
            prompts = [prompts]
        
        metrics = {}
        
        # Compute perplexity
        ppl_scores = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                ppl_scores.append(torch.exp(outputs.loss).item())
        
        metrics["perplexity"] = sum(ppl_scores) / len(ppl_scores)
        
        # Generate responses and evaluate
        generation_metrics = evaluate_generation_quality(
            self.model, self.tokenizer, prompts)
        metrics.update(generation_metrics)
        
        print(f"\n📊 Model evaluation after pruning:")
        print(f"  • Perplexity: {metrics['perplexity']:.2f}")
        if "fluency_score" in metrics:
            print(f"  • Fluency score: {metrics['fluency_score']:.2f}")
        if "relevance_score" in metrics:
            print(f"  • Relevance score: {metrics['relevance_score']:.2f}")
            
        return metrics

    def compare_outputs(self, prompt, original_model=None, max_tokens=100):
        """
        Compare outputs between original and pruned models.
        
        Args:
            prompt: Input prompt
            original_model: Original unpruned model (uses backup if None)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary comparing the outputs
        """
        # Generate with pruned model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pruned_outputs = self.model.generate(
                inputs.input_ids, 
                max_length=inputs.input_ids.shape[1] + max_tokens,
                do_sample=True,
                temperature=0.7
            )
        
        pruned_text = self.tokenizer.decode(pruned_outputs[0], skip_special_tokens=True)
        
        # Generate with original model (either provided or by restoring weights)
        if original_model is None:
            # Backup current params
            current_params = {name: param.detach().clone() 
                           for name, param in self.model.named_parameters()}
            
            # Restore original weights
            self.reset_model()
            
            # Generate with original model
            with torch.no_grad():
                orig_outputs = self.model.generate(
                    inputs.input_ids, 
                    max_length=inputs.input_ids.shape[1] + max_tokens,
                    do_sample=True,
                    temperature=0.7
                )
            
            orig_text = self.tokenizer.decode(orig_outputs[0], skip_special_tokens=True)
            
            # Restore pruned weights
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in current_params:
                        param.copy_(current_params[name])
        else:
            # Use provided original model
            with torch.no_grad():
                orig_outputs = original_model.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + max_tokens,
                    do_sample=True,
                    temperature=0.7
                )
            
            orig_text = self.tokenizer.decode(orig_outputs[0], skip_special_tokens=True)
        
        # Save comparison
        comparison_file = os.path.join(self.output_dir, "output_comparison.txt")
        with open(comparison_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Prompt: {prompt}\n\n")
            f.write(f"ORIGINAL MODEL OUTPUT:\n{orig_text}\n\n")
            f.write(f"PRUNED MODEL OUTPUT:\n{pruned_text}\n")
            f.write(f"{'='*50}\n")
        
        print(f"\n✅ Saved output comparison to {comparison_file}")
        
        return {
            "prompt": prompt,
            "original_output": orig_text,
            "pruned_output": pruned_text
        }

    def analyze_pruned_features(self, top_n=10):
        """
        Analyze what features/patterns were removed by pruning.
        
        Args:
            top_n: Number of top patterns to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not self.pruned_weights:
            print("No weights have been pruned to analyze")
            return {}
            
        # Analyze which model components were most affected
        component_counts = Counter()
        for layer_name in self.pruned_weights:
            # Extract component from layer name
            if "query" in layer_name:
                component = "attention_query"
            elif "key" in layer_name:
                component = "attention_key"
            elif "value" in layer_name:
                component = "attention_value"
            elif "out_proj" in layer_name:
                component = "attention_output"
            elif "mlp" in layer_name or "ffn" in layer_name:
                component = "ffn"
            elif "embedding" in layer_name:
                component = "embedding"
            elif "head" in layer_name or "lm_head" in layer_name:
                component = "output_head"
            else:
                component = "other"
                
            component_counts[component] += len(self.pruned_weights[layer_name])
        
        # Extract most pruned layers
        layer_counts = {layer: len(weights) 
                       for layer, weights in self.pruned_weights.items()}
        top_layers = sorted(layer_counts.items(), 
                           key=lambda x: x[1], reverse=True)[:top_n]
        
        # Get value range of pruned weights
        all_values = []
        for layer, weights in self.pruned_weights.items():
            all_values.extend(list(weights.values()))
        
        value_stats = {
            "min": min(all_values) if all_values else 0,
            "max": max(all_values) if all_values else 0,
            "mean": sum(all_values) / len(all_values) if all_values else 0,
            "std": np.std(all_values) if all_values else 0
        }
        
        # Create distribution plot of pruned weight values
        plt.figure(figsize=(10, 6))
        plt.hist(all_values, bins=50, alpha=0.7, color="royalblue")
        plt.xlabel("Weight Value")
        plt.ylabel("Count")
        plt.title("Distribution of Pruned Weight Values")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, "pruned_values_distribution.png"))
        plt.close()
        
        # Plot component distribution pie chart
        plt.figure(figsize=(10, 8))
        components = list(component_counts.keys())
        counts = list(component_counts.values())
        plt.pie(counts, labels=components, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title("Distribution of Pruned Weights by Component")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pruned_components_pie.png"))
        plt.close()
        
        print("\n📊 Analysis of pruned weights:")
        print(f"  • Total weights pruned: {sum(layer_counts.values())}")
        print("  • Component distribution:")
        for component, count in component_counts.most_common():
            print(f"    - {component}: {count} weights ({count/sum(component_counts.values())*100:.1f}%)")
        
        results = {
            "component_distribution": dict(component_counts),
            "top_pruned_layers": dict(top_layers),
            "pruned_value_stats": value_stats
        }
        
        return results

    def interactive_pruning(self):
        """
        Interactive CLI for selecting weights to prune based on sensitivity data.
        This allows for manual inspection and selection of weights.
        
        Returns:
            Dictionary of pruning targets selected interactively
        """
        if self.sensitivity_data is None:
            print("⚠️ No sensitivity data available. Computing default sensitivity...")
            self.compute_sensitivity(method="gradient")
            
        print("\n🔍 Interactive Weight Pruning Mode")
        print("="*50)
            
        # Flatten and sort all weights by sensitivity
        all_weights = []
        for layer_name, weights in self.sensitivity_data.items():
            for idx, score in weights:
                all_weights.append((layer_name, idx, score))
                
        # Sort by sensitivity score (descending)
        all_weights.sort(key=lambda x: x[2], reverse=True)
        
        # Display top and bottom weights
        top_n = 20
        print(f"\nTop {top_n} most sensitive weights:")
        print(f"{'Rank':<6}{'Layer':<40}{'Index':<12}{'Score':<10}")
        print("-" * 68)
        for i, (layer, idx, score) in enumerate(all_weights[:top_n]):
            print(f"{i+1:<6}{layer[:37]+'...' if len(layer) > 40 else layer:<40}{idx:<12}{score:<.4f}")
            
        print(f"\nBottom {top_n} least sensitive weights:")
        print(f"{'Rank':<6}{'Layer':<40}{'Index':<12}{'Score':<10}")
        print("-" * 68)
        for i, (layer, idx, score) in enumerate(all_weights[-top_n:]):
            print(f"{len(all_weights)-i:<6}{layer[:37]+'...' if len(layer) > 40 else layer:<40}{idx:<12}{score:<.4f}")
            
        # Display layer types
        layer_types = defaultdict(int)
        for layer in self.sensitivity_data.keys():
            if "query" in layer:
                layer_type = "attention_query"
            elif "key" in layer:
                layer_type = "attention_key"
            elif "value" in layer:
                layer_type = "attention_value"
            elif "out_proj" in layer:
                layer_type = "attention_output"
            elif "mlp" in layer or "ffn" in layer:
                layer_type = "ffn"
            elif "embedding" in layer:
                layer_type = "embedding"
            elif "head" in layer or "lm_head" in layer:
                layer_type = "output_head"
            else:
                layer_type = "other"
            layer_types[layer_type] += len(self.sensitivity_data[layer])
            
        print("\nWeight distribution by layer type:")
        for layer_type, count in layer_types.items():
            print(f"- {layer_type}: {count} weights")
        
        # Interactive selection menu
        print("\n--- Selection Menu ---")
        print("1. Select by sensitivity threshold")
        print("2. Select by percentile")
        print("3. Select specific layer type")
        print("4. Manual selection (advanced)")
        print("5. Cancel and return to automatic mode")
        
        choice = input("\nEnter your choice (1-5): ")
        
        pruning_targets = defaultdict(list)
        
        if choice == "1":
            threshold = float(input("Enter sensitivity threshold (lower = more weights pruned): "))
            for layer_name, weights in self.sensitivity_data.items():
                for idx, score in weights:
                    if score <= threshold:
                        pruning_targets[layer_name].append((idx, score))
                        
        elif choice == "2":
            percentile = float(input("Enter percentile threshold (0-100, lower = fewer weights kept): "))
            cutoff_idx = int(len(all_weights) * (1 - percentile/100))
            selected_weights = all_weights[cutoff_idx:]
            
            for layer_name, idx, score in selected_weights:
                pruning_targets[layer_name].append((idx, score))
                
        elif choice == "3":
            print("\nAvailable layer types:")
            for i, layer_type in enumerate(layer_types.keys()):
                print(f"{i+1}. {layer_type}")
                
            layer_choice = input("Select layer type number: ")
            try:
                selected_layer_type = list(layer_types.keys())[int(layer_choice)-1]
                proportion = float(input(f"What proportion of {selected_layer_type} weights to prune (0-1): "))
                
                for layer_name, weights in self.sensitivity_data.items():
                    if selected_layer_type == "attention_query" and "query" in layer_name:
                        target = True
                    elif selected_layer_type == "attention_key" and "key" in layer_name:
                        target = True
                    elif selected_layer_type == "attention_value" and "value" in layer_name:
                        target = True
                    elif selected_layer_type == "attention_output" and "out_proj" in layer_name:
                        target = True
                    elif selected_layer_type == "ffn" and ("mlp" in layer_name or "ffn" in layer_name):
                        target = True
                    elif selected_layer_type == "embedding" and "embedding" in layer_name:
                        target = True
                    elif selected_layer_type == "output_head" and ("head" in layer_name or "lm_head" in layer_name):
                        target = True
                    elif selected_layer_type == "other" and not any(x in layer_name for x in ["query", "key", "value", "out_proj", "mlp", "ffn", "embedding", "head"]):
                        target = True
                    else:
                        target = False
                        
                    if target:
                        # Sort by sensitivity
                        sorted_weights = sorted(weights, key=lambda x: x[1])
                        # Take the least sensitive weights up to the proportion
                        num_to_take = int(len(sorted_weights) * proportion)
                        for idx, score in sorted_weights[:num_to_take]:
                            pruning_targets[layer_name].append((idx, score))
            except (ValueError, IndexError):
                print("Invalid selection, returning to main menu")
                return self.interactive_pruning()
                
        elif choice == "4":
            print("\n⚠️ Advanced mode - manually select weight ranges")
            print("Example syntax: 'layer.weight:1,2,5-10 layer2.bias:1-3'")
            print("This would select indices 1,2,5-10 in layer.weight and 1-3 in layer2.bias")
            
            manual_selection = input("Enter selection pattern (or 'help' for syntax): ")
            
            if manual_selection.lower() == "help":
                print("\nSyntax help:")
                print("- Separate layers with spaces")
                print("- For each layer, use layername:indices format")
                print("- Indices can be comma-separated or ranges with hyphens")
                print("- Example: 'transformer.h.0.attn.c_proj.weight:0-100,200 transformer.h.1.mlp.c_fc.weight:0-50'")
                return self.interactive_pruning()
            
            try:
                selections = manual_selection.split()
                for selection in selections:
                    if ":" not in selection:
                        continue
                        
                    layer_name, indices_str = selection.split(":")
                    
                    if layer_name not in self.params_dict:
                        print(f"⚠️ Warning: Layer '{layer_name}' not found in model")
                        matching_layers = [l for l in self.params_dict.keys() if layer_name in l]
                        if matching_layers:
                            print(f"Did you mean one of: {', '.join(matching_layers[:3])}")
                        continue
                    
                    indices_parts = indices_str.split(",")
                    indices = []
                    
                    for part in indices_parts:
                        if "-" in part:
                            start, end = map(int, part.split("-"))
                            indices.extend(range(start, end + 1))
                        else:
                            indices.append(int(part))
                    
                    # Assign dummy scores for these manually selected weights
                    for idx in indices:
                        pruning_targets[layer_name].append((idx, 0.0))
            except Exception as e:
                print(f"⚠️ Error parsing selection: {e}")
                return self.interactive_pruning()
        
        elif choice == "5":
            print("Returning to automatic mode")
            return None
        
        else:
            print("Invalid choice, returning to main menu")
            return self.interactive_pruning()
        
        # Summarize selection
        total_weights = sum(len(indices) for indices in pruning_targets.values())
        print(f"\n✅ Selected {total_weights} weights for pruning across {len(pruning_targets)} layers")
        
        # Ask about pruning method
        print("\nSelect pruning method:")
        print("1. Zero out weights")
        print("2. Replace with mean values")
        print("3. Replace with small random noise")
        
        method_choice = input("Enter pruning method (1-3): ")
        method_map = {"1": "zero", "2": "mean", "3": "small_noise"}
        
        if method_choice in method_map:
            prune_method = method_map[method_choice]
            confirm = input(f"\nConfirm pruning {total_weights} weights using '{prune_method}' method? (y/n): ")
            
            if confirm.lower() == 'y':
                result = self.apply_pruning(pruning_targets, method=prune_method)
                print(f"\n✅ Applied {prune_method} pruning to {total_weights} weights")
                
                # Offer to evaluate
                eval_choice = input("Would you like to evaluate the pruned model? (y/n): ")
                if eval_choice.lower() == 'y':
                    self.evaluate_model()
                    
                return pruning_targets
            else:
                print("Pruning cancelled")
                return None
        else:
            print("Invalid method selection, using default 'zero' method")
            return pruning_targets


def run_selective_pruning(args):
    """
    Main function to run selective weight pruning.
    
    Args:
        args: Command line arguments
    """
    print(f"\n🚀 Running selective weight pruning for model: {args.model}")
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/selective_pruning_{args.model.replace('/', '_')}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get full model name
    model_name = get_model_by_key(args.model)
    
    # Load model and tokenizer
    print("\n⏳ Loading model...")
    try:
        from utils.model_loader import load_model_optimized
        model, tokenizer = load_model_optimized(model_name, load_8bit=False)
    except (ImportError, Exception) as e:
        print(f"⚠️ Optimized loader not available or failed: {e}")
        # Fall back to regular loading
        model_config = MODEL_CONFIG.copy()
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **model_config)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {device}")
        model.to(device)
    
    # Create pruner instance
    pruner = SelectiveWeightPruner(model, tokenizer, output_dir)
    
    # Either compute or load sensitivity data
    if args.sensitivity_file:
        pruner.load_sensitivity_data(args.sensitivity_file)
    else:
        pruner.compute_sensitivity(method=args.method, use_parallel=(not args.no_parallel))
    
    # Check if interactive mode is enabled
    if args.interactive:
        print("\n🔍 Starting interactive pruning mode...")
        targets = pruner.interactive_pruning()
        if targets is None:
            print("\n❌ Interactive pruning cancelled.")
            return None
    else:
        # Define pruning criteria based on command line args
        criteria = {
            'threshold': args.threshold,
            'percentile': args.percentile,
            'max_per_layer': args.max_per_layer
        }
        
        if args.layers != 'all':
            criteria['layers'] = args.layers.split(',')
            
        if args.component:
            criteria['component'] = args.component
        
        # Select weights to prune
        targets = pruner.select_weights_to_prune(criteria)
        
        # Apply pruning
        pruner.apply_pruning(targets, method=args.prune_method)
    
    # Evaluate pruned model
    if args.evaluate:
        # Try to use evaluation prompts if available
        try:
            from config import EVAL_PROMPTS
            prompts = EVAL_PROMPTS
        except ImportError:
            prompts = [TEST_PROMPT, "Explain how neural networks work:", 
                      "What is the capital of France?"]
            
        results = pruner.evaluate_model(prompts)
        
        # Save evaluation results
        eval_file = os.path.join(output_dir, "evaluation_results.json")
        with open(eval_file, 'w') as f:
            json.dump(results, f, indent=2)
            
    # Compare outputs
    pruner.compare_outputs(TEST_PROMPT)
    
    # Analyze what was pruned
    pruner.analyze_pruned_features()
    
    print(f"\n✅ Selective pruning complete! Results saved to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Selective Weight Pruning Tool")
    parser.add_argument("--model", type=str, default=MODEL_NAME, 
                       help="Model name or shortcut key")
    parser.add_argument("--method", type=str, default="gradient",
                       choices=["gradient", "z_score", "hessian", "integrated"],
                       help="Sensitivity analysis method")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Sensitivity threshold (higher = more important)")
    parser.add_argument("--percentile", type=float, default=100,
                       help="Top percentile to select (0-100)")
    parser.add_argument("--prune-method", type=str, default="zero",
                       choices=["zero", "mean", "small_noise"],
                       help="Pruning method to apply")
    parser.add_argument("--layers", type=str, default="all",
                       help="Comma-separated list of layers to prune, or 'all'")
    parser.add_argument("--component", type=str, default=None,
                       help="Specific component to target (query, key, value, etc.)")
    parser.add_argument("--max-per-layer", type=int, default=1000,
                       help="Maximum weights to prune per layer")
    parser.add_argument("--sensitivity-file", type=str, default=None,
                       help="Path to pre-computed sensitivity data")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel processing")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate model after pruning")
    parser.add_argument("--interactive", action="store_true",
                       help="Use interactive mode to manually select weights")
    
    args = parser.parse_args()
    run_selective_pruning(args)
    

if __name__ == "__main__":
    main()