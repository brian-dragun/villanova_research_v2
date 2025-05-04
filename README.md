# Extreme Sensitivity Analysis (ESA) for Large Language Models

This repository contains tools and frameworks for conducting advanced sensitivity analysis on transformer-based language models, focusing on identifying and characterizing "super weights" - the most critical parameters within neural network architectures.

## 📋 Research Overview

This framework supports Ph.D. research on Extreme Sensitivity Analysis (ESA) in Large Language Models, with the following key objectives:

- **Super Weight Identification**: Discover and prioritize the most critical weights in transformer architectures
- **Robustness Enhancement**: Improve model resilience against noise, perturbations, and adversarial attacks
- **Efficient Pruning**: Enable knowledge-guided compression by preserving critical parameters
- **Security Analysis**: Evaluate vulnerability to bit-flip attacks and memory corruption
- **Method Comparison**: Compare traditional gradient-based sensitivity with LLM-based attribution methods

The framework builds on Hugging Face Transformers and supports various architectures including Llama, GPT-Neo, GPT-J, OPT, BLOOM, Pythia, Falcon, and Mistral.

## 📁 Directory Structure

```
villanova_research_v2/
├── config.py                        # Configuration settings for models and analyses
├── run_analysis.py                  # Main entry point for running analyses
├── run_model.py                     # Script for running inference with models
├── iterate.py                       # Helper for running iterative analyses
├── llm_weight_sensitivity_analysis.py # Core sensitivity analysis module
├── llm_prune_model.py               # Model pruning functionality
├── llm_robustness_test.py           # Robustness testing module
├── llm_adversarial_test.py          # Adversarial attack testing
├── llm_bit_level_and_ablation_analysis.py # Bit-level manipulation analysis
├── llm_evaluate_models.py           # Model evaluation utilities
├── llm_integrated_analysis.py       # Combined analysis pipeline
├── llm_analyze_sensitivity.py       # Additional sensitivity analysis tools
├── llm_super_weights.py             # Super-weights identification algorithms
├── llm_robust_analysis_display.py   # Results visualization utilities
├── llm_train.py                     # Fine-tuning utilities
├── selective_weight_pruning.py      # Tool for targeted pruning based on sensitivity
├── requirements.txt                 # Python dependencies
├── LLM_RESEARCH.md                  # Research methodology documentation
├── model_cache/                     # Downloaded pre-trained models
├── data/                            # Modified model data (pruned/noisy versions)
├── outputs/                         # Analysis results
├── utils/                           # Utility functions
├── llm_analysis/                    # Additional analysis modules
└── archive/                         # Archived code versions
```

## 🔬 Research Methodology

This framework implements several approaches to sensitivity analysis:

1. **Layer Ablation**: Zero out weights and measure performance degradation
2. **Gradient-based Analysis**: Use gradients to estimate parameter importance
3. **Hessian-based Analysis**: Second-order derivatives for more accurate sensitivity
4. **Noise Injection**: Controlled noise addition to test robustness 
5. **Bit-level Manipulation**: Simulate hardware faults through bit flips
6. **LLM-based Attribution**: Novel approach using language models for weight attribution
7. **Integrated Gradients**: Attribution through gradient path integration

Each analysis method helps identify different aspects of weight sensitivity and contributes to a comprehensive understanding of model robustness.

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8+ 
- CUDA-compatible GPU recommended for larger models
- ~20GB disk space for model storage

### Step 1: Install Dependencies

```bash
# Clone the repository if you haven't already
# git clone https://your-repo-url.git
# cd villanova_research_v2

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Hugging Face Authentication (optional)

For models that require authentication (like Llama):

1. Create a `.env` file in the project root
2. Add your Hugging Face token: `HUGGINGFACE_TOKEN=your_token_here`
3. Get a token at https://huggingface.co/settings/tokens

### Step 3: Verify Setup

```bash
# Run a simple test to verify everything is working
python run_analysis.py --model gpt-neo-125m --analysis sensitivity
```

## ⚙️ Available Commands

### Running Analyses

```bash
# Basic syntax
python run_analysis.py --model MODEL_NAME --analysis ANALYSIS_TYPE

# Examples
python run_analysis.py  # Uses default model (gpt-neo-125m) and analysis (sensitivity)
python run_analysis.py --model llama-2-7b
python run_analysis.py --analysis pruning
python run_analysis.py --model gpt-neo-125m --analysis robustness
```

### Super-Weights Analysis Commands

```bash
# Identify super weights using gradient-based sensitivity
python run_analysis.py --model gpt-neo-125m --analysis super-weights --method gradient

# Use statistical z-score method for super weight identification
python run_analysis.py --model gpt-neo-125m --analysis super-weights --method z_score

# Use Hessian-based approach (more computationally intensive but more accurate)
python run_analysis.py --model gpt-neo-125m --analysis super-weights --method hessian

# Use integrated gradients for attribution (balanced approach)
python run_analysis.py --model gpt-neo-125m --analysis super-weights --method integrated
```

### Selective Weight Pruning Commands

```bash
# Basic syntax
python selective_weight_pruning.py --model MODEL_NAME --method METHOD --component COMPONENT --layers LAYERS --percentile PERCENTILE --threshold THRESHOLD --prune-method METHOD --evaluate

# Example: Prune specific transformer layers using gradient-based sensitivity
python selective_weight_pruning.py --model gpt-neo-125m --method gradient --component value --layers transformer.h.3,transformer.h.4 --percentile 80 --threshold 0.3 --prune-method zero --evaluate

# Important: Make sure to use the correct model name format with dashes (e.g., "gpt-neo-125m" NOT "gptneo125m")
```

### Comparative Analysis Commands

```bash
# Compare different sensitivity methods (comma-separated list)
python run_analysis.py --model gpt-neo-125m --analysis compare-sensitivity --methods gradient,z_score,integrated

# Run targeted perturbation on identified super weights
python run_analysis.py --model gpt-neo-125m --analysis targeted-perturbation
```

### Listing Available Options

```bash
# List available models
python run_analysis.py --list-models

# List available analysis types
python run_analysis.py --list-analyses
```

### Advanced Options

```bash
# Skip fine-tuning step in analyses that require it
python run_analysis.py --skip-finetune

# Specify custom output directory
python run_analysis.py --output-dir ./my_custom_output

# Change sensitivity threshold for super weight identification
python run_analysis.py --analysis super-weights --threshold 0.8
```

## 📊 Analysis Types

The framework supports several analysis types:

1. **sensitivity**: Analyze weight sensitivity through layer ablation
   - Identifies critical weights by zeroing out layers and measuring perplexity changes

2. **pruning**: Prune model weights to create smaller models
   - Removes less important weights based on sensitivity analysis
   - Creates pruned model files in the data directory

3. **robustness**: Test model robustness under various conditions
   - Adds controlled noise to weights and measures performance impact
   - Evaluates model stability under perturbations

4. **adversarial**: Run adversarial attacks against the model
   - Tests model response to carefully crafted inputs designed to mislead

5. **bit-level**: Perform bit-level and ablation analysis
   - Examines impact of bit flips in weights on model performance
   - Useful for studying quantization effects

6. **integrated**: Run comprehensive integrated analysis
   - Combines multiple analysis types for a complete evaluation

7. **evaluate**: Evaluate model performance on benchmark tasks
   - Measures standard metrics for comparing models

8. **super-weights**: Identify and analyze critical weights
   - Uses gradient or Hessian-based methods to find "super weights"
   - Supports multiple identification methods: gradient, z_score, hessian, integrated

9. **compare-sensitivity**: Compare different sensitivity methods
   - Evaluates gradient, ablation, z-score, and integrated gradients methods
   - Measures agreement between methods using Jaccard similarity
   - Compares computational efficiency and execution time

10. **targeted-perturbation**: Test robustness by perturbing only super weights
    - Focuses on the most critical weights for targeted analysis
    - Creates impact visualization showing relationship between % weights perturbed and performance degradation

11. **selective-weight-pruning**: Targeted pruning based on sensitivity analysis
    - Allows pruning specific layers or components using various sensitivity methods
    - Evaluates performance changes post-pruning

## 🤖 Working with Models

### Available Models

The framework supports various models defined in `config.py`, including:

- **Llama**: `llama-2-7b`, `llama-2-7b-chat`
- **GPT-Neo/J**: `gpt-neo-125m`, `gpt-neo-1.3b`, `gpt-j-6b`
- **OPT**: `opt-350m`, `opt-1.3b`
- **BLOOM**: `bloom-560m`, `bloom-1b1`
- **Pythia**: `pythia-70m`, `pythia-410m`
- **Falcon**: `falcon-rw-1b`
- **Mistral**: `mistral-7b`

### Model Storage

Models are stored in several locations:

- `model_cache/`: Downloaded pre-trained models from Hugging Face
- `data/gpu_llm_finetuned_*`: Fine-tuned model versions
- `data/gpu_llm_pruned_*.pth`: Pruned model versions
- `data/gpu_llm_noisy_*.pth`: Noisy model versions for robustness testing

The system automatically uses cached models when available to avoid re-downloading.

## 📈 Results Interpretation

Analysis results are saved to the `outputs/` directory with the following structure:

- `outputs/ANALYSIS_TYPE_MODEL_NAME_TIMESTAMP/`
  - `ablation_results.json`: Raw numerical results
  - `ablation_results.png`: Visualization of perplexity changes
  - `analysis_report.txt`: Human-readable report explaining results
  - `sensitivity_map.png`: Heatmap showing sensitivity across model layers
  - `super_weights.json`: Identified super-weights with sensitivity scores
  - `perturbation_impact.png`: Graph showing impact of perturbing super weights
  - `method_similarity.png`: Similarity matrix comparing sensitivity methods

For selective weight pruning analysis, the results include:
- `output_comparison.txt`: Shows the comparison between original and pruned model outputs
- Visualizations of how pruning affects specific layers
- Performance metrics before and after pruning

### Expected Output from Selective Weight Pruning

When running selective weight pruning commands, you can expect significant changes in model output behavior. For example, pruning the GPT-Neo-125m model shows:

**Original model** tends to repeat phrases and struggle with the actual question, producing text like:
```
In a galaxy far and away, in the far-west, far away you can see the galaxies in the far-west, far away. This is a great movie, and you can really see the galaxy in the far-west at the same time.
```

**Pruned model** produces dramatically different output with a more article-like structure:
```
(Image: Getty)

By

Michael A.

When you first get hooked on the genre, it's easy to see why. The movie industry is still largely unknown...
```

These differences highlight how selective pruning can fundamentally alter the model's generation patterns, which is valuable for understanding the role specific weights play in different aspects of text generation.

For super-weights analysis, the results include:
- Lists of the most sensitive weights per layer
- Layer-wise sensitivity distribution
- Perturbation impact analysis
- Performance degradation curves

## 🔧 Advanced Research Usage

### Super Weight Identification Methods

```python
# In llm_super_weights.py
# Gradient-based sensitivity
sensitivity_data = compute_gradient_sensitivity(model, tokenizer, prompt)

# Hessian-based sensitivity
sensitivity_data = compute_hessian_sensitivity(model, tokenizer, prompt)

# Integrated gradients
sensitivity_data = compute_integrated_gradients(model, tokenizer, prompt)

# Z-score statistical outliers
super_weights, layer_summary = identify_super_weights(model, z_threshold=2.5)
```

### Comparing Sensitivity Methods

```python
# Run comparative analysis between methods
results = compare_sensitivity_methods(
    model=model,
    tokenizer=tokenizer,
    prompt="What is machine learning?",
    output_dir="./outputs/comparison",
    methods=["gradient", "z_score", "integrated"]
)
```

### Targeted Perturbation Testing

```python
# Test impact of perturbing super weights
results = ablation_sensitivity_test(
    model=model,
    tokenizer=tokenizer,
    sensitivity_data=sensitivity_data,
    prompt="What is machine learning?",
    output_dir="./outputs/perturbation"
)
```

### Extending with Custom Sensitivity Metrics

Create new sensitivity metrics by adding methods to the `llm_super_weights.py` file:

```python
# Example of implementing a new sensitivity metric
def custom_sensitivity_metric(model, tokenizer, dataset):
    # Implementation here
    pass
```

## 📊 Visualization Examples

The framework generates various visualizations to help interpret results:

- **Layer-wise Sensitivity Distribution**: Bar chart showing which layers contain the most super weights
- **Perturbation Impact Curve**: Line chart showing how model performance degrades as more super weights are perturbed
- **Method Similarity Matrix**: Heatmap showing the Jaccard similarity between different sensitivity methods
- **Execution Time Comparison**: Bar chart comparing the computational efficiency of different methods

## 🧪 Troubleshooting

- **CUDA out of memory**: Use a smaller model or enable 8-bit quantization in `config.py`
- **Authentication errors**: Ensure your HF token is correctly set in `.env`
- **Module not found**: Check that all dependencies are installed via `pip install -r requirements.txt`
- **Model not found errors**: Make sure to use the correct model name format with dashes (e.g., "gpt-neo-125m" not "gptneo125m")

## 📝 Research Citations

When using this framework in research publications, please cite:

[Your citation information here]

## 📄 License

[Include license information here]

## 🙏 Acknowledgments

[Include acknowledgments here]
