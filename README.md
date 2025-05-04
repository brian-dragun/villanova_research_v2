# LLM Weight Sensitivity Analysis Framework

This repository contains tools and scripts for analyzing the sensitivity, robustness, and performance characteristics of Large Language Models (LLMs).

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Setup Instructions](#setup-instructions)
- [Available Commands](#available-commands)
- [Analysis Types](#analysis-types)
- [Working with Models](#working-with-models)
- [Results Interpretation](#results-interpretation)
- [Advanced Usage](#advanced-usage)

## 🔍 Project Overview

This framework allows researchers to analyze various aspects of LLMs including:

- **Weight Sensitivity**: Identify which layers/weights are most important for model performance
- **Model Pruning**: Remove less important weights to create smaller, efficient models
- **Robustness Testing**: Evaluate how models perform under noise or adversarial conditions
- **Bit-level Analysis**: Examine effects of bit-level manipulations on model performance

The framework is designed to work with Hugging Face Transformers models and supports a variety of architectures including Llama, GPT-Neo, GPT-J, OPT, BLOOM, Pythia, Falcon, and Mistral.

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
├── llm_super_weights.py             # Super-weights identification
├── llm_robust_analysis_display.py   # Results visualization utilities
├── llm_train.py                     # Fine-tuning utilities
├── requirements.txt                 # Python dependencies
├── LLM_RESEARCH.md                  # Research methodology documentation
├── model_cache/                     # Downloaded pre-trained models
│   ├── models--EleutherAI--gpt-neo-125m/  # Cached GPT-Neo model
│   └── models--meta-llama--Llama-2-7b-hf/ # Cached Llama model
├── data/                            # Modified model data (pruned/noisy versions)
│   ├── gpu_llm_finetuned_*          # Fine-tuned model directories
│   ├── gpu_llm_pruned_*.pth         # Pruned model weights
│   └── gpu_llm_noisy_*.pth          # Noisy model weights
├── outputs/                         # Analysis results
│   └── sensitivity_*/               # Sensitivity analysis results
│       ├── ablation_results.json    # Raw results data
│       ├── ablation_results.png     # Results visualization
│       └── analysis_report.txt      # Human-readable report
├── utils/                           # Utility functions
├── llm_analysis/                    # Additional analysis modules
└── archive/                         # Archived code versions
```

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

### Listing Available Options

```bash
# List available models
python run_analysis.py --list-models

# List available analysis types
python run_analysis.py --list-analyses
```

### Running a Model Directly

```bash
python run_model.py --model MODEL_NAME --prompt "Your prompt here"
```

### Advanced Options

```bash
# Skip fine-tuning step in analyses that require it
python run_analysis.py --skip-finetune

# Specify custom output directory
python run_analysis.py --output-dir ./my_custom_output
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

For sensitivity analysis, higher perplexity increases after ablation indicate more important layers.

## 🔧 Advanced Usage

### Custom Ablation Targets

Edit the appropriate analysis script to target specific layers:

```python
# Example in llm_weight_sensitivity_analysis.py
layers_to_ablate = ["model.layers.5.mlp.gate_proj.weight"]  # Change to test different layers
```

### Working with Locally Modified Models

```bash
# First run a pruning analysis to create a pruned model
python run_analysis.py --model gpt-neo-125m --analysis pruning

# Then analyze the pruned model's characteristics
# The system will automatically use the pruned version from data/
python run_analysis.py --model gpt-neo-125m --analysis robustness
```

### Extending the Framework

Create new analysis modules by following the pattern in existing files:
1. Create a new file `llm_my_analysis.py`
2. Implement a `main(model_name=None, output_dir=None)` function
3. Add your analysis to the dictionary in `run_analysis.py`

## 🧪 Troubleshooting

- **CUDA out of memory**: Use a smaller model or enable 8-bit quantization in `config.py`
- **Authentication errors**: Ensure your HF token is correctly set in `.env`
- **Module not found**: Check that all dependencies are installed via `pip install -r requirements.txt`

## 📄 License

[Include license information here]

## 🙏 Acknowledgments

[Include acknowledgments here]
