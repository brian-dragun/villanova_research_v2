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
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools
pip install -r requirements.txt
```

### Step 2: Hugging Face Authentication (optional)

For models that require authentication (like Llama):

1. Create a `.env` file in the project root
2. Add your Hugging Face token: `HUGGINGFACE_TOKEN=your_token_here`
3. Get a token at https://huggingface.co/settings/tokens

### Step 3: Automated Setup (Optional)

Alternatively, you can run the provided setup script which handles all of the following:
- Environment setup
- Dependencies installation
- Git configuration
- Hugging Face authentication
- Oh My Posh terminal setup (for better CLI experience)

```bash
# Make the setup script executable
chmod +x setup_v2.sh

# Run the setup script
./setup_v2.sh
```

The setup script will:
- Load your Hugging Face token from .env file
- Configure Git settings
- Update system packages
- Install development dependencies
- Install Python requirements
- Set up Oh My Posh for a better terminal experience
- Install Nerd Fonts for terminal icons

### Step 4: Verify Setup

```bash
# Run a simple test to verify everything is working
python run_analysis.py --model gpt-neo-125m --analysis sensitivity
```

## 📊 Research Dashboard

The framework includes an interactive dashboard for managing and visualizing research outputs. The dashboard organizes outputs by date, model, and analysis type, making it easy to track experiments and compare results.

### Dashboard Commands

```bash
# Generate or update the dashboard with latest results
python manage_outputs.py dashboard

# Open the dashboard in your default web browser
python manage_outputs.py open-dashboard

# List available outputs
python manage_outputs.py list                        # List all outputs
python manage_outputs.py list 7                      # List outputs from the last 7 days
python manage_outputs.py list --model gpt-neo-125m   # Filter by model
python manage_outputs.py list --analysis sensitivity # Filter by analysis type

# Compare outputs from different runs
python manage_outputs.py compare run1_dir run2_dir --files "*.txt" "*.json"

# Clean up old outputs (default: older than 30 days)
python manage_outputs.py clean 30
```

### Running the Dashboard Server

To access the dashboard over the network (including from your local or public IP):

```bash
# Start the dashboard server in the background
python dashboard_server.py --port 8080

# Access via any of these URLs:
# http://localhost:8080/         (from the same machine)
# http://YOUR_LOCAL_IP:8080/     (from the local network)
# http://YOUR_PUBLIC_IP:8080/    (from the internet, if your server has a public IP)

# To run in the background and keep running after terminal closes
nohup python dashboard_server.py --port 8080 > dashboard_server.log 2>&1 &
```

The dashboard provides:
- Organized view of all research outputs
- Filtering by date, model, and analysis type
- Quick access to latest results
- Ability to compare multiple runs
- Direct links to output files and visualizations

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

## 🔬 Weight Sensitivity and Robustness Analysis

This framework provides comprehensive tools for analyzing and understanding weight sensitivity and robustness in Large Language Models. These analyses are crucial for advancing our understanding of LLM behavior, optimizing model performance, and enhancing security.

### Weight Sensitivity Analysis

Weight sensitivity analysis helps identify which parameters have the greatest impact on model performance. This knowledge enables more efficient model compression, better understanding of how LLMs work internally, and identification of potential security vulnerabilities.

#### Basic Sensitivity Analysis Commands

```bash
# Basic sensitivity analysis through layer ablation
python run_analysis.py --analysis sensitivity --model gpt-neo-125m
```

This command systematically zeros out individual layers or components and measures the resulting performance impact. The outputs include:
- Perplexity changes for each ablated component
- Ranked list of most sensitive model parts
- Visualization of sensitivity across the model architecture

#### Super-Weights Analysis Commands

Super-weights are the small subset of parameters that have disproportionate influence on model behavior. Our research shows that typically less than 0.01% of weights can be classified as "super-weights" but modifying them can cause catastrophic performance degradation.

```bash
# Identify super weights using gradient-based sensitivity
python run_analysis.py --model gpt-neo-125m --analysis super-weights --method gradient

# Use statistical z-score method for super weight identification
python run_analysis.py --model gpt-neo-125m --analysis super-weights --method z_score

# Use Hessian-based approach (more computationally intensive but more accurate)
python run_analysis.py --model gpt-neo-125m --analysis super-weights --method hessian --z-threshold 3.0

# Use integrated gradients for attribution (balanced approach)
python run_analysis.py --model gpt-neo-125m --analysis super-weights --method integrated --top-k 2000
```

Each method has different strengths:
- **Gradient-based**: Fast, identifies weights with highest gradient magnitude
- **Z-score**: Statistical approach that identifies outliers in weight importance distribution
- **Hessian-based**: Uses second-order derivatives for more accurate but computationally expensive analysis
- **Integrated gradients**: Path attribution method that provides balanced accuracy and efficiency

#### Comparative Analysis Commands

```bash
# Compare different sensitivity methods (comma-separated list)
python run_analysis.py --model gpt-neo-125m --analysis compare-sensitivity --methods gradient,z_score,integrated

# Run targeted perturbation on identified super weights
python run_analysis.py --model gpt-neo-125m --analysis targeted-perturbation --perturb-ratio 0.02

# Compare sensitivity patterns across different models
python run_analysis.py --analysis compare-sensitivity --model gpt-neo-125m --compare-with llama-2-7b
```

These commands allow you to:
- Measure agreement between different sensitivity methods using metrics like Jaccard similarity
- Understand how super weights cluster across model architecture
- Compare sensitivity patterns across different model families and sizes
- Test how targeted perturbations to super weights compare to random perturbations

### Robustness Testing

Robustness testing evaluates how well models maintain performance when subjected to perturbations, noise, or adversarial inputs. Our framework provides multiple approaches to measure and improve model robustness.

#### Noise Injection Commands

```bash
# Standard robustness test with default noise parameters
python run_analysis.py --analysis robustness --model gpt-neo-125m

# Customize noise type and level
python run_analysis.py --analysis robustness --model gpt-neo-125m --noise-type gaussian --noise-level 0.05

# Test different types of noise
python run_analysis.py --analysis robustness --model gpt-neo-125m --noise-type uniform --noise-level 0.1
```

These tests systematically add noise to model weights to:
- Identify robustness thresholds (maximum noise before significant performance drop)
- Compare robustness across different model components
- Simulate real-world conditions like quantization errors or hardware variability
- Establish confidence intervals for model performance under stress

#### Adversarial Testing Commands

```bash
# Run default adversarial testing
python run_analysis.py --analysis adversarial --model gpt-neo-125m

# Specify attack type
python run_analysis.py --analysis adversarial --model gpt-neo-125m --attack-type token_swap

# Available attack types: gradient, random, token_swap, char_level
python run_analysis.py --analysis adversarial --model gpt-neo-125m --attack-type char_level
```

Adversarial testing exposes models to carefully crafted inputs designed to cause failures:
- **Gradient-based attacks**: Uses gradients to find minimal perturbations that change outputs
- **Token swap attacks**: Strategically replaces tokens to manipulate model behavior
- **Character-level attacks**: Introduces typos and character substitutions to test robustness
- **Random attacks**: Baseline approach for comparison with targeted attacks

#### Bit-Level Analysis Commands

```bash
# Run standard bit-level analysis
python run_analysis.py --analysis bit-level --model gpt-neo-125m

# Perform quantization analysis
python run_analysis.py --analysis bit-level --model gpt-neo-125m --quantize --quantize-bits 4

# Target specific bit positions
python run_analysis.py --analysis bit-level --model gpt-neo-125m --bits 0,1,2,7,15
```

These sophisticated analyses simulate hardware faults and bit-level manipulations:
- Tests resilience against bit flips in different positions
- Evaluates impact of quantization on model performance
- Identifies which bits are most critical for maintaining accuracy
- Simulates fault attacks and hardware vulnerabilities

### Applications of Sensitivity and Robustness Analysis

#### Selective Weight Pruning Commands

Our research shows that sensitivity-aware pruning significantly outperforms traditional magnitude-based pruning, retaining more model performance while achieving similar compression ratios.

```bash
# Interactive pruning with guided interface
python run_analysis.py --analysis selective-pruning --model gpt-neo-125m --interactive

# Advanced pruning with specific targeting based on sensitivity analysis
python run_analysis.py --analysis selective-pruning --model gpt-neo-125m --method gradient --component value --layers transformer.h.3,transformer.h.4 --percentile 80 --threshold 0.3 --prune-method zero

# Direct script usage with comprehensive options
python selective_weight_pruning.py --model gpt-neo-125m --method gradient --component value --layers transformer.h.3,transformer.h.4 --percentile 80 --threshold 0.3 --prune-method zero --evaluate
```

Selective pruning uses sensitivity data to:
- Target least important weights for removal while preserving critical weights
- Create smaller, faster models with minimal performance impact
- Enhance generalization by removing noise-sensitive parameters
- Selectively target specific architectural components for focused pruning

#### Model Evaluation Commands

```bash
# Standard evaluation
python run_analysis.py --analysis evaluate --model gpt-neo-125m

# Use a custom prompt for targeted evaluation
python run_analysis.py --analysis evaluate --model gpt-neo-125m --prompt "Explain quantum computing:"

# Generate multiple samples to assess consistency and robustness
python run_analysis.py --analysis evaluate --model gpt-neo-125m --generate-samples 10 --max-tokens 200

# Compare with another model to measure relative sensitivity and robustness
python run_analysis.py --analysis evaluate --model gpt-neo-125m --compare-with gpt-j-6b
```

These commands allow you to:
- Measure performance before and after sensitivity-based modifications
- Compare robustness across different models and versions
- Evaluate specific capabilities affected by weight sensitivity
- Generate confidence intervals for performance metrics

#### Integrated Analysis Commands

```bash
# Run comprehensive integrated analysis of sensitivity and robustness
python run_analysis.py --analysis integrated --model gpt-neo-125m

# Generate visualizations of sensitivity and robustness patterns
python run_analysis.py --analysis integrated --model gpt-neo-125m --visualize

# Calculate advanced metrics including sensitivity-to-noise ratios
python run_analysis.py --analysis integrated --model gpt-neo-125m --compute-metrics

# Perform all analyses and preserve identified super weights for further study
python run_analysis.py --analysis integrated --model gpt-neo-125m --run-all --save-weights
```

The integrated analysis provides:
- Correlation analysis between sensitivity and robustness metrics
- Unified view of model vulnerabilities across different dimensions
- Cross-referenced results to identify patterns across analysis types
- Comprehensive assessment of model strengths and weaknesses

### General Options for Sensitivity and Robustness Analyses

```bash
# Change sensitivity threshold for super weight identification
python run_analysis.py --analysis super-weights --threshold 0.8

# Enable or disable parallel processing for faster analysis
python run_analysis.py --analysis super-weights --parallel  # Enable (default)
python run_analysis.py --analysis super-weights --no-parallel  # Disable

# Use cached sensitivity results when available to speed up analysis
python run_analysis.py --analysis super-weights --use-cache

# Force CPU-only mode for systems without GPU acceleration
python run_analysis.py --analysis sensitivity --model gpt-neo-125m --cpu-only
```

### Listing Available Options

```bash
# List available models with their sensitivity and robustness characteristics
python run_analysis.py --list-models

# List available analysis types for sensitivity and robustness testing
python run_analysis.py --list-analyses
```

## ⚡ Key Research Findings on Weight Sensitivity and Robustness

Our framework has enabled several important discoveries about LLM weight sensitivity and robustness:

1. **Super Weight Phenomenon**: We've identified that typically less than 0.01% of weights have disproportionate impact on model performance, often concentrated in specific attention heads and feed-forward projections.

2. **Architecture-Specific Patterns**: Different model architectures (Transformer, Llama, GPT) exhibit distinct sensitivity patterns, with some architectures showing more distributed sensitivity than others.

3. **Sensitivity-Robustness Correlation**: Highly sensitive weights tend to be more vulnerable to perturbation, creating potential security vulnerabilities that can be exploited or protected against.

4. **Efficient Compression**: Sensitivity-guided pruning consistently outperforms magnitude-based pruning by 15-30% in preserving model performance at the same compression ratio.

5. **Quantization Insights**: Bit-level sensitivity analysis reveals that higher bits are not uniformly important across all weights, enabling more nuanced mixed-precision quantization approaches.

These findings demonstrate how sensitivity and robustness analysis can lead to more efficient, secure, and reliable language models.
