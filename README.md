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
python run_analysis.py --model gpt-neo-125m --analysis super-weights --method hessian --z-threshold 3.0

# Use integrated gradients for attribution (balanced approach)
python run_analysis.py --model gpt-neo-125m --analysis super-weights --method integrated --top-k 2000
```

### Selective Weight Pruning Commands

```bash
# Using centralized run_analysis.py interface (recommended)
python run_analysis.py --analysis selective-pruning --model gpt-neo-125m --interactive

# Advanced pruning with specific targeting
python run_analysis.py --analysis selective-pruning --model gpt-neo-125m --method gradient --component value --layers transformer.h.3,transformer.h.4 --percentile 80 --threshold 0.3 --prune-method zero

# Direct script usage (legacy approach)
python selective_weight_pruning.py --model gpt-neo-125m --method gradient --component value --layers transformer.h.3,transformer.h.4 --percentile 80 --threshold 0.3 --prune-method zero --evaluate

# Important: Make sure to use the correct model name format with dashes (e.g., "gpt-neo-125m" NOT "gptneo125m")
```

### Robustness Testing Commands

```bash
# Standard robustness test with default noise parameters
python run_analysis.py --analysis robustness --model gpt-neo-125m

# Customize noise type and level
python run_analysis.py --analysis robustness --model gpt-neo-125m --noise-type gaussian --noise-level 0.05

# Test different types of noise
python run_analysis.py --analysis robustness --model gpt-neo-125m --noise-type uniform --noise-level 0.1
```

### Adversarial Testing Commands

```bash
# Run default adversarial testing
python run_analysis.py --analysis adversarial --model gpt-neo-125m

# Specify attack type
python run_analysis.py --analysis adversarial --model gpt-neo-125m --attack-type token_swap

# Available attack types: gradient, random, token_swap, char_level
python run_analysis.py --analysis adversarial --model gpt-neo-125m --attack-type char_level
```

### Bit-Level Analysis Commands

```bash
# Run standard bit-level analysis
python run_analysis.py --analysis bit-level --model gpt-neo-125m

# Perform quantization analysis
python run_analysis.py --analysis bit-level --model gpt-neo-125m --quantize --quantize-bits 4

# Target specific bit positions
python run_analysis.py --analysis bit-level --model gpt-neo-125m --bits 0,1,2,7,15
```

### Model Evaluation Commands

```bash
# Standard evaluation
python run_analysis.py --analysis evaluate --model gpt-neo-125m

# Use a custom prompt for evaluation
python run_analysis.py --analysis evaluate --model gpt-neo-125m --prompt "Explain quantum computing:"

# Generate more samples during evaluation
python run_analysis.py --analysis evaluate --model gpt-neo-125m --generate-samples 10 --max-tokens 200

# Compare with another model
python run_analysis.py --analysis evaluate --model gpt-neo-125m --compare-with gpt-j-6b
```

### Comparative Analysis Commands

```bash
# Compare different sensitivity methods (comma-separated list)
python run_analysis.py --model gpt-neo-125m --analysis compare-sensitivity --methods gradient,z_score,integrated

# Run targeted perturbation on identified super weights
python run_analysis.py --model gpt-neo-125m --analysis targeted-perturbation --perturb-ratio 0.02

# Compare robustness across different models
python run_analysis.py --analysis compare-sensitivity --model gpt-neo-125m --compare-with llama-2-7b
```

### Integrated Analysis Commands

```bash
# Run comprehensive integrated analysis
python run_analysis.py --analysis integrated --model gpt-neo-125m

# Save visualizations
python run_analysis.py --analysis integrated --model gpt-neo-125m --visualize

# Compute additional metrics
python run_analysis.py --analysis integrated --model gpt-neo-125m --compute-metrics

# Run all analyses and save weights
python run_analysis.py --analysis integrated --model gpt-neo-125m --run-all --save-weights
```

### Listing Available Options

```bash
# List available models
python run_analysis.py --list-models

# List available analysis types
python run_analysis.py --list-analyses
```

### General Options (Apply to Most Analyses)

```bash
# Skip fine-tuning step in analyses that require it
python run_analysis.py --skip-finetune

# Specify custom output directory
python run_analysis.py --output-dir ./my_custom_output

# Change sensitivity threshold for super weight identification
python run_analysis.py --analysis super-weights --threshold 0.8

# Enable or disable parallel processing
python run_analysis.py --analysis super-weights --parallel  # Enable (default)
python run_analysis.py --analysis super-weights --no-parallel  # Disable

# Use cached results when available
python run_analysis.py --analysis super-weights --use-cache
```

## 📊 Analysis Types

The framework supports several analysis types:

1. **sensitivity**: Analyze weight sensitivity through layer ablation
   - Identifies critical weights by zeroing out layers and measuring perplexity changes
   - **Expected Output**: Perplexity scores for each layer when ablated, sensitivity maps showing which layers affect performance most, and performance degradation metrics
   - **Output Example**: `Layer transformer.h.8 shows highest sensitivity with 124% perplexity increase when ablated`

2. **pruning**: Prune model weights to create smaller models
   - Removes less important weights based on sensitivity analysis
   - Creates pruned model files in the data directory
   - **Expected Output**: Pruned model stats including size reduction (e.g., "Model size reduced by 43%"), inference speed improvement, and quality/performance trade-offs
   - **Output Example**: `Pruned model (80% threshold): 23MB (43% smaller), 1.4x faster inference, perplexity increased by 15%`

3. **robustness**: Test model robustness under various conditions
   - Adds controlled noise to weights and measures performance impact
   - Evaluates model stability under perturbations
   - **Expected Output**: Graphs showing performance vs. noise level, critical noise thresholds where model breaks down, and layer-wise robustness rankings
   - **Output Example**: `Model maintains 90% performance up to noise level 0.05; attention layers most sensitive to perturbation`

4. **adversarial**: Run adversarial attacks against the model
   - Tests model response to carefully crafted inputs designed to mislead
   - **Expected Output**: Success rates of different attack types, examples of successful adversarial inputs, and recommendations for improving model resilience
   - **Output Example**: `Token substitution attack success rate: 73%; Character-level attack success rate: 41%; Most vulnerable topic: finance`

5. **bit-level**: Perform bit-level and ablation analysis
   - Examines impact of bit flips in weights on model performance
   - Useful for studying quantization effects
   - **Expected Output**: Bit sensitivity maps showing which bit positions are most critical, sensitivity rankings by parameter type, and quantization recommendations
   - **Output Example**: `Sign bit flips in attention layers cause 87% performance drop; least significant 3 bits can be pruned with <5% quality impact`

6. **integrated**: Run comprehensive integrated analysis
   - Combines multiple analysis types for a complete evaluation
   - **Expected Output**: Consolidated results from all analysis types, cross-analysis correlations, and an executive summary of model strengths/weaknesses
   - **Output Example**: `Critical components: layers 4-7 attention matrices, vocabulary embeddings for common tokens, highest bit sensitivity in feed-forward networks`

7. **evaluate**: Evaluate model performance on benchmark tasks
   - Measures standard metrics for comparing models
   - **Expected Output**: Performance metrics (perplexity, accuracy, BLEU, etc.) on standard datasets, comparison with baseline models, and performance breakdown by task type
   - **Output Example**: `WikiText perplexity: 32.4, GLUE avg: 0.67, generation quality score: 7.3/10`

8. **super-weights**: Identify and analyze critical weights
   - Uses gradient or Hessian-based methods to find "super weights"
   - Supports multiple identification methods: gradient, z_score, hessian, integrated
   - **Expected Output**: Lists of super weights by layer, distribution analysis of super weight locations, and impact assessment when these weights are modified
   - **Output Example**: `Identified 217 super weights (0.003% of total); 68% located in attention layers; top 10 weights are in vocabulary embeddings`

9. **compare-sensitivity**: Compare different sensitivity methods
   - Evaluates gradient, ablation, z-score, and integrated gradients methods
   - Measures agreement between methods using Jaccard similarity
   - Compares computational efficiency and execution time
   - **Expected Output**: Similarity matrices between methods, execution time comparisons, and agreement analysis on super weight identification
   - **Output Example**: `Gradient and z-score methods show 76% agreement; Hessian method 4.3x slower but identifies 12% more critical weights`

10. **targeted-perturbation**: Test robustness by perturbing only super weights
    - Focuses on the most critical weights for targeted analysis
    - Creates impact visualization showing relationship between % weights perturbed and performance degradation
    - **Expected Output**: Performance degradation curves, comparison between targeted vs. random perturbation, and critical perturbation thresholds
    - **Output Example**: `Perturbing 0.01% of super weights causes same degradation as random perturbation of 5% weights; model breaks at 30% super weight noise`

11. **selective-weight-pruning**: Targeted pruning based on sensitivity analysis
    - Allows pruning specific layers or components using various sensitivity methods
    - Evaluates performance changes post-pruning
    - **Expected Output**: Output comparison between original and pruned models, performance metrics, and generated text examples showing behavior changes
    - **Output Example**: As shown in the example above comparing original and pruned model outputs

## 📈 Results Interpretation

Analysis results are saved to the `outputs/` directory with the following structure:

- `outputs/ANALYSIS_TYPE_MODEL_NAME_TIMESTAMP/`
  - `ablation_results.json`: Raw numerical results showing perplexity changes when layers are ablated
  - `ablation_results.png`: Visualization of perplexity changes across different layers
  - `analysis_report.txt`: Human-readable report explaining results and key findings
  - `sensitivity_map.png`: Heatmap showing sensitivity across model layers and components
  - `super_weights.json`: Identified super-weights with sensitivity scores and locations
  - `perturbation_impact.png`: Graph showing impact of perturbing super weights vs random weights
  - `method_similarity.png`: Similarity matrix comparing different sensitivity methods
  - `pruning_comparison.txt`: Performance metrics before and after different levels of pruning
  - `robustness_curve.png`: Visual representation of model performance under increasing noise
  - `bit_sensitivity.json`: Data showing impact of bit-level modifications on performance
  - `adversarial_examples.txt`: Examples of inputs that successfully fool the model
  - `model_output_samples.txt`: Sample outputs from original and modified models

### Expected Output Patterns by Analysis Type

#### Sensitivity Analysis
The output will show which layers and components are most critical to model performance. Look for:
- Perplexity increases when specific layers are ablated
- Heat maps highlighting sensitive components
- Lists of weights ranked by sensitivity score

```
Layer sensitivity ranking:
1. transformer.h.8 (124% perplexity increase)
2. transformer.h.4 (98% perplexity increase)
3. transformer.wte (word embeddings) (87% perplexity increase)
...
```

#### Super-Weights Analysis
You'll receive detailed information about the most critical weights in the model:
- Their locations within the model architecture
- Their relative importance scores
- Visualizations showing their distribution across layers
- The impact on performance when these weights are modified

Example output will include something like:
```
Super weight analysis complete. Found 217 super weights out of 125M parameters (0.00017%).
Distribution by layer type:
- Attention layers: 68%
- Feed-forward networks: 23%
- Embeddings: 9%

Top 5 super weights:
1. transformer.wte.weight[4242] - Sensitivity: 0.87
2. transformer.h.8.attn.c_attn.weight[512,768] - Sensitivity: 0.83
3. transformer.h.4.mlp.c_proj.weight[2048,768] - Sensitivity: 0.79
...
```

#### Robustness Testing
The output will show how the model performs under various perturbations:
- Performance degradation curves showing model quality vs noise level
- Critical thresholds where performance significantly drops
- Comparison of different noise types (Gaussian, uniform, targeted)
- Layer-wise robustness rankings

Example terminal output:
```
Robustness test complete.
- Model maintains >90% performance up to noise level σ=0.05
- Performance cliff at noise level σ=0.08 (drops to 42%)
- Most robust component: Feed-forward projection matrices
- Most sensitive component: Attention query matrices
```

#### Comparative Analysis
When comparing different sensitivity methods, expect:
- Similarity matrices showing agreement between methods
- Execution time comparisons
- Analysis of which methods identify which super weights
- Recommendations on which method to use for your specific needs

Generated output will include summaries like:
```
Method comparison results:
- Jaccard similarity scores:
  * Gradient vs Z-score: 0.76
  * Gradient vs Hessian: 0.65
  * Z-score vs Hessian: 0.59
  
- Execution times:
  * Gradient: 45.3s
  * Z-score: 12.7s
  * Hessian: 193.8s
  
- Super weight agreement:
  * All methods agree on 42% of weights
  * 18% uniquely identified by Hessian method
```

#### Pruning Analysis
After running pruning commands, expect:
- Model size comparisons before and after pruning
- Performance metrics on benchmark tasks
- Inference speed improvements
- Sample outputs comparing original and pruned models

For example:
```
Pruning complete:
- Original model size: 498MB
- Pruned model size: 214MB (57% reduction)
- Inference speedup: 2.3x
- Performance impact:
  * WikiText perplexity: +15%
  * GLUE score: -8%
  * Generation coherence: -11%
```

#### Selective Weight Pruning
As shown in the earlier example, you'll see significant differences in model behavior:
- The original model may show repetitive patterns or specific quirks
- The pruned model typically exhibits different generation style and content
- Performance metrics will quantify the changes
- Visualizations will show which components were most affected

In addition to text comparison examples, you'll also see metrics like:
```
Selective pruning of attention value components (80th percentile):
- Weights zeroed: 27,648 (20% of target components)
- Overall model size impact: 0.53% reduction
- Perplexity change: +17.3%
- Generation diversity: +43% (measured by unique n-grams)
- Topic coherence: -12%
```

#### Bit-Level Analysis
This analysis shows how bit-level changes affect model performance:
- Bit position sensitivity rankings
- Critical bits that cause model failure when flipped
- Quantization potential analysis
- Hardware fault simulation results

Example output:
```
Bit-level sensitivity analysis:
- Sign bit flips cause 87% average performance drop
- Most significant 8 bits are critical (>50% performance impact when modified)
- Bits 0-3 (least significant) can be truncated with <5% quality impact
- Quantization recommendation: 8-bit quantization safe for most layers
```
