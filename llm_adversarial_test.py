import torch
import torch.nn.functional as F
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME, TEST_PROMPT, EPSILON, get_model_by_key, get_model_config
from tqdm import tqdm
import random
from difflib import SequenceMatcher

def fgsm_attack(embeddings, epsilon, grad):
    """Generate adversarial perturbation using FGSM."""
    perturbation = epsilon * grad.sign()
    return embeddings + perturbation

def test_adversarial_robustness(model_name, epsilon=EPSILON, prompt=TEST_PROMPT, model_config=None, attack_type="gradient"):
    """
    Test the model's robustness against adversarial inputs.
    
    Args:
        model_name: Name or path of the model
        epsilon: Perturbation magnitude for gradient-based attacks
        prompt: Test prompt
        model_config: Configuration for loading the model
        attack_type: Type of attack (gradient, random, token_swap, char_level)
    
    Returns:
        Dictionary with attack results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⏳ Loading model: {model_name}")
    
    if model_config is None:
        model_config = {"trust_remote_code": True}
        
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **model_config)
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Running {attack_type} adversarial attack")
    
    # First get original output as baseline
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        original_outputs = model(**inputs)
        original_tokens = original_outputs.logits.argmax(dim=-1)[0]
        original_text = tokenizer.decode(original_tokens)
    
    print(f"\n📝 Original text from model:\n{original_text}\n")
    
    adversarial_results = {
        "original_text": original_text,
        "attack_type": attack_type,
        "epsilon": epsilon,
        "success": False
    }
    
    # Different attack types
    if attack_type == "gradient":
        # Gradient-based attack (FGSM)
        inputs_embeds = model.get_input_embeddings()(inputs.input_ids).detach().clone()
        inputs_embeds.requires_grad_()
        
        outputs = model(inputs_embeds=inputs_embeds)
        loss = outputs.logits.mean()
        model.zero_grad()
        loss.backward()
        grad = inputs_embeds.grad.data
        
        adv_embeds = fgsm_attack(inputs_embeds, epsilon, grad)
        
        with torch.no_grad():
            adv_outputs = model(inputs_embeds=adv_embeds)
            adv_tokens = adv_outputs.logits.argmax(dim=-1)[0]
            adv_text = tokenizer.decode(adv_tokens)
        
        adversarial_results["adversarial_text"] = adv_text
    
    elif attack_type == "random":
        # Random perturbation attack
        inputs_embeds = model.get_input_embeddings()(inputs.input_ids).detach().clone()
        # Add random noise instead of gradient-based noise
        random_noise = epsilon * (2 * torch.rand_like(inputs_embeds) - 1)
        adv_embeds = inputs_embeds + random_noise
        
        with torch.no_grad():
            adv_outputs = model(inputs_embeds=adv_embeds)
            adv_tokens = adv_outputs.logits.argmax(dim=-1)[0]
            adv_text = tokenizer.decode(adv_tokens)
        
        adversarial_results["adversarial_text"] = adv_text
    
    elif attack_type == "token_swap":
        # Token swap attack
        input_ids = inputs.input_ids[0].tolist()
        
        # Create a copy of input ids
        adversarial_ids = input_ids.copy()
        
        # Randomly swap some tokens (if possible)
        if len(input_ids) > 3:
            num_swaps = max(1, len(input_ids) // 10)  # Swap about 10% of tokens
            for _ in range(num_swaps):
                idx1 = random.randint(0, len(input_ids) - 2)
                idx2 = random.randint(0, len(input_ids) - 2)
                adversarial_ids[idx1], adversarial_ids[idx2] = adversarial_ids[idx2], adversarial_ids[idx1]
        
        with torch.no_grad():
            adv_inputs = torch.tensor([adversarial_ids]).to(device)
            adv_outputs = model(input_ids=adv_inputs)
            adv_tokens = adv_outputs.logits.argmax(dim=-1)[0]
            adv_text = tokenizer.decode(adv_tokens)
        
        adversarial_results["adversarial_text"] = adv_text
        adversarial_results["adversarial_prompt"] = tokenizer.decode(adversarial_ids)
    
    elif attack_type == "char_level":
        # Character-level attack
        def replace_random_chars(text, probability=0.1):
            result = list(text)
            for i in range(len(result)):
                if random.random() < probability:
                    # Replace with a similar-looking character or add noise
                    char_map = {
                        'a': ['@', '4', 'à', 'á'],
                        'e': ['3', 'é', 'è'],
                        'i': ['1', '!', 'í', 'ì'],
                        'o': ['0', 'ó', 'ò'],
                        's': ['5', '$'],
                        't': ['7', '+'],
                        ' ': ['\u200b', '\u200c', '\u200d'],  # Zero-width characters
                    }
                    
                    if result[i].lower() in char_map:
                        result[i] = random.choice(char_map[result[i].lower()])
            return ''.join(result)
        
        adversarial_prompt = replace_random_chars(prompt)
        adversarial_results["adversarial_prompt"] = adversarial_prompt
        
        with torch.no_grad():
            adv_inputs = tokenizer(adversarial_prompt, return_tensors="pt").to(device)
            adv_outputs = model(**adv_inputs)
            adv_tokens = adv_outputs.logits.argmax(dim=-1)[0]
            adv_text = tokenizer.decode(adv_tokens)
        
        adversarial_results["adversarial_text"] = adv_text
    
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    print(f"\n📝 Adversarial text from model:\n{adversarial_results['adversarial_text']}\n")
    
    # Compute text similarity to determine attack success
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()
    
    text_similarity = similarity(original_text, adversarial_results["adversarial_text"])
    adversarial_results["text_similarity"] = text_similarity
    
    # Consider attack successful if the output changed significantly
    adversarial_results["success"] = text_similarity < 0.8
    
    print(f"Text similarity: {text_similarity:.2f}")
    print(f"Attack {'successful' if adversarial_results['success'] else 'failed'}")
    
    return adversarial_results

def test_multiple_attacks(model_name, output_dir, model_config=None):
    """
    Run multiple types of adversarial attacks and compare their effectiveness.
    
    Args:
        model_name: Name or path of the model
        output_dir: Directory to save the results
        model_config: Configuration for loading the model
    
    Returns:
        Dictionary with results of all attacks
    """
    print(f"\n🧪 Testing multiple attack types")
    
    attack_types = ["gradient", "random", "token_swap", "char_level"]
    prompts = [
        "The meaning of life is",
        "I think artificial intelligence will",
        "The best way to solve climate change is",
        "In five years, the world will"
    ]
    
    all_results = {
        "model_name": model_name,
        "attack_results": [],
        "summary": {}
    }
    
    for attack_type in attack_types:
        print(f"\n🔄 Testing {attack_type} attack")
        attack_results = []
        
        for prompt in tqdm(prompts, desc=f"Running {attack_type} attacks"):
            if attack_type == "gradient":
                # For gradient attacks, try different epsilon values
                for epsilon in [0.01, 0.05, 0.1]:
                    result = test_adversarial_robustness(
                        model_name, 
                        epsilon=epsilon,
                        prompt=prompt, 
                        model_config=model_config,
                        attack_type=attack_type
                    )
                    result["prompt"] = prompt
                    attack_results.append(result)
            else:
                # For other attacks, use a single run
                result = test_adversarial_robustness(
                    model_name, 
                    prompt=prompt, 
                    model_config=model_config,
                    attack_type=attack_type
                )
                result["prompt"] = prompt
                attack_results.append(result)
        
        # Calculate success rate for this attack type
        success_count = sum(1 for r in attack_results if r["success"])
        success_rate = success_count / len(attack_results) if attack_results else 0
        
        all_results["attack_results"].extend(attack_results)
        all_results["summary"][attack_type] = {
            "success_rate": success_rate,
            "count": len(attack_results),
            "success_count": success_count
        }
        
        print(f"Success rate for {attack_type}: {success_rate:.2f} ({success_count}/{len(attack_results)})")
    
    # Save results to file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "adversarial_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate visualizations
    plt.figure(figsize=(10, 6))
    attack_types = list(all_results["summary"].keys())
    success_rates = [all_results["summary"][at]["success_rate"] for at in attack_types]
    
    plt.bar(range(len(attack_types)), success_rates, color='royalblue')
    plt.xticks(range(len(attack_types)), attack_types, rotation=45)
    plt.xlabel("Attack Type")
    plt.ylabel("Success Rate")
    plt.title("Adversarial Attack Success Rates")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attack_success_rates.png"))
    plt.close()
    
    # Create a human-readable report
    with open(os.path.join(output_dir, "adversarial_report.txt"), "w") as f:
        f.write("ADVERSARIAL ATTACK REPORT\n")
        f.write("=======================\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY\n")
        f.write("-------\n")
        for attack_type, summary in all_results["summary"].items():
            f.write(f"{attack_type} attack: {summary['success_rate']:.2f} success rate ")
            f.write(f"({summary['success_count']}/{summary['count']})\n")
        f.write("\n")
        
        f.write("ATTACK EXAMPLES\n")
        f.write("--------------\n")
        # Show one example of each attack type
        shown_attacks = set()
        for result in all_results["attack_results"]:
            if result["attack_type"] not in shown_attacks and result["success"]:
                shown_attacks.add(result["attack_type"])
                f.write(f"\n{result['attack_type'].upper()} ATTACK EXAMPLE:\n")
                f.write(f"Prompt: '{result['prompt']}'\n")
                if "adversarial_prompt" in result:
                    f.write(f"Adversarial prompt: '{result['adversarial_prompt']}'\n")
                f.write(f"Original output: '{result['original_text']}'\n")
                f.write(f"Adversarial output: '{result['adversarial_text']}'\n")
                f.write(f"Text similarity: {result['text_similarity']:.2f}\n")
        
        f.write("\n\nRECOMMENDATIONS\n")
        f.write("---------------\n")
        most_successful = max(all_results["summary"].items(), key=lambda x: x[1]["success_rate"])
        f.write(f"The model is most vulnerable to {most_successful[0]} attacks ")
        f.write(f"with a success rate of {most_successful[1]['success_rate']:.2f}.\n")
        f.write("Consider implementing the following defensive measures:\n")
        
        if most_successful[0] == "gradient":
            f.write("1. Adversarial training with gradient-based perturbations\n")
            f.write("2. Input preprocessing to reduce sensitivity to small perturbations\n")
        elif most_successful[0] == "token_swap":
            f.write("1. Implement a preprocessing step that detects unlikely token sequences\n")
            f.write("2. Add a language model quality check before processing inputs\n")
        elif most_successful[0] == "char_level":
            f.write("1. Add character normalization in the tokenization pipeline\n")
            f.write("2. Implement a text cleaning step to remove special characters\n")
        else:
            f.write("1. Implement input validation and cleaning\n")
            f.write("2. Consider ensemble methods to reduce vulnerability\n")
    
    print(f"\n✅ Adversarial testing completed. Results saved to {output_dir}")
    return all_results

def main(model_name=None, output_dir=None, attack_type="gradient", epsilon=EPSILON, 
         run_multiple=True, cpu_only=False):
    """
    Main function for adversarial testing, compatible with run_analysis.py
    
    Args:
        model_name: Name of the model to analyze (or None for default)
        output_dir: Directory to save results (or None for auto-generated)
        attack_type: Type of attack to run (gradient, random, token_swap, char_level)
        epsilon: Perturbation magnitude for gradient-based attacks
        run_multiple: Whether to run multiple attack types for comparison
        cpu_only: Whether to force CPU-only mode
    
    Returns:
        Dictionary with test results
    """
    # Use default model if not specified
    if model_name is None:
        model_name = MODEL_NAME
    else:
        # Get full model name if a short key was provided
        model_name = get_model_by_key(model_name)
    
    # Create output directory if not specified
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/adversarial_{model_name.replace('/', '_')}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📂 Output directory: {output_dir}")
    
    print(f"\n🔍 Running adversarial analysis for model: {model_name}")
    
    # Set CPU-only mode if requested
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["USE_CPU_ONLY"] = "1"
        print("⚠️ Using CPU-only mode for analysis")
    
    # Get model config
    model_config = get_model_config(model_name.split('/')[-1] if '/' in model_name else model_name)
    
    try:
        start_time = time.time()
        
        if run_multiple:
            results = test_multiple_attacks(model_name, output_dir, model_config)
        else:
            # Run a single attack type
            result = test_adversarial_robustness(
                model_name,
                epsilon=epsilon,
                prompt=TEST_PROMPT,
                model_config=model_config,
                attack_type=attack_type
            )
            
            # Save single attack result
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "adversarial_result.json"), "w") as f:
                json.dump({
                    "model_name": model_name,
                    "attack_type": attack_type,
                    "result": result,
                    "execution_time": time.time() - start_time
                }, f, indent=2)
            
            results = {
                "model_name": model_name,
                "attack_type": attack_type,
                "success": result["success"],
                "text_similarity": result["text_similarity"]
            }
        
        print(f"\n✅ Adversarial analysis completed successfully!")
        print(f"📂 Results saved to {output_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during adversarial testing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
