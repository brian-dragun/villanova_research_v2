import os
import torch
from config import MODEL_PATHS, MODEL_NAME, TEST_PROMPT, EPSILON
from llm_train import train_model
from llm_prune_model import prune_model
from llm_evaluate_models import evaluate_model
from llm_robustness_test import apply_robustness_test
from llm_adversarial_test import test_adversarial_robustness
from llm_integrated_analysis import run_integrated_analysis
from llm_bit_level_and_ablation_analysis import run_bit_level_and_ablation_analysis
from llm_robust_analysis_display import run_robust_analysis_display
from llm_weight_sensitivity_analysis import main as run_weight_sensitivity_experiments
from colorama import Fore, Style
from transformers import AutoModelForCausalLM, AutoTokenizer

def debug_print(message):
    """Print debug messages in yellow."""
    print(Fore.YELLOW + "[DEBUG] " + message + Style.RESET_ALL)

def display_generated_answer(model_name, prompt):
    debug_print(f"Running `display_generated_answer` from {__file__}")

    # Load the model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()

    # Tokenize the input prompt with an attention mask
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate an answer with some generation parameters; adjust as desired.
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return answer

def load_or_train_model(model_dir, skip_finetune=False):
    debug_print(f"Running `load_or_train_model` from {__file__}")

    if not os.path.isdir(model_dir):
        if skip_finetune:
            raise FileNotFoundError(f"Model directory '{model_dir}' not found and skip_finetune is True")
        
        print(f"🚀 No model directory found at '{model_dir}'. Starting fine-tuning...")
        train_model(model_dir)
    else:
        print(f"✅ Found model directory at '{model_dir}', skipping re-training.")

    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    return model

def main(epsilon=EPSILON, prompt=TEST_PROMPT, skip_finetune=False, 
         skip_pruning=False, skip_robustness=False, skip_eval=False,
         skip_adversarial=False, skip_sensitivity=False, skip_bitwise=False,
         skip_display=False, only_step=None, output_dir="outputs", quick_test_mode=False):
    """
    Run the LLM analysis pipeline with given parameters.
    
    Args:
        epsilon: Epsilon value for noise testing
        prompt: Test prompt for model evaluation
        skip_*: Boolean flags to skip specific steps
        only_step: If provided, run only this step
        output_dir: Directory to save outputs
        quick_test_mode: If True, only generate a response to the prompt without other analysis
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # If only_step is specified, skip everything else
    if only_step:
        skip_finetune = only_step != 'finetune'
        skip_pruning = only_step != 'pruning'
        skip_robustness = only_step != 'robustness'
        skip_eval = only_step != 'eval'
        skip_adversarial = only_step != 'adversarial'
        skip_sensitivity = only_step != 'sensitivity'
        skip_bitwise = only_step != 'bitwise'
        skip_display = only_step != 'display'
        
    # If quick_test_mode is True, override most skip flags
    if quick_test_mode:
        # Keep finetune (to load the model) but skip everything else except the prompt test
        skip_pruning = True
        skip_robustness = True
        skip_eval = True
        skip_adversarial = True
        skip_sensitivity = True
        skip_bitwise = True
        skip_display = True

    debug_print(f"Running `main` from {__file__}")
    print("\nUsing model paths:", MODEL_PATHS)
    print("\nEpsilon:", Fore.RED + str(epsilon) + Style.RESET_ALL)
    print("Prompt:", Fore.RED + prompt + Style.RESET_ALL)
    
    if quick_test_mode:
        print(f"{Fore.CYAN}Running in quick test mode (prompt response only){Style.RESET_ALL}")

    model = None
    
    # Step 1: Fine-tuning or Loading
    if not skip_finetune:
        print(Fore.YELLOW + "\n🚀 **Step 1: Fine-tuning (or Loading) the LLM Model**" + Style.RESET_ALL)
        model = load_or_train_model(MODEL_PATHS["finetuned"], skip_finetune)
    else:
        print(Fore.YELLOW + "\n⏩ **Step 1: Fine-tuning/Loading SKIPPED**" + Style.RESET_ALL)

    # Step 2: Pruning
    if not skip_pruning:
        print(Fore.YELLOW + "\n🔍 **Step 2: Pruning the Model**" + Style.RESET_ALL)
        debug_print("Calling `prune_model` from llm_prune_model.py")
        prune_model(MODEL_PATHS["finetuned"], MODEL_PATHS["pruned"])
    else:
        print(Fore.YELLOW + "\n⏩ **Step 2: Pruning SKIPPED**" + Style.RESET_ALL)

    # Step 3: Robustness Testing
    if not skip_robustness:
        print(Fore.YELLOW + "\n🎭 **Step 3: Applying Robustness Test (Adding Noise)**" + Style.RESET_ALL)
        debug_print("Calling `apply_robustness_test` from llm_robustness_test.py")
        apply_robustness_test(MODEL_NAME, MODEL_PATHS["noisy"], epsilon=epsilon)
    else:
        print(Fore.YELLOW + "\n⏩ **Step 3: Robustness Testing SKIPPED**" + Style.RESET_ALL)

    # Step 4: Model Evaluation
    if not skip_eval:
        print(Fore.YELLOW + "\n📊 **Step 4: Evaluating Model Performance (Perplexity)**" + Style.RESET_ALL)
        debug_print("Calling `evaluate_model` from llm_evaluate_models.py")
        
        try:
            if os.path.exists(MODEL_PATHS["finetuned"]):
                evaluate_model(MODEL_PATHS["finetuned"])
            else:
                print(f"⚠️ Skipping evaluation of finetuned model: {MODEL_PATHS['finetuned']} not found")
                
            if os.path.exists(MODEL_PATHS["pruned"]):
                evaluate_model(MODEL_PATHS["pruned"])
            else:
                print(f"⚠️ Skipping evaluation of pruned model: {MODEL_PATHS['pruned']} not found")
                
            if os.path.exists(MODEL_PATHS["noisy"]):
                evaluate_model(MODEL_PATHS["noisy"])
            else:
                print(f"⚠️ Skipping evaluation of noisy model: {MODEL_PATHS['noisy']} not found")
        except Exception as e:
            print(f"❌ Error during model evaluation: {str(e)}")
    else:
        print(Fore.YELLOW + "\n⏩ **Step 4: Model Evaluation SKIPPED**" + Style.RESET_ALL)

    # Step 5: Adversarial Testing
    if not skip_adversarial:
        print(Fore.YELLOW + "\n🛡 **Step 5: Adversarial Testing (FGSM Attack on Embeddings)**" + Style.RESET_ALL)
        debug_print("Calling `test_adversarial_robustness` from llm_adversarial_test.py")
        try:
            adv_text = test_adversarial_robustness(MODEL_NAME, epsilon=epsilon, prompt=prompt)
            print(Fore.YELLOW + "Adversarial generated text (FGSM attack):" + Style.RESET_ALL, adv_text)
        except Exception as e:
            print(f"❌ Error during adversarial testing: {str(e)}")
    else:
        print(Fore.YELLOW + "\n⏩ **Step 5: Adversarial Testing SKIPPED**" + Style.RESET_ALL)

    # Step 6: Integrated Sensitivity Analysis
    if not skip_sensitivity:
        print(Fore.YELLOW + "\n🔍 **Step 6: Integrated Sensitivity and Super Weight Analysis**" + Style.RESET_ALL)
        debug_print("Calling `run_integrated_analysis` from llm_integrated_analysis.py")
        try:
            run_integrated_analysis(input_text=prompt)
        except Exception as e:
            print(f"❌ Error during integrated analysis: {str(e)}")
    else:
        print(Fore.YELLOW + "\n⏩ **Step 6: Integrated Sensitivity Analysis SKIPPED**" + Style.RESET_ALL)

    # Step 7: Bit-level Analysis
    if not skip_bitwise:
        print(Fore.YELLOW + "\n🔍 **Step 7: Bit-level Sensitivity Analysis and Ablation Study**" + Style.RESET_ALL)
        debug_print("Calling `run_bit_level_and_ablation_analysis` from llm_bit_level_and_ablation_analysis.py")
        try:
            run_bit_level_and_ablation_analysis(prompt=prompt)
        except Exception as e:
            print(f"❌ Error during bit-level analysis: {str(e)}")
    else:
        print(Fore.YELLOW + "\n⏩ **Step 7: Bit-level Analysis SKIPPED**" + Style.RESET_ALL)

    # Step 8: Display Analysis
    if not skip_display:
        print(Fore.YELLOW + "\n🔍 **Step 8: Robust Analysis Display**" + Style.RESET_ALL)
        debug_print("Calling `run_robust_analysis_display` from llm_robust_analysis_display.py")
        try:
            run_robust_analysis_display(output_dir=output_dir)
        except Exception as e:
            print(f"❌ Error during robust analysis display: {str(e)}")
    else:
        print(Fore.YELLOW + "\n⏩ **Step 8: Robust Analysis Display SKIPPED**" + Style.RESET_ALL)

    # Always run this step to show model response
    print(Fore.YELLOW + "\n🔍 **Step 9: Generated Answer to the Prompt**" + Style.RESET_ALL)
    debug_print("Calling `display_generated_answer` from this script")
    try:
        answer = display_generated_answer(MODEL_NAME, prompt)
        print("Prompt Response:")
        print(answer)
    except Exception as e:
        print(f"❌ Error generating response: {str(e)}")

    # Step 10: Weight Sensitivity Experiments
    if not skip_sensitivity:
        print(Fore.YELLOW + "\n🔍 **Step 10: Weight Sensitivity Experiments**" + Style.RESET_ALL)
        debug_print("Calling `run_weight_sensitivity_experiments` from llm_weight_sensitivity_analysis.py")
        try:
            run_weight_sensitivity_experiments()
        except Exception as e:
            print(f"❌ Error during weight sensitivity experiments: {str(e)}")
    else:
        print(Fore.YELLOW + "\n⏩ **Step 10: Weight Sensitivity Experiments SKIPPED**" + Style.RESET_ALL)

    print(Fore.GREEN + "\n✅ **All steps completed successfully!**" + Style.RESET_ALL)
    return True

if __name__ == "__main__":
    main()
