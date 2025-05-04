import os
import time
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
from llm_adversarial_test import test_adversarial_robustness
from config import MODEL_NAME, TEST_PROMPT, EPSILON

def debug_save_plot(filename):
    if os.path.exists(filename):
        print(f"✅ Successfully saved: {filename}")
    else:
        print(f"❌ Error: {filename} not found after saving.")

def run_robust_analysis_display():
    """
    Run robust analysis by generating adversarial text and plotting token distributions.
    """
    print("Running robust analysis display...")

    # Ensure output directory exists
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate unique filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, f"robust_analysis_{timestamp}.png")

    # Generate adversarial text using the FGSM attack
    adv_text = test_adversarial_robustness(MODEL_NAME, epsilon=EPSILON, prompt=TEST_PROMPT)
    print("\nAdversarial generated text (PGD attack):")
    print(f"  {adv_text}\n")

    # Create Token Distribution Plot
    plt.figure(figsize=(8, 4))
    tokens = ["Token1", "Token2", "Token3"]
    probs = [0.04, 0.03, 0.02]
    plt.bar(tokens, probs)
    plt.title("Token Distribution at Position 0")
    plt.xlabel("Tokens")
    plt.ylabel("Probability")

    # Save plot before show
    plt.savefig(filename)
    print(f"✅ Saved plot: {filename}")
    debug_save_plot(filename)

    # Free memory
    plt.close()

def pgd_attack(model, inputs_embeds, epsilon=EPSILON, alpha=0.01, num_iter=40):
    """
    Perform a PGD (Projected Gradient Descent) attack on the input embeddings.
    """
    adv_embeds = inputs_embeds.clone().detach()
    adv_embeds.requires_grad = True

    for _ in range(num_iter):
        outputs = model(inputs_embeds=adv_embeds)
        loss = outputs.logits.mean()  # Use an appropriate loss for your task
        model.zero_grad()
        loss.backward()
        adv_embeds = adv_embeds + alpha * adv_embeds.grad.sign()
        perturbation = torch.clamp(adv_embeds - inputs_embeds, min=-epsilon, max=epsilon)
        adv_embeds = (inputs_embeds + perturbation).detach()
        adv_embeds.requires_grad = True

    return adv_embeds

def compute_fisher_information(model, data_loader, loss_fn):
    """
    Compute an approximate diagonal Fisher Information Matrix for the model parameters.
    """
    fisher_info = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    model.eval()

    for batch in data_loader:
        inputs, targets = batch["inputs"], batch["targets"]
        model.zero_grad()
        outputs = model(**inputs)
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        targets = targets.view(-1)
        loss = loss_fn(logits, targets)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_info[name] += param.grad.detach() ** 2

    num_batches = len(data_loader)
    for name in fisher_info:
        fisher_info[name] /= num_batches

    return fisher_info

def plot_fisher_info(fisher_info):
    """
    Plot mean Fisher Information per parameter.
    """
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, f"fisher_info_{timestamp}.png")

    param_names = list(fisher_info.keys())
    values = [fisher_info[name].mean().item() for name in param_names]

    plt.figure(figsize=(10, 6))
    plt.barh(param_names, values)
    plt.xlabel("Mean Fisher Information")
    plt.title("Fisher Information per Parameter")
    plt.tight_layout()

    # Save before show
    plt.savefig(filename)
    debug_save_plot(filename)
    print(f"✅ Saved plot: {filename}")

    # Free memory
    plt.close()

def plot_token_distribution(outputs, token_idx, tokenizer, top_k=10):
    """
    Plot the top_k token probability distribution for a given token position.
    """
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, f"token_distribution_{timestamp}.png")

    # Get logits for the token at position token_idx
    logits = outputs.logits[0, token_idx]
    probs = torch.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, top_k)

    tokens = [tokenizer.decode([idx]).strip() for idx in topk_indices]

    # Create plot
    plt.figure(figsize=(8, 4))
    plt.bar(tokens, topk_probs.detach().cpu().numpy())
    plt.xlabel("Tokens")
    plt.ylabel("Probability")
    plt.title(f"Top {top_k} token probabilities for token position {token_idx}")

    # Save before show
    plt.savefig(filename)
    debug_save_plot(filename)
    print(f"✅ Saved plot: {filename}")

    # Free memory
    plt.close()

def main():
    """
    Main function to run robust analysis, PGD attack, and Fisher information analysis.
    """
    model_name = MODEL_NAME
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using model:", model_name)
    print("Device:", device)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ### PGD Attack Demonstration ###
    prompt = TEST_PROMPT
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    embeddings = model.get_input_embeddings()(inputs.input_ids).detach().clone()
    embeddings.requires_grad = True

    adv_embeds = pgd_attack(model, embeddings, epsilon=EPSILON, alpha=0.01, num_iter=10)

    adv_outputs = model(inputs_embeds=adv_embeds)
    adv_input_ids = adv_outputs.logits.argmax(dim=-1)
    adv_text = tokenizer.decode(adv_input_ids[0])

    print("\nAdversarial generated text (PGD attack):")
    print(adv_text)

    plot_token_distribution(adv_outputs, token_idx=0, tokenizer=tokenizer, top_k=10)

    ### Fisher Information Demonstration ###
    batch = {"inputs": inputs, "targets": inputs.input_ids}
    data_loader = [batch] * 5  # Simulate 5 batches

    loss_fn = torch.nn.CrossEntropyLoss()
    fisher_info = compute_fisher_information(model, data_loader, loss_fn)

    plot_fisher_info(fisher_info)

if __name__ == "__main__":
    run_robust_analysis_display()
