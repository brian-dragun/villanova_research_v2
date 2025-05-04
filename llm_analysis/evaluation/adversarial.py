"""
Adversarial Testing Module

This module implements adversarial attacks against LLMs and measures their robustness.
"""

import torch
import numpy as np

from ..core.utils import debug_print, Timer, log_section
from ..config import MODEL_NAME, ADVERSARIAL_CONFIG, EPSILON

def fgsm_attack(model, input_ids, attention_mask, epsilon=EPSILON):
    """
    Fast Gradient Sign Method (FGSM) attack on embeddings.
    
    Args:
        model: The model to attack
        input_ids: Input token IDs
        attention_mask: Attention mask
        epsilon: Attack strength parameter
        
    Returns:
        perturbed_embedding: Adversarially perturbed embedding
    """
    # Get the word embeddings module
    embedding_layer = model.get_input_embeddings()
    
    # Create embeddings from input_ids
    embeddings = embedding_layer(input_ids)
    
    # We need to track gradients of embeddings
    embeddings.requires_grad = True
    
    # Forward pass
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Use the model's own predictions as targets
    targets = logits.argmax(dim=-1)
    
    # Calculate loss
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]), 
        targets.view(-1)
    )
    
    # Backward pass to get gradients on embeddings
    model.zero_grad()
    loss.backward()
    
    # Create adversarial example with FGSM
    with torch.no_grad():
        perturbed_embedding = embeddings + epsilon * embedding_layer.weight.sign()
    
    return perturbed_embedding

def pgd_attack(model, input_ids, attention_mask, epsilon=EPSILON, alpha=0.01, num_steps=10):
    """
    Projected Gradient Descent (PGD) attack on embeddings.
    
    Args:
        model: The model to attack
        input_ids: Input token IDs
        attention_mask: Attention mask
        epsilon: Attack strength parameter
        alpha: Step size for PGD
        num_steps: Number of PGD steps
        
    Returns:
        perturbed_embedding: Adversarially perturbed embedding
    """
    # Get the word embeddings module
    embedding_layer = model.get_input_embeddings()
    
    # Create embeddings from input_ids
    original_embedding = embedding_layer(input_ids).detach()
    
    # Initialize with random noise
    perturbed_embedding = original_embedding + torch.zeros_like(original_embedding).uniform_(-epsilon, epsilon)
    perturbed_embedding = torch.clamp(perturbed_embedding, 
                                     original_embedding - epsilon, 
                                     original_embedding + epsilon)
    
    for step in range(num_steps):
        # Prepare for gradient calculation
        perturbed_embedding.requires_grad = True
        
        # Forward pass
        outputs = model(inputs_embeds=perturbed_embedding, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Use the model's own predictions as targets (to maximize loss)
        targets = logits.argmax(dim=-1)
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), 
            targets.view(-1)
        )
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update with gradient step
        with torch.no_grad():
            grad_sign = perturbed_embedding.grad.sign()
            perturbed_embedding = perturbed_embedding + alpha * grad_sign
            
            # Project back to epsilon ball
            perturbed_embedding = torch.clamp(perturbed_embedding, 
                                            original_embedding - epsilon, 
                                            original_embedding + epsilon)
        
        # Reset gradients
        perturbed_embedding.requires_grad = False
    
    return perturbed_embedding

def test_adversarial_robustness(model_name=MODEL_NAME, epsilon=EPSILON, prompt=None):
    """
    Test model robustness against adversarial attacks.
    
    Args:
        model_name: Name of the model to test
        epsilon: Attack strength parameter
        prompt: Text prompt to use for testing
        
    Returns:
        Dictionary with attack results
    """
    log_section("Adversarial Testing")
    
    # Set default prompt if none provided
    if prompt is None:
        prompt = "Explain the concept of artificial intelligence."
    
    debug_print(f"Testing adversarial robustness on prompt: '{prompt}'")
    
    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Regular inference for comparison
    with torch.no_grad():
        original_output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )
        original_text = tokenizer.decode(original_output[0], skip_special_tokens=True)
        
    # FGSM Attack
    with Timer("FGSM attack"):
        perturbed_embedding = fgsm_attack(
            model, 
            inputs["input_ids"], 
            inputs["attention_mask"], 
            epsilon
        )
        
        # Generate from perturbed embedding
        outputs_fgsm = model.generate(
            inputs_embeds=perturbed_embedding,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            do_sample=False
        )
        fgsm_text = tokenizer.decode(outputs_fgsm[0], skip_special_tokens=True)
    
    # PGD Attack
    with Timer("PGD attack"):
        perturbed_embedding_pgd = pgd_attack(
            model, 
            inputs["input_ids"], 
            inputs["attention_mask"], 
            epsilon,
            alpha=ADVERSARIAL_CONFIG["pgd_alpha"],
            num_steps=ADVERSARIAL_CONFIG["pgd_steps"]
        )
        
        # Generate from perturbed embedding
        outputs_pgd = model.generate(
            inputs_embeds=perturbed_embedding_pgd,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            do_sample=False
        )
        pgd_text = tokenizer.decode(outputs_pgd[0], skip_special_tokens=True)
    
    # Print outputs for comparison
    debug_print(f"\nOriginal output: {original_text}")
    debug_print(f"\nFGSM attack output: {fgsm_text}")
    debug_print(f"\nPGD attack output: {pgd_text}")
    
    # Calculate similarity between original and attacked outputs
    from difflib import SequenceMatcher
    fgsm_similarity = SequenceMatcher(None, original_text, fgsm_text).ratio()
    pgd_similarity = SequenceMatcher(None, original_text, pgd_text).ratio()
    
    results = {
        "prompt": prompt,
        "epsilon": epsilon,
        "original_output": original_text,
        "fgsm_output": fgsm_text,
        "pgd_output": pgd_text,
        "fgsm_similarity": fgsm_similarity,
        "pgd_similarity": pgd_similarity
    }
    
    # Log results
    debug_print(f"FGSM similarity to original: {fgsm_similarity:.2%}")
    debug_print(f"PGD similarity to original: {pgd_similarity:.2%}")
    
    return results