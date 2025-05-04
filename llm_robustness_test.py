import torch
import copy
from transformers import AutoModelForCausalLM
from config import MODEL_NAME

def apply_robustness_test(model_name, output_path, noise_std=0.01):
    """
    Applies Gaussian noise to model weights to test robustness.
    Loads a pre-trained model, adds noise, and saves the noisy state dict.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    noisy_model = copy.deepcopy(model)
    for name, param in noisy_model.named_parameters():
        if param.requires_grad:
            param.data += noise_std * torch.randn_like(param)
    torch.save(noisy_model.state_dict(), output_path)
    print(f"✅ Noise added to model. Saved as {output_path}")

if __name__ == "__main__":
    apply_robustness_test(MODEL_NAME, "data/llm_noisy.pth")
