import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from config import MODEL_NAME, TEST_PROMPT

ssize = 10

def compute_hessian_diagonal_subset(loss_fn, param_data, subset_size=ssize, epsilon=1e-5):
    """
    Compute the diagonal of the Hessian for only a subset of parameters
    using a finite difference approximation.
    """
    flat_param = param_data.view(-1)
    diag = torch.zeros(subset_size, device=flat_param.device)
    base_loss = loss_fn(flat_param)
    for i in range(subset_size):
        orig = flat_param[i].item()
        flat_param[i] = orig + epsilon
        loss_plus = loss_fn(flat_param)
        flat_param[i] = orig - epsilon
        loss_minus = loss_fn(flat_param)
        flat_param[i] = orig  # Restore original value
        diag[i] = (loss_plus - 2 * base_loss + loss_minus) / (epsilon ** 2)
    return diag

def compute_hessian_sensitivity(model, input_text, device=torch.device("cpu"), subset_size=ssize):
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Focus on a smaller parameter: lm_head.weight
    if not hasattr(model, "lm_head"):
        raise ValueError("Model does not have an 'lm_head' attribute.")
    param = model.lm_head.weight
    param_data = param.data.clone().detach().requires_grad_(True)

    def loss_fn(param_vec):
        reshaped = param_vec.view_as(param)
        with torch.no_grad():
            model.lm_head.weight.copy_(reshaped)
        outputs = model(**inputs)
        logits = outputs.logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), inputs["input_ids"].view(-1))

    try:
        print("[DEBUG] Computing Hessian diagonal for first {} elements of lm_head.weight...".format(subset_size))
        diag = compute_hessian_diagonal_subset(loss_fn, param_data, subset_size=subset_size)
        sensitivity_score = diag.abs().sum().item()
        return {"lm_head.weight (first {} elements)".format(subset_size): sensitivity_score}
    except RuntimeError as e:
        print(f"Diagonal Hessian computation failed: {e}")
        print("Falling back to gradient-based sensitivity...")
        try:
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else outputs.logits.mean()
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
            grad_sensitivity = {}
            for (name, _), grad in zip(model.named_parameters(), grads):
                if grad is not None:
                    grad_sensitivity[name] = grad.abs().sum().item()
            return grad_sensitivity
        except RuntimeError as e_grad:
            print(f"Gradient-based sensitivity failed: {e_grad}")
            return {}

def plot_sensitivity(sensitivity_scores):
    if not sensitivity_scores:
        print("No sensitivity scores to plot.")
        return
    names = list(sensitivity_scores.keys())
    scores = [sensitivity_scores[name] for name in names]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(names)), scores)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.xlabel("Parameter")
    plt.ylabel("Sensitivity Score")
    plt.title("LLM Weight Sensitivity Scores (Subset Hessian Diagonal)")
    plt.tight_layout()
    plt.savefig("llm_diagram_sensitivity_plot.png")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(device)
    test_text = TEST_PROMPT
    sensitivity_scores = compute_hessian_sensitivity(model, test_text, device=device, subset_size=ssize)
    print("Sensitivity Scores:")
    for name, score in sensitivity_scores.items():
        print(f"{name}: {score:.4f}")
    plot_sensitivity(sensitivity_scores)
