import os
# Set CUDA allocator configuration to help with fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, DownloadConfig
from config import MODEL_NAME

# Get the absolute path to the DeepSpeed config file.
deepspeed_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds_config.json")

def tokenize_function(examples, tokenizer):
    # Tokenize text with truncation, max_length, and padding to max_length.
    output = tokenizer(examples["text"], truncation=True, max_length=64, padding="max_length")
    # For causal language modeling, set labels equal to input_ids.
    output["labels"] = output["input_ids"].copy()
    return output

def train_model(output_dir="data/llm_finetuned"):
    # Clear GPU cache if available.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Ensure the tokenizer has a pad token; if not, use the EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Enable gradient checkpointing to reduce memory usage.
    model.gradient_checkpointing_enable()
    
    # Create a DownloadConfig (default settings).
    download_config = DownloadConfig()
    
    # Load and tokenize the dataset.
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", download_config=download_config)
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # Use a data collator designed for language modeling (mlm=False).
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Set up training arguments with DeepSpeed integration.
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,       # Do not overwrite existing checkpoints.
        num_train_epochs=1,               # Adjust epochs as needed.
        per_device_train_batch_size=1,    # Lower batch size to reduce memory usage.
        gradient_accumulation_steps=8,      # Increase to simulate a larger effective batch size.
        fp16=True,                        # Enable mixed precision training.
        save_steps=500,
        save_total_limit=2,
        deepspeed=deepspeed_config_path  # Provide the absolute path to the DeepSpeed config file.
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Clear cache again before starting training.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    #trainer.train()
    #model.save_pretrained(output_dir)
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(output_dir)
    print(f"✅ Model fine-tuned and saved to {output_dir}")

if __name__ == "__main__":
    train_model()
