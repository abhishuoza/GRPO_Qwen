"""
GRPO training script for Qwen2.5-1.5B-Instruct on GSM8K.

This trains a small language model to do math reasoning using Group Relative
Policy Optimization (GRPO) — the same RL algorithm behind DeepSeek-R1, but
at small scale with QLoRA for memory efficiency.

How GRPO works:
1. For each math problem, generate a group of completions (num_generations=4)
2. Score each completion with reward functions (correctness + format)
3. Compute advantage within the group (subtract mean, divide by std)
4. Update policy to increase probability of above-average completions
5. KL penalty (beta) prevents the model from drifting too far from its starting point

No reward model needed — rewards are rule-based (did it get the right answer?).

Usage:
    python train.py
    python train.py --config path/to/config.yaml
"""

import argparse

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from rewards import correctness_reward, format_reward, int_reward


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_dataset(config: dict):
    """
    Load GSM8K train split and format for GRPOTrainer.

    GRPOTrainer expects a dataset with a `prompt` column containing chat-format
    messages (list of dicts with role/content). Any other columns are passed as
    kwargs to reward functions — we keep `answer` so correctness_reward can
    compare against ground truth.
    """
    dataset = load_dataset(
        config["dataset_name"], config["dataset_config"], split="train"
    )

    system_prompt = config["system_prompt"]

    def format_prompt(example):
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["question"]},
            ],
        }

    dataset = dataset.map(format_prompt)
    dataset = dataset.select_columns(["prompt", "answer"])
    return dataset


def load_model_and_tokenizer(config: dict):
    """
    Load model in 4-bit with BitsAndBytesConfig.
    Falls back to smaller model if OOM.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, config["bnb_4bit_compute_dtype"]),
    )

    model_name = config["model_name"]
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",  # safer with 4-bit than flash_attention
        )
    except torch.cuda.OutOfMemoryError:
        print(f"OOM loading {model_name}, falling back to {config['fallback_model_name']}")
        model_name = config["fallback_model_name"]
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, model_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Training with config from {args.config}")

    # 1. Dataset
    print("Loading GSM8K dataset...")
    dataset = prepare_dataset(config)
    max_samples = config.get("max_train_samples")
    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
    print(f"Training on {len(dataset)} examples")

    # 2. Model + tokenizer
    print(f"Loading model: {config['model_name']}...")
    model, tokenizer, model_name = load_model_and_tokenizer(config)
    print(f"Loaded {model_name} in 4-bit")

    # 3. LoRA config — passed to GRPOTrainer, not applied manually
    peft_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        task_type="CAUSAL_LM",
    )

    # 4. GRPO training arguments
    training_args = GRPOConfig(
        output_dir=config["output_dir"],
        num_train_epochs=config.get("num_train_epochs", 1),
        max_steps=config.get("max_steps", -1),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        max_completion_length=config["max_completion_length"],
        # max_prompt_length removed — not a GRPOConfig param; prompts are truncated by tokenizer
        num_generations=config["num_generations"],
        beta=config["beta"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        gradient_checkpointing=config["gradient_checkpointing"],
        optim=config["optim"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        max_grad_norm=config["max_grad_norm"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config.get("weight_decay", 0.0),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        logging_dir=f"{config['output_dir']}/logs",
        report_to="none",
        use_vllm=config.get("use_vllm", False),
        generation_batch_size=config.get("generation_batch_size", config["num_generations"]),
        reward_weights=[
            config["correctness_reward_weight"],
            config["format_reward_weight"],
            config["int_reward_weight"],
        ],
    )

    # 5. Create trainer
    #    reward_funcs: list of functions called on each batch
    #    reward_weights are in GRPOConfig above
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward, format_reward, int_reward],
    )

    # 6. Train
    print("Starting GRPO training...")
    print(f"  Effective batch size: {config['per_device_train_batch_size']} × {config['num_generations']} generations × {config['gradient_accumulation_steps']} accum = {config['per_device_train_batch_size'] * config['num_generations'] * config['gradient_accumulation_steps']}")
    print(f"  KL penalty (beta): {config['beta']}")
    print(f"  Reward weights: correctness={config['correctness_reward_weight']}, format={config['format_reward_weight']}, int={config['int_reward_weight']}")

    trainer.train(resume_from_checkpoint=True)

    # 7. Save
    print(f"Saving to {config['output_dir']}...")
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print("Done!")


if __name__ == "__main__":
    main()
