"""
Evaluate Qwen model on GSM8K test split.

Usage:
    python eval.py                                    # baseline (no training)
    python eval.py --checkpoint ./grpo-qwen-gsm8k     # after GRPO training
    python eval.py --num_samples 50                   # quick check on 50 examples
    python eval.py --batch_size 8                     # batched generation for speed
"""

import argparse
import re

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_name: str, checkpoint_path: str | None = None):
    """Load model in 4-bit. If checkpoint_path given, load LoRA adapter on top."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    if checkpoint_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, checkpoint_path)
        print(f"Loaded LoRA adapter from {checkpoint_path}")

    model.eval()
    return model


def extract_answer(text: str) -> str | None:
    """Extract number from <answer>...</answer> tags in model output."""
    match = re.search(r"<answer>\s*([\d,.\-]+)\s*</answer>", text)
    return match.group(1).replace(",", "").strip() if match else None


def extract_last_number(text: str) -> str | None:
    """Fallback: extract the last number found anywhere in the text."""
    matches = re.findall(r"-?[\d,]+\.?\d*", text)
    return matches[-1].replace(",", "").strip() if matches else None


def extract_gt(answer_text: str) -> str:
    """Extract number from GSM8K '#### number' format."""
    match = re.search(r"####\s*([\d,.\-]+)", answer_text)
    return match.group(1).replace(",", "").strip() if match else answer_text.strip()


def evaluate(
    model, tokenizer, dataset, system_prompt: str,
    batch_size: int = 1, verbose: bool = False, lenient: bool = False,
) -> float:
    """Run 0-shot greedy evaluation with batched generation."""
    correct = 0
    total = len(dataset)

    # Pre-format all prompts
    all_prompts = []
    for example in dataset:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
        ]
        all_prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    # Process in batches
    for batch_start in tqdm(range(0, total, batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, total)
        batch_prompts = all_prompts[batch_start:batch_end]

        # Tokenize batch with left-padding (required for batched generation)
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        # Decode each example in the batch
        input_len = inputs["input_ids"].shape[1]  # same for all (padded to longest)
        for j in range(batch_end - batch_start):
            idx = batch_start + j
            # Slice from end of input (includes padding) — model output starts after
            generated = tokenizer.decode(
                outputs[j][input_len:], skip_special_tokens=True
            )

            pred = extract_answer(generated)
            if pred is None and lenient:
                pred = extract_last_number(generated)
            gt = extract_gt(dataset[idx]["answer"])

            if verbose and idx < 5:
                print(f"\n--- Example {idx+1} ---")
                print(f"Question: {dataset[idx]['question']}...")
                print(f"Model output: {generated}")
                print(f"Extracted: {pred} | Ground truth: {gt}")

            if pred is not None:
                try:
                    if float(pred) == float(gt):
                        correct += 1
                except ValueError:
                    pass

    accuracy = correct / total * 100
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate on GSM8K test split")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--num_samples", type=int, default=None, help="Evaluate on N samples (default: all)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation (default: 4)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Print first 5 model outputs for debugging")
    parser.add_argument("--lenient", action="store_true", help="Fall back to last number in output when <answer> tags missing")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name = config["model_name"]
    system_prompt = config["system_prompt"]

    print(f"Model: {model_name}")
    print(f"Checkpoint: {args.checkpoint or 'None (baseline)'}")
    print(f"Extraction mode: {'lenient (fallback to last number)' if args.lenient else 'strict (<answer> tags only)'}")
    print(f"Batch size: {args.batch_size}")

    model = load_model(model_name, args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset(config["dataset_name"], config["dataset_config"], split="test")
    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} examples...")
    accuracy = evaluate(
        model, tokenizer, dataset, system_prompt,
        batch_size=args.batch_size, verbose=args.verbose, lenient=args.lenient,
    )
    print(f"\nAccuracy: {accuracy:.1f}% ({int(accuracy * len(dataset) / 100)}/{len(dataset)})")


if __name__ == "__main__":
    main()