# GRPO Math Reasoning: Training Qwen2.5-1.5B-Instruct with Reinforcement Learning

Training a small language model to improve at math reasoning using Group Relative Policy Optimization (GRPO), the RL technique behind DeepSeek-R1, applied at small scale with QLoRA on a single consumer GPU.

GRPO is a reinforcement learning algorithm for language models that doesn't need a reward model. For each prompt, it generates a group of completions, scores them with rule-based rewards, and computes advantage relative to the group mean. The policy is updated to make above-average completions more likely, with a KL penalty to prevent catastrophic drift from the base model.

## Results

Evaluated on 200 GSM8K test examples. "Strict" counts only answers inside `<answer>` tags; "Lenient" falls back to the last number in the output.

| Model | Strict Accuracy | Lenient Accuracy |
|---|---|---|
| Qwen2.5-1.5B-Instruct (baseline) | 2.0% | 38.5% |
| + GRPO (ours) | **38.5%** | **41.0%** |
| **Delta** | **+36.5 pts** | **+2.5 pts** |

The baseline model can solve math problems but doesn't use the `<answer>` tag format — hence the massive gap between its strict (2%) and lenient (38.5%) scores. After GRPO training, the model almost always outputs the trained `<reasoning>...</reasoning><answer>...</answer>` XML structure, closing that gap entirely. Underlying math accuracy also improved modestly (+2.5 pts).

**Example output (trained model)**
```
<reasoning>
Janet's ducks produce 16 eggs per day. She eats 3 eggs for breakfast each morning,
so that leaves 16 - 3 = 13 eggs. She uses 4 eggs to bake muffins for her friends,
leaving 13 - 4 = 9 eggs to sell at the market. At the market, Janet sells each egg
for $2, so she makes 9 * $2 = $18.
</reasoning>
<answer>18</answer>
```
**Takeaway**

GRPO rapidly teaches the model to follow a structured output format. This is the most visible effect of RL fine-tuning so far. Longer training on answer correctness should yield results Single consumer GPU seems viable. The training + eval runs on an RTX 4060 (8GB) using QLoRA + gradient checkpointing + paged optimizer.

Need to see if tuning rewards better (lesser reward on formatting and more on correctness) allows training on consumer gpu, but good signs so far.
Will train on for longer and on better hardware to see the effect properly.
## Setup

| Component           | Details |
|---------------------|---|
| **Model**           | Qwen2.5-1.5B-Instruct, 4-bit QLoRA (rank 16, alpha 32) |
| **Algorithm**       | GRPO via TRL's GRPOTrainer |
| **Dataset**         | GSM8K train split (2,000 examples, shuffled subset) |
| **Rewards**         | Correctness (weight 2.0): exact match or partial credit for last-number fallback. Format (weight 0.5): `<reasoning>/<answer>` XML tags. Integer check (weight 0.5): valid number inside `<answer>` tags. |
| **KL penalty**      | beta=0.01 prevents drift from the instruct model |
| **Optimizer**       | PagedAdamW 8-bit, cosine LR schedule, warmup 10%, weight decay 0.1 |
| **Effective batch** | 1 x 4 generations x 4 accumulation = 16 |
| **Hardware**        | NVIDIA RTX 4060 (8GB VRAM) |

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

**Evaluate baseline (before training), train with GRPO, evaluate after training:**
```bash
python eval.py --num_samples 200 --lenient       # lenient extraction
python eval.py --num_samples 200                  # strict (answer tags only)

python train.py                                   # full run (~7-8 hours on RTX 4060)
python train.py --config config_smoke.yaml        # 20-step smoke test
=
python eval.py --checkpoint ./grpo-qwen-gsm8k --num_samples 200 --lenient
python eval.py --checkpoint ./grpo-qwen-gsm8k --num_samples 200
```

## Project Structure

```
├── config.yaml          # All hyperparameters
├── config_smoke.yaml    # Quick 20-step smoke test config
├── rewards.py           # Rule-based reward functions (correctness, format, int)
├── train.py             # GRPO training script
├── eval.py              # Batched GSM8K evaluation
├── requirements.txt     # Dependencies
└── README.md
```


