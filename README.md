# GRPO Math Reasoning: Training Qwen2.5-1.5B-Instruct with Reinforcement Learning

Training a small language model to improve at math reasoning using Group Relative Policy Optimization (GRPO), the RL technique behind DeepSeek-R1, applied at small scale with LoRA on 8 RTX A4000 GPUs (4 for vLLM generation, 4 for DDP training).

GRPO is a reinforcement learning algorithm for language models that doesn't need a reward model. For each prompt, it generates a group of completions, scores them with rule-based rewards, and computes advantage relative to the group mean. The policy is updated to make above-average completions more likely, with a KL penalty to prevent catastrophic drift from the base model.

## Results

Evaluated on the full GSM8K test set (1,319 examples). "Strict" counts only answers inside `<answer>` tags; "Lenient" falls back to the last number in the output.

| Model | Strict Accuracy | Lenient Accuracy |
|---|---|---|
| Qwen2.5-1.5B-Instruct (baseline) | 1.7% | 38.3% |
| + GRPO (ours) | **48.7%** | **55.0%** |
| **Delta** | **+47.0 pts** | **+16.7 pts** |

The baseline model can solve math problems but doesn't use the `<answer>` tag format, hence the massive gap between its strict (1.7%) and lenient (38.3%) scores. After GRPO training, the model consistently outputs the trained `<reasoning>...</reasoning><answer>...</answer>` XML structure (strict/lenient gap narrowed from 36.6 pts to 6.3 pts). Math reasoning accuracy improved significantly (+16.7 pts lenient).

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

GRPO rapidly teaches the model to follow a structured output format and improves math reasoning. Scaling from a single consumer GPU (2k examples, 1 epoch, +2.5 pts) to a multi-GPU cluster with vLLM (7.5k examples, 2 epochs, 8 generations per prompt) yielded +16.7 pts on lenient accuracy.
## Setup

| Component           | Details |
|---------------------|---|
| **Model**           | Qwen2.5-1.5B-Instruct, LoRA (rank 16, alpha 32) |
| **Algorithm**       | GRPO via TRL's GRPOTrainer |
| **Dataset**         | GSM8K train split (7,473 examples, 2 epochs) |
| **Rewards**         | Correctness (weight 2.0): exact match or partial credit for last-number fallback. Format (weight 0.5): `<reasoning>/<answer>` XML tags. Integer check (weight 0.5): valid number inside `<answer>` tags. |
| **KL penalty**      | beta=0.01 prevents drift from the instruct model |
| **Optimizer**       | PagedAdamW 8-bit, cosine LR schedule, warmup 10%, weight decay 0.1 |
| **Effective batch** | 2 x 8 generations x 4 accumulation = 64 |
| **Hardware**        | 8x NVIDIA RTX A4000 (16GB) -- 4 for vLLM generation, 4 for DDP training |

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

**Evaluate:**
```bash
python eval.py --lenient                          # baseline lenient (full test set)
python eval.py                                    # baseline strict
python eval.py --checkpoint ./grpo-qwen-gsm8k --lenient
python eval.py --checkpoint ./grpo-qwen-gsm8k
```

**Train (single GPU):**
```bash
python train.py --config config_smoke.yaml        # 20-step smoke test
python train.py                                   # full run
```

**Train (multi-GPU with vLLM):**

GRPO's bottleneck is autoregressive generation, not training. Using [vLLM as a separate generation server](https://huggingface.co/docs/trl/main/en/speeding_up_training#vllm-for-fast-generation-in-online-methods) on dedicated GPUs dramatically speeds this up. Set `use_vllm: true` and `vllm_mode: "server"` in the config, then split GPUs between generation and training:

```bash
# Terminal 1: vLLM server on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct

# Terminal 2: DDP training on GPUs 4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 train.py
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


