"""
Reward functions for GRPO training on GSM8K.

Three functions, each called by TRL's GRPOTrainer on every batch:
- correctness_reward: Did the model get the right number? (weight 2.0)
- format_reward: Did the model use the <reasoning>/<answer> XML format? (weight 0.5)
- int_reward: Is the extracted answer a valid integer? (weight 0.5)

TRL's GRPOTrainer calls these with keyword args:
  - completions: list of completions. In conversational mode, each completion is
    a list of message dicts like [{"role": "assistant", "content": "..."}].
  - answer: list of ground truth strings (from the dataset's 'answer' column)
  - prompts, completion_ids, trainer_state: also passed but we ignore them via **kwargs
"""

import re


def _get_completion_text(completion) -> str:
    """Extract text from a completion, handling both string and conversational formats."""
    if isinstance(completion, str):
        return completion
    # Conversational format: list of message dicts
    return completion[-1]["content"] if completion else ""


def _extract_xml_answer(text: str) -> str | None:
    """Extract content from <answer>...</answer> tags."""
    if "<answer>" in text:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    return None


def _extract_gt(answer_text: str) -> str:
    """Extract ground truth number from GSM8K '#### number' format."""
    if "####" in answer_text:
        return answer_text.split("####")[1].strip()
    return answer_text.strip()


def correctness_reward(*, completions, answer, **kwargs) -> list[float]:
    """
    Check if model's answer matches ground truth. Rewards:
    1.0: correct answer in <answer> tags (ideal)
    0.5: correct answer found anywhere in text (right math, wrong format)
    0.0: wrong answer or no answer found

    The 0.5 intermediate reward helps for early training, gives the model
    a gradient signal even before it learns the XML format.
    """
    rewards = []
    for completion, gt in zip(completions, answer):
        text = _get_completion_text(completion)
        gt_ans = _extract_gt(gt)

        # Check <answer> tags first (full reward)
        xml_ans = _extract_xml_answer(text)
        if xml_ans is not None:
            try:
                rewards.append(1.0 if float(xml_ans.replace(",", "")) == float(gt_ans.replace(",", "")) else 0.0)
            except ValueError:
                rewards.append(0.0)
            continue

        # No <answer> tags — try to find the last number in text (partial reward)
        numbers = re.findall(r"-?[\d,]+\.?\d*", text)
        if numbers:
            last_num = numbers[-1].replace(",", "")
            try:
                rewards.append(0.5 if float(last_num) == float(gt_ans.replace(",", "")) else 0.0)
            except ValueError:
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    return rewards


def format_reward(*, completions, **kwargs) -> list[float]:
    """
    Check if output follows <reasoning>...</reasoning><answer>...</answer> structure.
    Returns 1.0 if valid XML format, 0.0 otherwise.
    """
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    return [
        1.0 if re.search(pattern, _get_completion_text(c), re.DOTALL) else 0.0
        for c in completions
    ]


def int_reward(*, completions, **kwargs) -> list[float]:
    """
    Check if the content inside <answer> tags is a valid integer/number.
    Returns 1.0 if it's a valid number, 0.0 otherwise.

    Provides gradient signal even when the answer is wrong but the model
    learned to output a number in the right place, to reward partial progress
    toward the correct format.
    """
    rewards = []
    for completion in completions:
        text = _get_completion_text(completion)
        xml_ans = _extract_xml_answer(text)
        if xml_ans is not None:
            # Strip whitespace and check if it's a valid number
            cleaned = xml_ans.replace(",", "").strip()
            try:
                float(cleaned)
                rewards.append(1.0)
            except ValueError:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards