import re
from typing import List, Dict, Optional
from datasets import load_dataset


def load_gsm8k(split: str = "train") -> List[Dict]:
    """Load GSM8K dataset and format for use."""
    ds = load_dataset("openai/gsm8k", "main", split=split)
    examples = []
    for i, item in enumerate(ds):
        answer = extract_answer(item["answer"])
        examples.append({
            "id": f"gsm8k_{split}_{i}",
            "question": item["question"],
            "answer": answer,
            "raw_answer": item["answer"],
        })
    return examples


def extract_answer(solution: str) -> float:
    """Extract the numerical answer from GSM8K solution string (after ####)."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", solution)
    if match:
        return float(match.group(1).replace(",", ""))
    raise ValueError(f"Could not extract answer from: {solution}")


def extract_model_answer(text: str) -> Optional[float]:
    """Extract numerical answer from model-generated text."""
    # Try #### pattern first
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    # Try \\boxed{} pattern
    match = re.search(r"\\boxed\{(-?[\d,]+\.?\d*)\}", text)
    if match:
        return float(match.group(1).replace(",", ""))
    # Try last number in text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            return None
    return None


GSM8K_SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer after ####."
)


def format_prompt(question: str, tokenizer=None) -> str:
    """Format a GSM8K question as a prompt for the model.

    If a tokenizer is supplied and it has a chat template configured
    (Instruct / chat-tuned models), apply the chat template with a system
    message that asks for the '#### <answer>' format and a user message
    containing the question. This produces the exact input shape the model
    was fine-tuned on.

    If no tokenizer is given, or the tokenizer has no chat template, fall
    back to a plain "Question: ... Solution:" completion prompt.
    """
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": GSM8K_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return (
        f"{GSM8K_SYSTEM_PROMPT}\n\n"
        f"Question: {question}\n\n"
        f"Solution:"
    )


def compute_format_reward(text: str) -> float:
    """Format reward for GSM8K completions.

    The prompt asks the model to put its final answer after ####. This
    reward is granted independently of correctness — it's a shaping signal
    that teaches the model to produce a parseable answer before it can get
    the arithmetic right. Weighted below correctness in the scorer.

    Returns in [0, 1]:
        1.0  — completion contains '#### <number>' (full format match)
        0.5  — completion has '####' but no parseable number after it
        0.0  — no '####' marker at all

    The total scorer reward combines this with correctness; see
    scorers/verifier_scorer.py for the weighting.
    """
    if "####" not in text:
        return 0.0
    # Full match: #### followed (after optional whitespace) by a number.
    if re.search(r"####\s*-?[\d,]+\.?\d*", text):
        return 1.0
    return 0.5
