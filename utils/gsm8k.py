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


def format_prompt(question: str) -> str:
    """Format a GSM8K question as a prompt for the model."""
    return (
        f"Solve the following math problem step by step. "
        f"Put your final answer after ####.\n\n"
        f"Question: {question}\n\n"
        f"Solution:"
    )
