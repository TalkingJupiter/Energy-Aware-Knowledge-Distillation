# ept_data.py
import os
from typing import List, Optional, cast, Mapping, Any

from datasets import load_dataset, Dataset


def build_dolly_prompt(example: Mapping[str, Any]) -> str:
    """
    Build a clean instruction-style prompt from a Dolly example.

    Fields:
      - instruction
      - context
    We use instruction (+ optional context) and let the model generate the answer.
    """
    instr = (example.get("instruction") or "").strip()
    ctx = (example.get("context") or "").strip()

    if ctx:
        return (
            f"Instruction: {instr}\n\n"
            f"Context: {ctx}\n\n"
            f"Answer:"
        )
    else:
        return f"Instruction: {instr}\n\nAnswer:"


def _dolly_cache_exists(cache_dir: str) -> bool:
    """
    Check if Dolly dataset exists in the HF datasets cache.
    """
    dataset_root = os.path.join(cache_dir, "databricks___databricks-dolly-15k")
    if not os.path.isdir(dataset_root):
        return False

    for sub in os.listdir(dataset_root):
        if "default" in sub:
            return True

    return False


def load_dolly_prompts(
    num_prompts: int = 100,
    seed: int = 42,
    categories: Optional[List[str]] = None,
    cache_dir: str = "~/.cache/huggingface/datasets",
) -> List[str]:
    """
    Load a subset of databricks/databricks-dolly-15k as prompts.

    - Skips download if dataset is already cached.
    - Optionally filters by category.
    - Returns a list of formatted instruction prompts.
    """
    cache_dir = os.path.expanduser(cache_dir)

    if _dolly_cache_exists(cache_dir):
        print("[EPT] Dolly dataset found in cache — reusing.")
    else:
        print("[EPT] Dolly dataset NOT found — downloading...")

    # split='train' -> plain Dataset (cast for Pylance)
    ds = cast(
        Dataset,
        load_dataset(
            "databricks/databricks-dolly-15k",
            split="train",
            cache_dir=cache_dir,
            download_mode="reuse_cache_if_exists",
        ),
    )

    # Optional category filtering
    if categories:
        cat_set = set(categories)
        ds = ds.filter(lambda ex: ex["category"] in cat_set)

    # Shuffle + select subset
    num = min(num_prompts, len(ds))
    ds = ds.shuffle(seed=seed).select(range(num))

    # Tell Pylance each row is a Mapping[str, Any]
    prompts: List[str] = [
        build_dolly_prompt(cast(Mapping[str, Any], ex))
        for ex in ds
    ]
    return prompts
