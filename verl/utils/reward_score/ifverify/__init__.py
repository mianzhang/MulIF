"""IFVerify instruction following evaluation package for VERL reward scoring.

Supports IFBench (rule-based OOD constraints), RECAST, and rubric LLM checks.
Reward metrics follow strict per-instruction checks: accuracy is the fraction
of constraints satisfied; `follow_all_instructions` requires every constraint.
"""

from __future__ import annotations

from typing import Dict, Any, Union

from .evaluation import (
    compute_score_internal_batch,
    evaluate_instruction_following,
)


def compute_score_batch(
    items: list[tuple[str, Union[str, Dict[str, Any]]]],
    strict: bool = True,
    return_verl_reward: bool = True,
    num_workers: int | None = None,
) -> list[Dict[str, Any]]:
    """Batch IFVerify scoring (one async event loop, semaphore-limited like ifverify/run_eval)."""
    return compute_score_internal_batch(
        items,
        strict=strict,
        return_verl_reward=return_verl_reward,
        num_workers=num_workers,
    )


__all__ = [
    "compute_score_batch",
    "evaluate_instruction_following",
]

