"""IFVerify instruction following evaluation package for VERL reward scoring.

This package reuses the shared `framework/` evaluation stack as a VERL
reward function, suitable for RECAST-style instruction-following datasets.
"""

from __future__ import annotations

from typing import Dict, Any, Union

from .evaluation import evaluate_instruction_following, compute_score_internal


def compute_score(
    solution_str: str,
    ground_truth: Union[str, Dict[str, Any]],
    strict: bool = True,
    return_verl_reward: bool = True,
) -> Dict[str, Any]:
    """Public entry point used by `verl.utils.reward_score.default_compute_score`."""
    return compute_score_internal(
        solution_str, ground_truth, strict=strict, return_verl_reward=return_verl_reward
    )


def compute_score_loose(
    solution_str: str,
    ground_truth: Union[str, Dict[str, Any]],
    return_verl_reward: bool = True,
) -> Dict[str, Any]:
    """Convenience helper that uses the loose evaluation mode."""
    return compute_score(
        solution_str, ground_truth, strict=False, return_verl_reward=return_verl_reward
    )


__all__ = ["compute_score", "compute_score_loose", "evaluate_instruction_following"]

