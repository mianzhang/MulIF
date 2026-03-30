"""Core evaluation functions for IFVerify instruction following.

This wraps the original evaluation logic under `framework/` so it can be used
as a VERL reward function.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple, Union

from .utils import import_ifverify_modules, MockInputExample

logger = logging.getLogger(__name__)


def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from the text."""
    pattern = r"<think>.*?</think>"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)
    return cleaned_text.strip()


def _prepare_instructions(
    instructions: List[Dict[str, Any]]
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """Extract instruction ids, kwargs, and modes from RECAST-style instructions."""
    instruction_ids: List[str] = []
    kwargs_list: List[Dict[str, Any]] = []
    mode_list: List[str] = []
    for inst in instructions:
        instruction_ids.append(inst["instruction_id"])
        kwargs_list.append(inst.get("args", {}) or {})
        mode_list.append(inst.get("mode", "rule"))
    return instruction_ids, kwargs_list, mode_list


def evaluate_instruction_following(
    response: str,
    instructions: List[Dict[str, Any]],
    prompt: str = "",
    strict: bool = True,
) -> Dict[str, Any]:
    """Evaluate whether a response follows the given instructions.

    Args:
        response: The response text to evaluate.
        instructions: List of instruction dicts (RECAST-style), each containing
          at least `instruction_id`, `args`, and `mode`.
        prompt: The original prompt (used by rubric-based instructions).
        strict: Whether to use strict evaluation (True) or loose evaluation (False).
    """
    if not instructions:
        raise ValueError("No instructions provided for IFVerify evaluation.")

    _instructions_registry, evaluation_lib = import_ifverify_modules()
    instruction_ids, kwargs_list, mode_list = _prepare_instructions(instructions)

    inp = MockInputExample(instruction_id_list=instruction_ids, kwargs=kwargs_list, mode_list=mode_list, prompt=prompt)
    prompt_to_response = {prompt: response}

    result = asyncio.run(
        evaluation_lib.test_instruction_following_strict_async(
            inp, prompt_to_response
        )
    )

    return {
        "instruction_id_list": result.instruction_id_list,
        "prompt": result.prompt,
        "response": result.response,
        "follow_all_instructions": result.follow_all_instructions,
        "follow_instruction_list": result.follow_instruction_list,
        "num_instructions": len(instruction_ids),
        "num_followed": sum(result.follow_instruction_list),
        "accuracy": (
            sum(result.follow_instruction_list) / len(instruction_ids)
            if instruction_ids
            else 0.0
        ),
    }


def _output_example_to_reward_dict(
    result: Any,
    instruction_ids: List[str],
) -> Dict[str, Any]:
    """Map `OutputExample` to the same shape as `compute_score_internal` success output."""
    return {
        "instruction_id_list": result.instruction_id_list,
        "prompt": result.prompt,
        "response": result.response,
        "follow_all_instructions": result.follow_all_instructions,
        "follow_instruction_list": result.follow_instruction_list,
        "num_instructions": len(instruction_ids),
        "num_followed": sum(result.follow_instruction_list),
        "accuracy": (
            sum(result.follow_instruction_list) / len(instruction_ids)
            if instruction_ids
            else 0.0
        ),
    }


def compute_score_internal_batch(
    items: List[Tuple[str, Union[str, Dict[str, Any]]]],
    strict: bool = True,
    return_verl_reward: bool = True,
    num_workers: int | None = None,
) -> List[Dict[str, Any]]:
    """Batch IFVerify scoring with one event loop and async strict eval (like ifverify/run_eval).
    """
    if not items:
        return []

    assert strict, "Loose evaluation is not supported for batch scoring."

    _instructions_registry, evaluation_lib = import_ifverify_modules()

    n = len(items)
    slot: List[Dict[str, Any] | None] = [None] * n
    pending: List[Tuple[int, Any, str, List[str]]] = []

    for i, (solution_str, ground_truth) in enumerate(items):
        cleaned_solution = remove_think_tags(solution_str)
        gt_data: Dict[str, Any] = (
            json.loads(ground_truth)
            if isinstance(ground_truth, str)
            else ground_truth
        )
        instructions = gt_data.get("instructions") or []
        prompt = gt_data.get("prompt", "")
        if not instructions:
            raise ValueError(
                "ground_truth must contain a non-empty 'instructions' list."
            )
        instruction_ids, kwargs_list, mode_list = _prepare_instructions(
            instructions
        )
        inp = MockInputExample(
            instruction_id_list=instruction_ids,
            kwargs=kwargs_list,
            mode_list=mode_list,
            prompt=prompt,
        )
        pending.append((i, inp, cleaned_solution, instruction_ids))

    if pending:
        pairs = [(inp, resp) for _, inp, resp, _ in pending]
        if num_workers is None:
            num_workers = int(os.environ.get("REWARDS_SCORE_MAX_WORKERS", "32"))
        outputs = asyncio.run(
            evaluation_lib.run_all_evals(
                pairs,
                evaluation_lib.test_instruction_following_strict_async,
                num_workers,
            )
        )

        for (idx, inp, _cleaned, instruction_ids), out in zip(pending, outputs):
            ev = _output_example_to_reward_dict(out, instruction_ids)
            score = float(ev["accuracy"])
            base_out: Dict[str, Any] = {
                "score": score,
                "follow_instruction_list": ev["follow_instruction_list"],
                "has_error": False,
            }
            if not return_verl_reward:
                base_out = {
                    **base_out,
                    "follow_all_instructions": ev["follow_all_instructions"],
                    "num_instructions": ev["num_instructions"],
                    "num_followed": ev["num_followed"],
                    "accuracy": ev["accuracy"],
                    "evaluation_mode": "strict",
                    "instruction_id_list": ev["instruction_id_list"],
                    "prompt": ev["prompt"],
                    "response": ev["response"],
                }
            slot[idx] = base_out

    return [slot[i] for i in range(n)]
