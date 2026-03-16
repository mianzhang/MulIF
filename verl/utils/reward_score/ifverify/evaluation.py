"""Core evaluation functions for IFVerify instruction following.

This wraps the original evaluation logic under `framework/` so it can be used
as a VERL reward function.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Any, Union, Tuple

from .utils import import_ifverify_modules, MockInputExample


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

    if strict:
        result = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    else:
        result = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)

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


def compute_score_internal(
    solution_str: str,
    ground_truth: Union[str, Dict[str, Any]],
    strict: bool = True,
    return_verl_reward: bool = True,
) -> Dict[str, Any]:
    """Compute the IFVerify instruction following score.

    Expected `ground_truth` schema (as produced by RECAST → VERL conversion):

        {
          "instructions": [...],  # list of RECAST-style instruction dicts
          "prompt": "<optional original prompt string>"
        }
    """
    try:
        cleaned_solution = remove_think_tags(solution_str)

        if isinstance(ground_truth, str):
            gt_data: Dict[str, Any] = json.loads(ground_truth)
        else:
            gt_data = ground_truth

        instructions = gt_data.get("instructions") or []
        prompt = gt_data.get("prompt", "")

        if not instructions:
            raise ValueError("ground_truth must contain a non-empty 'instructions' list.")

        eval_result = evaluate_instruction_following(
            response=cleaned_solution,
            instructions=instructions,
            prompt=prompt,
            strict=strict,
        )

        score = 1.0 if eval_result["follow_all_instructions"] else 0.0

        # Always include follow_instruction_list so reward managers (e.g. GII/VIA) can use it.
        base_out = {
            "score": score,
            "follow_instruction_list": eval_result["follow_instruction_list"],
            "has_error": False,
        }
        if return_verl_reward:
            return base_out
        else:
            return {
                **base_out,
                "follow_all_instructions": eval_result["follow_all_instructions"],
                "num_instructions": eval_result["num_instructions"],
                "num_followed": eval_result["num_followed"],
                "accuracy": eval_result["accuracy"],
                "evaluation_mode": "strict" if strict else "loose",
            }

    except Exception as e:
        print(f"IFVerify Reward Score Error: {e}")
        err_out = {
            "score": 0.0,
            "follow_instruction_list": [],
            "has_error": True,
        }
        if return_verl_reward:
            return err_out
        return {
            **err_out,
            "error": str(e),
            "follow_all_instructions": False,
            "num_instructions": 0,
            "num_followed": 0,
            "accuracy": 0.0,
            "evaluation_mode": "strict" if strict else "loose",
        }

