# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward manager that applies GII (Group Instruction-following Index) and VIA
(Variance of Instruction-following Across rollouts) as batch-normalized penalties
for response groups when the reward function is IFVerify recast* / advancedif).
Higher GII/VIA indicate worse exploration and reduce the reward after min--max
normalization within the batch.
"""

import os
import random
import textwrap
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.ifverify.evaluation import compute_score_internal_batch
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _prompt_gii(info_matrix: np.ndarray) -> float:
    """GII from KxN binary matrix (K rollouts, N instructions). See ifverify/calculate_inference_index.py."""
    if info_matrix.size == 0:
        return 0.0
    k, n = info_matrix.shape
    if n <= 1:
        return 0.0
    counts = info_matrix.sum(axis=0)
    p_i = counts / k
    co_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if counts[i] == 0:
            continue
        for j in range(n):
            if counts[j] == 0:
                continue
            if i == j:
                co_matrix[i, i] = p_i[i]
                continue
            both = np.sum(info_matrix[:, i] * info_matrix[:, j])
            co_matrix[i, j] = both / counts[j]
    gii_sum = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            gii_sum += co_matrix[i, i] - co_matrix[i, j]
    return float(gii_sum / (n * (n - 1)))


def _prompt_via(info_matrix: np.ndarray) -> float:
    """Variance of per-instruction pass rates across rollouts (VIA)."""
    if info_matrix.size == 0:
        return 0.0
    k = info_matrix.shape[0]
    if k == 0:
        return 0.0
    pass_rates = info_matrix.sum(axis=0) / k
    return float(np.var(pass_rates))


def _print_gii_via_debug_sample(
    idx: int,
    prompt_str: str,
    mat: np.ndarray | None,
    gii_v: float,
    via_v: float,
    final_r: float,
) -> None:
    """Pretty-print one random debug sample (stdout)."""
    w = 88
    line = "-" * w
    raw_max = int(os.environ.get("IFVERIFY_DEBUG_PROMPT_MAX_CHARS", "2000"))
    if len(prompt_str) > raw_max:
        prompt_show = prompt_str[: max(0, raw_max - 3)] + "..."
    else:
        prompt_show = prompt_str
    prompt_wrapped = textwrap.fill(
        prompt_show,
        width=w - 4,
        break_long_words=True,
        break_on_hyphens=True,
        replace_whitespace=False,
    )
    prompt_lines = "\n".join(f"    {ln}" for ln in prompt_wrapped.splitlines()) or "    (empty)"

    if mat is not None:
        mat_block = np.array2string(
            mat,
            precision=4,
            suppress_small=False,
            floatmode="fixed",
            max_line_width=w - 4,
        )
        mat_lines = "\n".join(f"    {ln}" for ln in mat_block.splitlines())
        mat_note = f"    shape {mat.shape[0]}×{mat.shape[1]}  (rows = rollouts, cols = instructions)"
    else:
        mat_lines = "    (none — GII/VIA skipped: weights 0, or group too small / bad follow lists / no uid)"
        mat_note = ""

    print(line)
    print(f"  ifverify_gii_via  ·  random debug sample  ·  batch index {idx}")
    print(line)
    print("  Prompt")
    print(prompt_lines)
    print()
    print("  Matrix K×N  (binary pass/fail)")
    if mat_note:
        print(mat_note)
    print(mat_lines)
    print()
    print("  Summary")
    print(f"    {'gii':<22} {gii_v:>12.6f}")
    print(f"    {'via':<22} {via_v:>12.6f}")
    print(f"    {'final_reward (last tok)':<22} {final_r:>12.6f}")
    print(line)
    print()


@register("ifverify_gii_via")
class IfverifyGiiViaRewardManager(AbstractRewardManager):
    """Like NaiveRewardManager but adjusts rewards with GII and VIA for IFVerify response groups.

    For each prompt group (same uid), builds a KxN matrix from per-response instruction
    pass/fail (from ``compute_score``), computes GII and VIA, min--max normalizes them
    using batch extrema, and subtracts ``gii_weight * GII_norm + via_weight * VIA_norm``
    from each response's reward (higher GII/VIA indicate worse exploration).
    ``follow_instruction_list`` is consumed internally for GII/VIA and is not returned in
    ``reward_extra_info``.
    """

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Any = None,
        reward_fn_key: str = "data_source",
        gii_weight: float = 0.1,
        via_weight: float = 0.1,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.gii_weight = float(gii_weight)
        self.via_weight = float(via_weight)
        _ = kwargs

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {k: data.non_tensor_batch[k] for k in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)
        already_print_data_sources: dict[str, int] = {}
        # For GII/VIA: per-index pass/fail lists from compute_score (not returned in reward_extra_info)
        last_reward_pos: list[int] = []
        uids = data.non_tensor_batch.get("uid")

        n = len(data)
        score_inputs: list[tuple[str, str, Any, str, dict[str, Any]]] = []
        for i in range(n):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            base_extra = data_item.non_tensor_batch.get("extra_info", {})
            extra_info = dict(base_extra) if base_extra else {}
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            last_pos = int(valid_response_length - 1)
            last_reward_pos.append(last_pos)
            score_inputs.append((prompt_str, response_str, ground_truth, data_source, extra_info))

        max_workers = int(os.environ.get("REWARDS_SCORE_MAX_WORKERS", "32"))
        batch_items = [(score_inputs[i][1], score_inputs[i][2]) for i in range(n)]
        scores = compute_score_internal_batch(
            batch_items,
            strict=True,
            return_verl_reward=True,
            num_workers=max_workers,
        )

        follow_lists_internal: list[Any] = [None] * n
        for i in range(n):
            score = scores[i]
            prompt_str, response_str, ground_truth, data_source, _ = score_inputs[i]

            reward = score["score"]
            for key, value in score.items():
                if key == "follow_instruction_list":
                    follow_lists_internal[i] = value
                    continue
                reward_extra_info[key].append(value)

            last_pos = last_reward_pos[i]
            reward_tensor[i, last_pos] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                for k, v in score.items():
                    print(f"[{k}]", v)

        def _min_max_norm(x: float, lo: float, hi: float) -> float:
            if hi > lo:
                return float((x - lo) / (hi - lo))
            return 0.0

        # Add GII/VIA-based extras per response group.
        n_items = len(data)
        reward_extra_info["gii"] = [0.0] * n_items
        reward_extra_info["via"] = [0.0] * n_items
        reward_extra_info["gii_norm"] = [0.0] * n_items
        reward_extra_info["via_norm"] = [0.0] * n_items
        reward_extra_info["gii_batch_min"] = [0.0] * n_items
        reward_extra_info["gii_batch_max"] = [0.0] * n_items
        reward_extra_info["via_batch_min"] = [0.0] * n_items
        reward_extra_info["via_batch_max"] = [0.0] * n_items
        reward_extra_info["gii_via_penalty"] = [0.0] * n_items
        # Per-group ratio of responses that follow all instructions.
        reward_extra_info["prompt_acc"] = [0.0] * n_items
        # KxN pass/fail matrix used for GII/VIA (shared within each uid group); None if not computed.
        info_matrix_per_idx: list[np.ndarray | None] = [None] * n_items
        # One (gii, via) per computed group for batch min/max.
        batch_gii_via_pairs: list[tuple[float, float]] = []
        if uids is not None:
            follow_lists = follow_lists_internal
            uid_to_indices: dict[Any, list[int]] = defaultdict(list)
            for idx in range(n_items):
                uid_to_indices[uids[idx]].append(idx)

            for _, indices in uid_to_indices.items():
                if len(indices) == 0:
                    continue
                rows = []
                for idx in indices:
                    fl = follow_lists[idx]
                    if not isinstance(fl, (list, tuple)):
                        break
                    rows.append([1 if b else 0 for b in fl])
                else:
                    all_follow_rate = float(np.mean([1.0 if all(r) else 0.0 for r in rows]))
                    for idx in indices:
                        reward_extra_info["prompt_acc"][idx] = all_follow_rate
                    n_inst = len(rows[0])
                    if not all(len(r) == n_inst for r in rows):
                        continue
                    info_matrix = np.array(rows, dtype=np.float32)
                    gii = _prompt_gii(info_matrix)
                    via = _prompt_via(info_matrix)
                    batch_gii_via_pairs.append((gii, via))
                    for idx in indices:
                        reward_extra_info["gii"][idx] = gii
                        reward_extra_info["via"][idx] = via
                        info_matrix_per_idx[idx] = info_matrix

            if batch_gii_via_pairs:
                gii_min = min(g for g, _ in batch_gii_via_pairs)
                gii_max = max(g for g, _ in batch_gii_via_pairs)
                via_min = min(v for _, v in batch_gii_via_pairs)
                via_max = max(v for _, v in batch_gii_via_pairs)
            else:
                gii_min = gii_max = via_min = via_max = 0.0

            for idx in range(n_items):
                reward_extra_info["gii_batch_min"][idx] = gii_min
                reward_extra_info["gii_batch_max"][idx] = gii_max
                reward_extra_info["via_batch_min"][idx] = via_min
                reward_extra_info["via_batch_max"][idx] = via_max

            for idx in range(n_items):
                if info_matrix_per_idx[idx] is None:
                    continue
                gii = reward_extra_info["gii"][idx]
                via = reward_extra_info["via"][idx]
                gii_n = _min_max_norm(gii, gii_min, gii_max)
                via_n = _min_max_norm(via, via_min, via_max)
                reward_extra_info["gii_norm"][idx] = gii_n
                reward_extra_info["via_norm"][idx] = via_n
                # Penalize high GII/VIA (bad exploration): subtract weighted normalized terms.
                penalty = self.gii_weight * gii_n + self.via_weight * via_n
                pos = last_reward_pos[idx]
                reward_tensor[idx, pos] -= penalty
                reward_extra_info["gii_via_penalty"][idx] = penalty

        # Random debug print: prompt, GII/VIA matrix, gii/via, final scalar reward at last token.
        raw_n = os.environ.get("DEBUG_SAMPLES", "3")
        try:
            n_debug = max(0, int(raw_n))
        except ValueError:
            n_debug = 3
        if n_items > 0 and n_debug > 0:
            k = min(n_debug, n_items)
            for idx in sorted(random.sample(range(n_items), k=k)):
                prompt_str = score_inputs[idx][0]
                mat = info_matrix_per_idx[idx]
                gii_v = reward_extra_info["gii"][idx]
                via_v = reward_extra_info["via"][idx]
                last_pos = last_reward_pos[idx]
                final_r = float(reward_tensor[idx, last_pos].item())
                _print_gii_via_debug_sample(idx, prompt_str, mat, gii_v, via_v, final_r)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor
