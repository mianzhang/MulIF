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

"""Reward manager that adds GII (Group Instruction-following Index) and VIA
(Variance of Instruction-following Across rollouts) as extra rewards for
response groups when the reward function is IFVerify (recast* / advancedif).
"""

from collections import defaultdict
from typing import Any, List, Set

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def _is_ifverify_data_source(data_source: str, ifverify_sources: Set[str]) -> bool:
    """True if this data_source should use IFVerify (and thus can have GII/VIA)."""
    if not data_source:
        return False
    for prefix in ifverify_sources:
        if data_source == prefix or data_source.startswith(prefix):
            return True
    return False


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


@register("ifverify_gii_via")
class IfverifyGiiViaRewardManager(AbstractRewardManager):
    """Like NaiveRewardManager but adds GII and VIA as extra rewards for IFVerify response groups.

    Only applies when data_source is one of the IFVerify sources (e.g. recast*, advancedif).
    For each prompt group (same uid), builds a KxN matrix from follow_instruction_list,
    computes GII and VIA, and adds gii_weight*GII + via_weight*VIA to each response's reward.
    """

    DEFAULT_IFVERIFY_SOURCES = ("recast", "advancedif")

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Any = None,
        reward_fn_key: str = "data_source",
        use_gii_via: bool = True,
        gii_weight: float = 0.1,
        via_weight: float = 0.1,
        ifverify_data_sources: List[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.use_gii_via = use_gii_via
        self.gii_weight = float(gii_weight)
        self.via_weight = float(via_weight)
        self.ifverify_sources = set(
            ifverify_data_sources
            if ifverify_data_sources is not None
            else list(self.DEFAULT_IFVERIFY_SOURCES)
        )

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
        # For GII/VIA: record reward position per index and collect follow_instruction_list
        last_reward_pos: list[int] = []
        uids = data.non_tensor_batch.get("uid")
        data_sources: list[str] = []

        for i in range(len(data)):
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
            data_sources.append(data_source)
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            last_pos = int(valid_response_length - 1)
            last_reward_pos.append(last_pos)
            reward_tensor[i, last_pos] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for k, v in score.items():
                        print(f"[{k}]", v)
                else:
                    print("[score]", score)

        # Add GII and VIA rewards per response group when IFVerify and use_gii_via
        n_items = len(data)
        reward_extra_info["gii"] = [0.0] * n_items
        reward_extra_info["via"] = [0.0] * n_items
        if self.use_gii_via and uids is not None and "follow_instruction_list" in reward_extra_info:
            follow_lists = reward_extra_info["follow_instruction_list"]
            uid_to_indices: dict[Any, list[int]] = defaultdict(list)
            for idx in range(n_items):
                uid_to_indices[uids[idx]].append(idx)

            for uid, indices in uid_to_indices.items():
                if len(indices) == 0:
                    continue
                ds = data_sources[indices[0]]
                if not _is_ifverify_data_source(ds, self.ifverify_sources):
                    continue
                rows = []
                for idx in indices:
                    fl = follow_lists[idx]
                    if not isinstance(fl, (list, tuple)):
                        break
                    rows.append([1 if b else 0 for b in fl])
                else:
                    if len(rows) < 2:
                        continue
                    n_inst = len(rows[0])
                    if n_inst <= 1:
                        continue
                    if not all(len(r) == n_inst for r in rows):
                        continue
                    info_matrix = np.array(rows, dtype=np.float32)
                    gii = _prompt_gii(info_matrix)
                    via = _prompt_via(info_matrix)
                    bonus = self.gii_weight * gii + self.via_weight * via
                    for idx in indices:
                        pos = last_reward_pos[idx]
                        reward_tensor[idx, pos] += bonus
                        reward_extra_info["gii"][idx] = gii
                        reward_extra_info["via"][idx] = via

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor
