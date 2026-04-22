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

"""Reward manager for IFVerify groups with a K×N binary ``info_matrix`` (K rollouts, N instructions).

When uid grouping yields a valid ``info_matrix``, ``inst_weight_mode`` selects how the tensor
reward is set from per-instruction passes (marginals ``P_i`` from the group):

- **none** (default): use the verifier score mean only (no exploration weighting).
- **linear**: base mean ``(1/N) * sum_i pass_i * (1 - P_i)`` with per-instruction weight
  ``(1 - P_i)``; no separate exploration term.
- **log**: base mean ``(1/N) * sum_i pass_i * W_i`` with ``W_i = -log(max(P_i, ε))`` (natural log;
  see ``_LOG_INST_WEIGHT_MIN_P`` in ``ifverify_utils``).

Config: ``inst_weight_mode``.
Pairwise add-on config: ``pairwise_alpha`` and ``pairwise_epsilon``.

Per uid group, **VSA**, **VIA**, and **ICR** (instruction coverage rate) are logged as
``vsa`` / ``via`` / ``icr`` per sample. **inst_mixed_portion** is the fraction of instructions
whose marginal pass rate ``P_i`` is strictly between 0 and 1 (marginal ``P_i`` still matters for
linear and log weighting).
"""

import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.ifverify.evaluation import compute_score_internal_batch
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.ifverify_utils import (
    _debug_sample_count,
    _instruction_coverage_rate,
    _marginal_pass_rates,
    _pairwise_synergy_bonus_for_row,
    _prompt_vsa,
    _prompt_via,
    _truncate,
    _weighted_instruction_base_reward,
    _wrap_indent,
)


@dataclass(frozen=True)
class IfverifyDebugRollout:
    """One rollout row for structured IFVerify debug stdout."""

    row_index: int
    batch_idx: int
    instruction_pass: tuple[int, ...]
    exploration_reward: float
    base_reward: float
    final_reward: float
    response: str


@dataclass(frozen=True)
class IfverifyDebugReport:
    """One debug sample: prompt, task_id from extra_info, matrix, and per-rollout rewards."""

    sample_batch_idx: int
    prompt: str
    task_id: Any
    info_matrix: np.ndarray | None
    vsa: float | None
    via: float | None
    icr: float | None
    exploration_reward: float | None
    log_exploration_reward: bool
    rollouts: tuple[IfverifyDebugRollout, ...]


def print_ifverify_debug_report(report: IfverifyDebugReport) -> None:
    """Pretty-print one structured IFVerify VSA/VIA debug sample (stdout)."""
    w = 88
    sep = "=" * w
    sub = "-" * w
    raw_max = int(os.environ.get("IFVERIFY_DEBUG_PROMPT_MAX_CHARS", "2000"))
    resp_max = int(os.environ.get("IFVERIFY_DEBUG_RESPONSE_MAX_CHARS", "2000"))
    text_w = w - 4

    prompt_show = _truncate(report.prompt, raw_max)

    mat = report.info_matrix
    tid_s = repr(report.task_id)

    print(sep)
    print(
        f"  ifverify_sg_score  ·  DEBUG SAMPLE  ·  spotlight batch_idx = {report.sample_batch_idx}"
    )
    print(sep)

    print()
    print("  [1] PROMPT")
    print(f"  {sub}")
    print(_wrap_indent(prompt_show, text_w, "  "))

    print()
    print("  [2] RESPONSE GROUP")
    print(f"  {sub}")
    if mat is not None:
        k, n_inst = mat.shape
        print(f"  task_id: {tid_s}")
        print(f"  size: K = {k} rollouts × N = {n_inst} instructions  (matrix rows = rollouts)")
        if report.vsa is not None and report.via is not None and report.icr is not None:
            print(
                f"  group metrics:  vsa = {report.vsa:.6f}  ·  via = {report.via:.6f}  ·  "
                f"icr = {report.icr:.6f}"
            )
        if report.exploration_reward is not None:
            print(
                f"  exploration_reward (per weighted-inst mode; see inst_weight_mode): "
                f"{report.exploration_reward:.6f}"
            )
    else:
        print(f"  task_id: {tid_s}")
        print(
            "  (no matrix: pair/single group bonus skipped — bad follow lists, "
            "group too small, or missing rollout uid grouping.)"
        )

    print()
    print("  [3] CHECKING RESULTS  (info_matrix: 1 = pass per instruction)")
    print(f"  {sub}")
    if mat is not None:
        mat_block = np.array2string(
            mat,
            precision=1,
            suppress_small=False,
            floatmode="fixed",
            max_line_width=text_w,
        )
        for ln in mat_block.splitlines():
            print(f"  {ln}")
    else:
        print("  (none)")

    print()
    if report.log_exploration_reward:
        print("  [4] PER-RESPONSE  (exploration reward, base reward, final reward)")
    else:
        print("  [4] PER-RESPONSE  (base reward, final reward)")
    print(f"  {sub}")
    if not report.rollouts:
        print("  (no rollouts in this report)")
    else:
        for r in report.rollouts:
            pass_str = " ".join(str(x) for x in r.instruction_pass)
            if report.log_exploration_reward:
                hdr = (
                    f"  row {r.row_index}  batch_idx={r.batch_idx}  "
                    f"explore={r.exploration_reward:>10.6f}  "
                    f"base={r.base_reward:>10.6f}  final={r.final_reward:>10.6f}"
                )
            else:
                hdr = (
                    f"  row {r.row_index}  batch_idx={r.batch_idx}  "
                    f"base={r.base_reward:>10.6f}  final={r.final_reward:>10.6f}"
                )
            print(hdr)
            print(f"         pass: [{pass_str}]")
            resp = _truncate(r.response, resp_max)
            print("         response:")
            print(_wrap_indent(resp, w - 9, "         "))
            print()

    print("  [5] SPOTLIGHT  (this random sample)")
    print(f"  {sub}")
    sp = next((x for x in report.rollouts if x.batch_idx == report.sample_batch_idx), None)
    if sp is not None:
        if report.log_exploration_reward:
            print(
                f"  batch_idx={sp.batch_idx}  row={sp.row_index}  "
                f"explore={sp.exploration_reward:.6f}  "
                f"base={sp.base_reward:.6f}  final={sp.final_reward:.6f}"
            )
        else:
            print(
                f"  batch_idx={sp.batch_idx}  row={sp.row_index}  "
                f"base={sp.base_reward:.6f}  final={sp.final_reward:.6f}"
            )
    else:
        print(f"  batch_idx={report.sample_batch_idx}  (no matching row in group table above)")
    print(sep)
    print()


def _make_debug_rollouts(
    *,
    spotlight_idx: int,
    mat: np.ndarray | None,
    group_indices: list[int] | None,
    last_reward_pos: list[int],
    reward_tensor: torch.Tensor,
    prompt_extra_bonus: list[float],
    exploration_reward: list[float],
    score_inputs: list[tuple[str, str, Any, str, dict[str, Any]]],
) -> list[IfverifyDebugRollout]:
    out: list[IfverifyDebugRollout] = []
    if mat is not None and group_indices is not None:
        for row_i, j in enumerate(group_indices):
            last_p = last_reward_pos[j]
            final_r = float(reward_tensor[j, last_p].item())
            extra = float(prompt_extra_bonus[j])
            row_vec = (
                tuple(int(x) for x in mat[row_i].tolist()) if row_i < mat.shape[0] else ()
            )
            out.append(
                IfverifyDebugRollout(
                    row_index=row_i,
                    batch_idx=j,
                    instruction_pass=row_vec,
                    exploration_reward=float(exploration_reward[j]),
                    base_reward=final_r - extra,
                    final_reward=final_r,
                    response=score_inputs[j][1],
                )
            )
        return out

    j = spotlight_idx
    last_p = last_reward_pos[j]
    final_r = float(reward_tensor[j, last_p].item())
    extra = float(prompt_extra_bonus[j])
    out.append(
        IfverifyDebugRollout(
            row_index=0,
            batch_idx=j,
            instruction_pass=(),
            exploration_reward=float(exploration_reward[j]),
            base_reward=final_r - extra,
            final_reward=final_r,
            response=score_inputs[j][1],
        )
    )
    return out


@register("ifverify_sg_score")
class IfverifySgScoreRewardManager(AbstractRewardManager):
    """IFVerify group rewards via weighted instruction base scores (see module).

    Groups by ``uid``; builds K×N from ``follow_instruction_list``. When grouping is valid,
    **Base reward** follows ``inst_weight_mode``: ``none`` (plain mean pass), ``linear``
    (``mean_i pass_i * (1 - P_i)``), or ``log``
    (``mean_i pass_i * (-log(max(P_i, ε)))``); see ``_weighted_instruction_base_reward``.

    Also logs **VSA**, **VIA**, **ICR**, and **inst_mixed_portion** (share of instructions with
    ``0 < P_i < 1``) per group for monitoring.
    ``follow_instruction_list`` is consumed internally and not returned in
    ``reward_extra_info``.
    """

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Any = None,
        reward_fn_key: str = "data_source",
        inst_weight_mode: str = "none",
        pairwise_alpha: float = 0.0,
        pairwise_epsilon: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        mode = str(inst_weight_mode).lower().strip()
        if mode not in ("none", "linear", "log"):
            raise ValueError(
                f"inst_weight_mode must be 'none', 'linear', or 'log', got {inst_weight_mode!r}"
            )
        self.inst_weight_mode = mode
        # Backward-compatible: ignore deprecated focal_gamma if passed via kwargs.
        kwargs.pop("focal_gamma", None)
        # Backward-compatible: ignore deprecated exp lambda args if still present.
        kwargs.pop("focal_exp_lambda", None)
        kwargs.pop("exp_lambda", None)
        self.pairwise_alpha = float(pairwise_alpha)
        self.pairwise_epsilon = float(pairwise_epsilon)
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
            prompt_str, response_str, ground_truth, data_source, extra_info_row = score_inputs[i]

            reward = score["score"]
            for key, value in score.items():
                if key == "follow_instruction_list":
                    follow_lists_internal[i] = value
                    continue
                reward_extra_info[key].append(value)

            # Per-response instruction accuracy (fraction of constraints satisfied); batch mean
            # is logged as training/instruction_acc. Uid-group instruction weighting overwrites
            # tensor reward when ``inst_weight_mode`` is ``linear`` or ``log``.
            reward_extra_info["instruction_acc"].append(float(score["score"]))

            last_pos = last_reward_pos[i]
            reward_tensor[i, last_pos] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[task_id]", extra_info_row["task_id"])
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                for k, v in score.items():
                    print(f"[{k}]", v)

        n_items = len(data)
        reward_extra_info["pair_bonus"] = [0.0] * n_items
        reward_extra_info["single_bonus"] = [0.0] * n_items
        reward_extra_info["prompt_extra_bonus"] = [0.0] * n_items
        reward_extra_info["prompt_acc"] = [0.0] * n_items
        reward_extra_info["vsa"] = [0.0] * n_items
        reward_extra_info["via"] = [0.0] * n_items
        reward_extra_info["icr"] = [0.0] * n_items
        reward_extra_info["inst_mixed_portion"] = [0.0] * n_items
        info_matrix_per_idx: list[np.ndarray | None] = [None] * n_items

        uid_to_indices: dict[Any, list[int]] | None = None
        if uids is not None:
            follow_lists = follow_lists_internal
            uid_to_indices = defaultdict(list)
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
                    counts = info_matrix.sum(axis=0)
                    vsa = _prompt_vsa(info_matrix)
                    via = _prompt_via(info_matrix)
                    icr = _instruction_coverage_rate(info_matrix)
                    for idx in indices:
                        reward_extra_info["vsa"][idx] = vsa
                        reward_extra_info["via"][idx] = via
                        reward_extra_info["icr"][idx] = icr
                    p_inst = _marginal_pass_rates(info_matrix)
                    if n_inst > 0:
                        mixed = (p_inst > 0) & (p_inst < 1)
                        inst_mixed_portion = float(np.mean(mixed))
                    else:
                        inst_mixed_portion = 0.0
                    for idx in indices:
                        reward_extra_info["inst_mixed_portion"][idx] = inst_mixed_portion

                    for idx in indices:
                        info_matrix_per_idx[idx] = info_matrix

                    for g, idx in enumerate(indices):
                        pos = last_reward_pos[idx]
                        row = info_matrix[g]
                        if self.inst_weight_mode in ("linear", "log") and n_inst > 0:
                            w_base = _weighted_instruction_base_reward(
                                row,
                                p_inst,
                                inst_weight_mode=self.inst_weight_mode,
                            )
                            pair_bonus = _pairwise_synergy_bonus_for_row(
                                row,
                                info_matrix,
                                counts,
                                alpha=self.pairwise_alpha,
                                epsilon=self.pairwise_epsilon,
                            )
                            reward_extra_info["single_bonus"][idx] = float(w_base)
                            reward_extra_info["pair_bonus"][idx] = float(pair_bonus)
                            reward_tensor[idx, pos] = float(w_base + pair_bonus)

        n_debug = _debug_sample_count()
        if n_items > 0 and n_debug > 0:
            for idx in sorted(random.sample(range(n_items), k=min(n_debug, n_items))):
                mat = info_matrix_per_idx[idx]
                group_indices = (
                    uid_to_indices[uids[idx]] if mat is not None and uids is not None else None
                )
                ex_list = reward_extra_info["pair_bonus"]
                rollouts_list = _make_debug_rollouts(
                    spotlight_idx=idx,
                    mat=mat,
                    group_indices=group_indices,
                    last_reward_pos=last_reward_pos,
                    reward_tensor=reward_tensor,
                    prompt_extra_bonus=reward_extra_info["prompt_extra_bonus"],
                    exploration_reward=ex_list,
                    score_inputs=score_inputs,
                )
                has_metrics = mat is not None
                report = IfverifyDebugReport(
                    sample_batch_idx=idx,
                    prompt=score_inputs[idx][0],
                    task_id=score_inputs[idx][4]["task_id"],
                    info_matrix=mat,
                    vsa=float(reward_extra_info["vsa"][idx]) if has_metrics else None,
                    via=float(reward_extra_info["via"][idx]) if has_metrics else None,
                    icr=float(reward_extra_info["icr"][idx]) if has_metrics else None,
                    exploration_reward=None,
                    log_exploration_reward=False,
                    rollouts=tuple(rollouts_list),
                )
                print_ifverify_debug_report(report)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor
