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

"""Reward manager for IFVerify groups on a K×N binary ``info_matrix`` (K rollouts, N instructions).

**Low-acc pairs** use score ``both / min(pass_i, pass_j)``, i.e. ``max(both/pass_i, both/pass_j)``
(lowest first). For the same joint count, **balanced** marginals get a **lower** score than
**imbalanced** ones (e.g. 9/9 vs 16/2 with one joint hit → 1/9 vs 1/2), so balanced A/B singles
make the pair rank as **harder** (more “low-acc”) and are prioritized for the reward hook.
**Low-acc singles** must have ``0 < P_i < 0.5``; among those, the lowest ``P_i`` are kept.
**Low-acc pairs** must have ``0 < pair score < 0.5`` (no zero marginal / zero joint-only noise).

**Final reward** (within a uid group, when the matrix is available):

- **1.0** if this response passes **all** instructions.
- Else **0.1** if it passes any **selected** low-acc single (``0 < P_i < 0.5``) or **both**
  instructions of any **selected** low-acc pair (pair score in ``(0, 0.5)``).
- Otherwise **0.0**.

Config ``use_hit_rewards`` (default ``True``): when ``False``, the hit-reward branch is disabled;
tensor reward is **1.0** only when all instructions pass, else **0.0**.

Per uid group, **VSA**, **VIA**, and **ICR** (instruction coverage rate) are logged as
``vsa`` / ``via`` / ``icr`` per sample.
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
    LowAccEntry,
    LowPairAccEntry,
    _LOW_ACC_ELIGIBLE_MAX,
    _LOW_ACC_K,
    _debug_sample_count,
    _instruction_coverage_rate,
    _lowest_pair_acc_mask_and_tuples,
    _lowest_single_acc_mask_and_tuples,
    _marginal_pass_rates,
    _prompt_vsa,
    _prompt_via,
    _truncate,
    _wrap_indent,
)

# Tensor reward when a selected low-acc single or pair is hit but not all instructions pass.
_HIT_REWARD = 0.1


def _low_acc_highlight_reward_for_row(
    row: np.ndarray,
    *,
    n_inst: int,
    low_acc_inst: set[int],
    low_pair_ranked: tuple[LowPairAccEntry, ...],
    use_hit_rewards: bool = True,
) -> float:
    """Reward 1 if all instructions pass; else _HIT_REWARD if use_hit_rewards and low-acc hit."""
    if n_inst <= 0:
        return 0.0
    if bool(np.all(row[:n_inst] > 0)):
        return 1.0
    if not use_hit_rewards:
        return 0.0
    for i in low_acc_inst:
        if 0 <= i < n_inst and row[i] > 0:
            return float(_HIT_REWARD)
    for i, j, _p_pair in low_pair_ranked:
        if 0 <= i < n_inst and 0 <= j < n_inst and row[i] > 0 and row[j] > 0:
            return float(_HIT_REWARD)
    return 0.0


@dataclass(frozen=True)
class IfverifyDebugRollout:
    """One rollout row for structured IFVerify debug stdout."""

    row_index: int
    batch_idx: int
    instruction_pass: tuple[int, ...]
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
    lowest_pair_accs: tuple[LowPairAccEntry, ...] | None
    low_acc_instructions: tuple[LowAccEntry, ...] | None
    use_hit_rewards: bool
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
        f"  ifverify_bg  ·  DEBUG SAMPLE  ·  spotlight batch_idx = {report.sample_batch_idx}"
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
                f"icr = {report.icr:.6f}  (instruction coverage rate)"
            )
        if report.lowest_pair_accs:
            hit_note = (
                f"{_HIT_REWARD} reward if both pass (when not all instructions pass)"
                if report.use_hit_rewards
                else "hit rewards off — info only"
            )
            print(
                f"  lowest pair score pairs (i, j, both/min; 0 < score < {_LOW_ACC_ELIGIBLE_MAX}) "
                f"— {hit_note}:"
            )
            for rank, (i, j, p_pair) in enumerate(report.lowest_pair_accs, start=1):
                print(f"    #{rank}  (i={i}, j={j})  pair_score = {p_pair:.6f}")
        if report.low_acc_instructions:
            hit_note = (
                f"{_HIT_REWARD} reward when idx passes (when not all instructions pass)"
                if report.use_hit_rewards
                else "hit rewards off — info only"
            )
            print(
                f"  lowest-acc instructions (idx, P_i; 0 < P_i < {_LOW_ACC_ELIGIBLE_MAX}) "
                f"— {hit_note}:"
            )
            for rank, (inst_i, p_i) in enumerate(report.low_acc_instructions, start=1):
                print(f"    #{rank}  instruction {inst_i}  P = {p_i:.6f}")
    else:
        print(f"  task_id: {tid_s}")
        print(
            "  (no matrix: uid-group reward skipped — bad follow lists, "
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
    if report.use_hit_rewards:
        final_rule = (
            f"final = 1 if all pass, else {_HIT_REWARD} if selected low-acc single/pair hit, else 0"
        )
    else:
        final_rule = "final = 1 if all pass, else 0"
    print(
        f"  [4] PER-RESPONSE  (base = instruction_acc from scorer; {final_rule})"
    )
    print(f"  {sub}")
    if not report.rollouts:
        print("  (no rollouts in this report)")
    else:
        for r in report.rollouts:
            pass_str = " ".join(str(x) for x in r.instruction_pass)
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
    instruction_acc: list[float],
    score_inputs: list[tuple[str, str, Any, str, dict[str, Any]]],
) -> list[IfverifyDebugRollout]:
    out: list[IfverifyDebugRollout] = []
    if mat is not None and group_indices is not None:
        for row_i, j in enumerate(group_indices):
            last_p = last_reward_pos[j]
            final_r = float(reward_tensor[j, last_p].item())
            row_vec = (
                tuple(int(x) for x in mat[row_i].tolist()) if row_i < mat.shape[0] else ()
            )
            out.append(
                IfverifyDebugRollout(
                    row_index=row_i,
                    batch_idx=j,
                    instruction_pass=row_vec,
                    base_reward=float(instruction_acc[j]),
                    final_reward=final_r,
                    response=score_inputs[j][1],
                )
            )
        return out

    j = spotlight_idx
    last_p = last_reward_pos[j]
    final_r = float(reward_tensor[j, last_p].item())
    out.append(
        IfverifyDebugRollout(
            row_index=0,
            batch_idx=j,
            instruction_pass=(),
            base_reward=float(instruction_acc[j]),
            final_reward=final_r,
            response=score_inputs[j][1],
        )
    )
    return out


@register("ifverify_bg")
class IfverifyBgRewardManager(AbstractRewardManager):
    """IFVerify group rewards: 1.0 all pass; optional hit reward on low-acc single/pair; else 0.

    Groups by ``uid``; builds K×N from ``follow_instruction_list``. Low-acc singles must
    have ``0 < P_i < 0.5``; low-acc pairs must have ``0 <`` pair score ``< 0.5`` (see module).
    When ``use_hit_rewards`` is **True** (default): tensor reward is **1.0** if all instructions
    pass; **0.1** if not but the response hits any selected low-acc single or both ends of a
    selected low-acc pair; otherwise **0.0**. When ``use_hit_rewards`` is **False**, only **1.0**
    (all pass) or **0.0** apply.
    Also logs **VSA**, **VIA**, and **ICR** per group.
    ``follow_instruction_list`` is consumed internally and not returned in
    ``reward_extra_info``.
    """

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Any = None,
        reward_fn_key: str = "data_source",
        use_hit_rewards: bool = True,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.use_hit_rewards = bool(use_hit_rewards)
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
            # is logged as training/instruction_acc.
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
        reward_extra_info["prompt_acc"] = [0.0] * n_items
        reward_extra_info["vsa"] = [0.0] * n_items
        reward_extra_info["via"] = [0.0] * n_items
        reward_extra_info["icr"] = [0.0] * n_items
        info_matrix_per_idx: list[np.ndarray | None] = [None] * n_items
        lowest_pair_accs_per_idx: list[tuple[LowPairAccEntry, ...] | None] = [None] * n_items
        low_acc_instructions_per_idx: list[tuple[LowAccEntry, ...] | None] = [None] * n_items

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
                    vsa = _prompt_vsa(info_matrix)
                    via = _prompt_via(info_matrix)
                    icr = _instruction_coverage_rate(info_matrix)
                    for idx in indices:
                        reward_extra_info["vsa"][idx] = vsa
                        reward_extra_info["via"][idx] = via
                        reward_extra_info["icr"][idx] = icr
                    p_inst = _marginal_pass_rates(info_matrix)
                    _, low_pair_ranked = _lowest_pair_acc_mask_and_tuples(
                        info_matrix, n_inst, k=_LOW_ACC_K
                    )
                    low_acc_inst, low_acc_tuples = _lowest_single_acc_mask_and_tuples(
                        p_inst, n_inst, k=_LOW_ACC_K
                    )

                    for idx in indices:
                        info_matrix_per_idx[idx] = info_matrix
                        lowest_pair_accs_per_idx[idx] = low_pair_ranked
                        low_acc_instructions_per_idx[idx] = low_acc_tuples

                    for g, idx in enumerate(indices):
                        pos = last_reward_pos[idx]
                        fr = _low_acc_highlight_reward_for_row(
                            info_matrix[g],
                            n_inst=n_inst,
                            low_acc_inst=low_acc_inst,
                            low_pair_ranked=low_pair_ranked,
                            use_hit_rewards=self.use_hit_rewards,
                        )
                        reward_tensor[idx, pos] = float(fr)

        n_debug = _debug_sample_count()
        if n_items > 0 and n_debug > 0:
            for idx in sorted(random.sample(range(n_items), k=min(n_debug, n_items))):
                mat = info_matrix_per_idx[idx]
                group_indices = (
                    uid_to_indices[uids[idx]] if mat is not None and uids is not None else None
                )
                rollouts_list = _make_debug_rollouts(
                    spotlight_idx=idx,
                    mat=mat,
                    group_indices=group_indices,
                    last_reward_pos=last_reward_pos,
                    reward_tensor=reward_tensor,
                    instruction_acc=reward_extra_info["instruction_acc"],
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
                    lowest_pair_accs=lowest_pair_accs_per_idx[idx] if has_metrics else None,
                    low_acc_instructions=low_acc_instructions_per_idx[idx] if has_metrics else None,
                    use_hit_rewards=self.use_hit_rewards,
                    rollouts=tuple(rollouts_list),
                )
                print_ifverify_debug_report(report)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        return reward_tensor
