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

"""Shared helpers for IFVerify reward managers (ifverify_sg_rank, ifverify_bg, ifverify_sg_score)."""

import os
import textwrap

import numpy as np

_LOW_ACC_K = 1
# Singles: 0 < P_i < this. Pairs: 0 < pair_score < this (both ends exclusive below this cap).
# Used by ifverify_sg_rank, ifverify_bg, and ifverify_sg_score for lowest-pair / lowest-single selection.
_LOW_ACC_ELIGIBLE_MAX = 0.5

# (instruction i, instruction j, pair score); i < j.
LowPairAccEntry = tuple[int, int, float]
# (instruction index, marginal pass rate P_i).
LowAccEntry = tuple[int, float]


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _prompt_gii(info_matrix: np.ndarray) -> float:
    """GII from K×N binary matrix (K rollouts, N instructions)."""
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


def _instruction_coverage_rate(info_matrix: np.ndarray) -> float:
    """Instruction Coverage Rate (ICR): fraction of instructions with ≥1 passing rollout.

    For a K×N binary ``info_matrix`` (rows = rollouts, cols = instructions), ICR is
    (# columns with at least one 1) / N. Returns a value in ``[0, 1]`` (multiply by 100 for %).
    """
    if info_matrix.size == 0:
        return 0.0
    _k_roll, n = info_matrix.shape
    if n <= 0:
        return 0.0
    covered = np.any(info_matrix > 0, axis=0)
    return float(np.sum(covered) / float(n))


def _marginal_pass_rates(info_matrix: np.ndarray) -> np.ndarray:
    """P_i = fraction of rollouts passing instruction i; shape (N,)."""
    if info_matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    k, _n = info_matrix.shape
    if k <= 0:
        return np.zeros(info_matrix.shape[1], dtype=np.float32)
    return (info_matrix.sum(axis=0) / k).astype(np.float32)


def _weighted_instruction_base_reward(row: np.ndarray, p_inst: np.ndarray) -> float:
    """Per-response base reward with instruction-acc-dependent weights.

    For each instruction ``i``, marginal pass rate ``P_i`` (instruction acc in the group)
    defines weight ``1 + (1 - P_i) = 2 - P_i`` when that instruction is satisfied.
    Returns the mean over instructions: ``(1/N) * sum_i row_i * (2 - P_i)``.

    When all ``P_i == 1``, this equals the unweighted fulfilled fraction ``mean(row)``.
    """
    if row.size == 0 or p_inst.size == 0:
        return 0.0
    n = int(row.shape[0])
    if n != int(p_inst.shape[0]):
        return float(np.mean(row))
    rw = row.astype(np.float64)
    p = p_inst.astype(np.float64)
    return float(np.sum(rw * (2.0 - p)) / n)


def _lowest_pair_acc_mask_and_tuples(
    info_matrix: np.ndarray, n_inst: int, k: int = _LOW_ACC_K
) -> tuple[set[tuple[int, int]], tuple[LowPairAccEntry, ...]]:
    """Lowest pair scores: (both pass) / min(pass_i, pass_j), with i < j.

    Equivalent to max(both/pass_i, both/pass_j) when both marginals are positive. For fixed
    ``both``, **balanced** pass counts yield a **smaller** score than **imbalanced** ones (e.g.
    1/9 vs 1/2 for 9/9 vs 16/2 with a single joint hit). Pairs with min(pass_i, pass_j) == 0
    are skipped. Only pairs with ``0 < pair score < _LOW_ACC_ELIGIBLE_MAX`` are candidates.
    """
    if info_matrix.size == 0 or n_inst < 2:
        return set(), ()
    k_roll = int(info_matrix.shape[0])
    if k_roll <= 0:
        return set(), ()
    counts = info_matrix.sum(axis=0)
    scored: list[tuple[float, int, int]] = []
    for i in range(n_inst):
        for j in range(i + 1, n_inst):
            ci = float(counts[i])
            cj = float(counts[j])
            denom = min(ci, cj)
            if denom <= 0:
                continue
            both = float(np.sum(info_matrix[:, i] * info_matrix[:, j]))
            pair_score = both / denom
            if pair_score <= 0 or pair_score >= _LOW_ACC_ELIGIBLE_MAX:
                continue
            scored.append((pair_score, i, j))
    scored.sort(key=lambda t: t[0])
    take = scored[: min(k, len(scored))]
    ranked = tuple((int(i), int(j), float(p)) for p, i, j in take)
    pair_set = {(int(i), int(j)) for p, i, j in take}
    return pair_set, ranked


def _lowest_single_acc_mask_and_tuples(
    p_inst: np.ndarray, n_inst: int, k: int = _LOW_ACC_K
) -> tuple[set[int], tuple[LowAccEntry, ...]]:
    """Lowest marginal pass rates among instructions with ``0 < P_i < _LOW_ACC_ELIGIBLE_MAX``.

    Returns at most ``min(k, #eligible instructions)`` entries.
    """
    if n_inst <= 0 or p_inst.size == 0:
        return set(), ()
    nz = [
        i
        for i in range(n_inst)
        if 0 < float(p_inst[i]) < _LOW_ACC_ELIGIBLE_MAX
    ]
    if not nz:
        return set(), ()
    nz.sort(key=lambda i: float(p_inst[i]))
    take_n = min(k, len(nz))
    order = nz[:take_n]
    inst_set = set(order)
    ranked = tuple((int(i), float(p_inst[i])) for i in order)
    return inst_set, ranked


def _pair_single_bonuses_for_row(
    row: np.ndarray,
    *,
    low_pair_ranked: tuple[LowPairAccEntry, ...],
    low_acc_inst: set[int],
    p_inst: np.ndarray,
    n_inst: int,
    gii_weight: float,
    via_weight: float,
) -> tuple[float, float]:
    pair_bonus = 0.0
    for i, j, p_pair in low_pair_ranked:
        if row[i] > 0 and row[j] > 0:
            pair_bonus += gii_weight * float(1.0 - p_pair)
    single_bonus = 0.0
    for i in range(n_inst):
        if row[i] > 0 and i in low_acc_inst:
            single_bonus += via_weight * float(1.0 - p_inst[i])
    return float(pair_bonus), float(single_bonus)


def _wrap_indent(text: str, width: int, indent: str) -> str:
    wrapped = textwrap.fill(
        text,
        width=width,
        break_long_words=True,
        break_on_hyphens=True,
        replace_whitespace=False,
    )
    lines = wrapped.splitlines() or ["(empty)"]
    return "\n".join(f"{indent}{ln}" for ln in lines)


def _debug_sample_count() -> int:
    raw = os.environ.get("DEBUG_SAMPLES", "3")
    try:
        return max(0, int(raw))
    except ValueError:
        return 3
