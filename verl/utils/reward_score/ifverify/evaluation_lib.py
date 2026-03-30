from __future__ import annotations

import asyncio
import collections
import dataclasses
import inspect
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import tqdm.asyncio

from . import instructions_registry

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]
    mode_list: list[str]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


async def run_all_evals(
    examples: List[Tuple[Any, str]],
    eval_fn,
    num_workers: int,
) -> List[OutputExample]:
    """Run many strict evaluations with a global concurrency limit (like ifverify/run_eval)."""
    nw = 32 if num_workers is None else int(num_workers)
    sem = asyncio.Semaphore(max(1, nw))

    async def wrapped_task(inp: Any, response: str) -> OutputExample:
        async with sem:
            prompt_to_response = {inp.prompt: response}
            if inspect.iscoroutinefunction(eval_fn):
                return await eval_fn(inp, prompt_to_response)
            return await asyncio.to_thread(eval_fn, inp, prompt_to_response)

    tasks = [wrapped_task(inp, resp) for inp, resp in examples]
    return await tqdm.asyncio.tqdm.gather(
        *tasks, 
        desc="IFVerify reward batch", 
        mininterval=10.0,    # 每 2 秒才刷新一次屏幕（大幅降低终端渲染压力）
        maxinterval=20.0,   # 最长不超过 10 秒刷新一次
        smoothing=0.1       # 降低速度预测的波动感
    )


async def test_instruction_following_strict_async(
    inp: InputExample,
    prompt_to_response: Dict[str, str],
) -> OutputExample:
    response = prompt_to_response[inp.prompt]
    instruction_list = inp.instruction_id_list
    modes = inp.mode_list
    tasks = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id, modes[index])
        inp.kwargs[index] = {
            key: value for key, value in inp.kwargs[index].items() if value is not None
        }

        if instruction_id.startswith("rubric:"):
            instruction.build_description(prompt=inp.prompt, **inp.kwargs[index])
        elif instruction_id.startswith("recast:"):
            instruction.build_description(**inp.kwargs[index])
        else:
            inp.kwargs[index].pop("description", None)
            instruction.build_description(**inp.kwargs[index])

        if instruction_id.startswith("rubric:"):
            tasks.append(instruction.check_following_async(response))
        else:
            tasks.append(asyncio.to_thread(instruction.check_following, response))

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    is_following_list = []
    for instruction_id, result in zip(instruction_list, raw_results):
        if isinstance(result, Exception):
            logger.warning(
                "Instruction check failed; marking as not-followed. instruction_id=%s error=%r",
                instruction_id,
                result,
            )
            is_following_list.append(False)
        else:
            is_following_list.append(bool(result))

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )
