from __future__ import annotations

import collections
import dataclasses
import json
from typing import Dict, Optional, Union

from . import instructions_registry


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


def test_instruction_following_strict(
    inp: InputExample,
    prompt_to_response: Dict[str, str],
) -> OutputExample:
    response = prompt_to_response[inp.prompt]
    instruction_list = inp.instruction_id_list
    modes = inp.mode_list
    is_following_list = []

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

        if response and response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp: InputExample,
    prompt_to_response: Dict[str, str],
) -> OutputExample:
    response = prompt_to_response[inp.prompt]
    if response is None:
        return OutputExample(
            instruction_id_list=inp.instruction_id_list,
            prompt=inp.prompt,
            response="",
            follow_all_instructions=False,
            follow_instruction_list=[False] * len(inp.instruction_id_list),
        )

    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []
    modes = inp.mode_list
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

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )

