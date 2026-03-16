from __future__ import annotations

import logging
import os
import random
import re
import string
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import json
import openai as OpenAI

from . import instructions_util


logger = logging.getLogger(__name__)

_InstructionArgsDtype = Optional[Dict[str, Union[int, str, Sequence[str]]]]

_NUM_KEYWORDS = 2
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500
_NUM_NUMBERS = 6
_NUM_WORD_CYCLE = 30
_MAX_REPEATS = 5
_NUM_KEYWORD_SENTENCE = 20
_NUM_PRONOUNS = 25
_NUM_INCREMENT = 5
_NUM_CONJUNCTIONS = 6


class Instruction:
    def __init__(self, instruction_id, mode):
        self.id = instruction_id
        self.mode = mode

    def build_description(self, **kwargs):
        raise NotImplementedError("`build_description` not implemented.")

    def get_instruction_args(self):
        if self.mode == "rule":
            raise NotImplementedError("`get_instruction_args` not implemented.")

    def get_instruction_args_keys(self):
        if self.mode == "rule":
            raise NotImplementedError("`get_instruction_args_keys` not implemented.")

    def check_following(self, value):
        raise NotImplementedError("`check_following` not implemented.")


class RubricLLMChecker(Instruction):
    def build_description(self, *, description=None, prompt=None, **kwargs):
        self.description = description
        self.prompt = prompt

    def get_instruction_args(self):
        return {"description": self.description}

    def _parse_vllm_decision(self, raw_content: str) -> bool:
        content_lower = raw_content.strip().lower()
        pos_indicators = ("yes", "true", "pass", "followed")
        neg_indicators = ("no", "false", "fail", "not followed")
        pos_last = max(content_lower.rfind(s) for s in pos_indicators)
        neg_last = max(content_lower.rfind(s) for s in neg_indicators)
        if pos_last >= 0 and (neg_last < 0 or pos_last > neg_last):
            return True
        if neg_last >= 0 and (pos_last < 0 or neg_last > pos_last):
            return False
        raise ValueError(f"Cannot determine decision from vLLM response: {raw_content!r}")

    def check_following(self, value):
        model = os.environ.get("OPENAI_RUBRIC_MODEL")
        vllm_base_url = os.environ.get("VLLM_BASE_URL")
        use_vllm = bool(vllm_base_url)

        base_prompt = (
            "Your job is to objectively and fairly assess if the model's response to the user's prompt "
            "correctly follows a specific instruction.\n\n"
            "User's prompt:\n"
            f"{self.prompt}\n\n"
            "Model's response:\n"
            f"{value}\n\n"
            "Instruction:\n"
            f"{self.description}\n\n"
        )
        if use_vllm:
            content = (
                base_prompt
                + "Please choose 'Yes' or 'No' to answer whether the instruction provided is met. "
                "Do not include any other text in your response."
            )
        else:
            content = base_prompt + (
                "Your response should be a JSON blob with the following schema:\n"
                '{"instruction_check": true/false}'
            )
        messages = [{"role": "user", "content": content}]

        if use_vllm:
            client = OpenAI.OpenAI(
                base_url=vllm_base_url,
                api_key="EMPTY",
            )
            max_retries = 50
            for attempt in range(max_retries):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                raw_content = response.choices[0].message.content or ""
                try:
                    return self._parse_vllm_decision(raw_content)
                except ValueError as exc:
                    if attempt < max_retries - 1:
                        logger.warning(
                            "vLLM decision parse failed (attempt %d/%d): %s. Retrying.",
                            attempt + 1,
                            max_retries,
                            exc,
                        )
                    else:
                        raise
        else:
            client = OpenAI.OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content or ""
            payload = json.loads(raw_content)
            if not isinstance(payload, dict) or "instruction_check" not in payload:
                raise ValueError(
                    f"Extracted JSON missing 'instruction_check' key: {payload!r}"
                )
            return bool(payload["instruction_check"])


class RecastWordLengthChecker(Instruction):
    def build_description(
        self,
        *,
        description=None,
        word_num_bottom_top=None,
        around_word_num=None,
        word_length_bar=None,
        word_length_template_type=None,
    ):
        if description is None:
            raise ValueError("description must be provided for RecastWordLengthChecker.")
        self._word_num_bottom_top = (
            word_num_bottom_top if word_num_bottom_top is not None else [0, 10**9]
        )
        self._around_word_num = around_word_num if around_word_num is not None else 0
        self._word_length_bar = word_length_bar if word_length_bar is not None else 0
        self._word_length_template_type = (
            word_length_template_type if word_length_template_type is not None else 2
        )
        self._description_pattern = description
        return self._description_pattern

    def get_instruction_args(self):
        return {
            "word_num_bottom_top": self._word_num_bottom_top,
            "around_word_num": self._around_word_num,
            "word_length_bar": self._word_length_bar,
            "word_length_template_type": self._word_length_template_type,
        }

    def get_instruction_args_keys(self):
        return [
            "word_num_bottom_top",
            "around_word_num",
            "word_length_bar",
            "word_length_template_type",
        ]

    def check_following(self, value):
        return instructions_util.recast_instructions_util.evaluate_word_length(
            value,
            self._word_num_bottom_top,
            self._around_word_num,
            self._word_length_bar,
            self._word_length_template_type,
        )


class RecastSentenceLengthChecker(Instruction):
    def build_description(
        self,
        *,
        description=None,
        sentence_length_template_type=None,
        sentence_length_target=None,
        sentence_length_bottom_top=None,
    ):
        if description is None:
            raise ValueError(
                "description must be provided for RecastSentenceLengthChecker."
            )
        self._sentence_length_template_type = (
            sentence_length_template_type
            if sentence_length_template_type is not None
            else 0
        )
        self._sentence_length_target = (
            sentence_length_target if sentence_length_target is not None else 0
        )
        self._sentence_length_bottom_top = (
            sentence_length_bottom_top
            if sentence_length_bottom_top is not None
            else [0, 10**9]
        )
        self._description_pattern = description
        return self._description_pattern

    def get_instruction_args(self):
        return {
            "sentence_length_template_type": self._sentence_length_template_type,
            "sentence_length_target": self._sentence_length_target,
            "sentence_length_bottom_top": self._sentence_length_bottom_top,
        }

    def get_instruction_args_keys(self):
        return [
            "sentence_length_template_type",
            "sentence_length_target",
            "sentence_length_bottom_top",
        ]

    def check_following(self, value):
        return instructions_util.recast_instructions_util.evaluate_sentence_length(
            value,
            self._sentence_length_template_type,
            self._sentence_length_target,
            self._sentence_length_bottom_top,
        )


class RecastKeywordChecker(Instruction):
    def build_description(self, *, description=None, keyword=None, num=None):
        if description is None:
            raise ValueError("description must be provided for RecastKeywordChecker.")
        self._keyword = keyword if keyword is not None else ""
        self._num = num if num is not None else 0
        self._description_pattern = description
        return self._description_pattern

    def get_instruction_args(self):
        return {"keyword": self._keyword, "num": self._num}

    def get_instruction_args_keys(self):
        return ["keyword", "num"]

    def check_following(self, value):
        return instructions_util.recast_instructions_util.evaluate_keyword(
            value, self._keyword, self._num
        )


class RecastFormatChecker(Instruction):
    def build_description(self, *, description=None, format_type=None):
        if description is None:
            raise ValueError("description must be provided for RecastFormatChecker.")
        self._format_type = format_type if format_type is not None else "NO_FORMAT"
        self._description_pattern = description
        return self._description_pattern

    def get_instruction_args(self):
        return {"format_type": self._format_type}

    def get_instruction_args_keys(self):
        return ["format_type"]

    def check_following(self, value):
        return instructions_util.recast_instructions_util.evaluate_format(
            value, self._format_type
        )


class RecastStartWithChecker(Instruction):
    def build_description(self, *, description=None, start_with_word=None):
        if description is None:
            raise ValueError(
                "description must be provided for RecastStartWithChecker."
            )
        self._start_with_word = start_with_word if start_with_word is not None else ""
        self._description_pattern = description
        return self._description_pattern

    def get_instruction_args(self):
        return {"start_with_word": self._start_with_word}

    def get_instruction_args_keys(self):
        return ["start_with_word"]

    def check_following(self, value):
        return instructions_util.recast_instructions_util.evaluate_start_with(
            value, self._start_with_word
        )


class RecastEndWithChecker(Instruction):
    def build_description(self, *, description=None, end_with_word=None):
        if description is None:
            raise ValueError("description must be provided for RecastEndWithChecker.")
        self._end_with_word = end_with_word if end_with_word is not None else ""
        self._description_pattern = description
        return self._description_pattern

    def get_instruction_args(self):
        return {"end_with_word": self._end_with_word}

    def get_instruction_args_keys(self):
        return ["end_with_word"]

    def check_following(self, value):
        return instructions_util.recast_instructions_util.evaluate_end_with(
            value, self._end_with_word
        )


class RecastEnglishUppercaseChecker(Instruction):
    def build_description(self, *, description=None):
        if description is None:
            raise ValueError(
                "description must be provided for RecastEnglishUppercaseChecker."
            )
        self._description_pattern = description
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        return bool(
            instructions_util.recast_instructions_util.check_english_uppercase(value)
        )


class RecastEnglishLowercaseChecker(Instruction):
    def build_description(self, *, description=None):
        if description is None:
            raise ValueError(
                "description must be provided for RecastEnglishLowercaseChecker."
            )
        self._description_pattern = description
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        return []

    def check_following(self, value):
        return bool(
            instructions_util.recast_instructions_util.check_english_lowercase(value)
        )


class RecastNoPunctuationChecker(Instruction):
    def build_description(self, *, description=None, punctuation_chars=None):
        if description is None:
            raise ValueError(
                "description must be provided for RecastNoPunctuationChecker."
            )
        self._punctuation_chars = (
            punctuation_chars if punctuation_chars is not None else ["，", ",", "﹐"]
        )
        self._description_pattern = description
        return self._description_pattern

    def get_instruction_args(self):
        return {"punctuation_chars": self._punctuation_chars}

    def get_instruction_args_keys(self):
        return ["punctuation_chars"]

    def check_following(self, value):
        return instructions_util.recast_instructions_util.contains_no_punctuation(
            value,
            set(self._punctuation_chars),
        )


# NOTE: For IFVerify-as-RECAST we only need the RECAST instruction family. If you
# later want to support full IFBench / OOD instructions here, you can copy the
# additional checker classes as needed.

