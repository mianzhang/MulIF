"""Registry of IFVerify instructions (self-contained)."""

from __future__ import annotations

from . import instructions


INSTRUCTION_DICT = {
    "rubric:llm": instructions.RubricLLMChecker,
    "recast:word_length": instructions.RecastWordLengthChecker,
    "recast:sentence_length": instructions.RecastSentenceLengthChecker,
    "recast:keyword": instructions.RecastKeywordChecker,
    "recast:format": instructions.RecastFormatChecker,
    "recast:start_with": instructions.RecastStartWithChecker,
    "recast:end_with": instructions.RecastEndWithChecker,
    "recast:english_uppercase": instructions.RecastEnglishUppercaseChecker,
    "recast:english_lowercase": instructions.RecastEnglishLowercaseChecker,
    "recast:no_punctuation": instructions.RecastNoPunctuationChecker,
}

