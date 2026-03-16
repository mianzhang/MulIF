import functools
import json
import re
from typing import Tuple


_EN_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_EN_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_JSON_FULL_RE = re.compile(r"^\s*(\{.*\}|\[.*\])\s*$", re.DOTALL)
_JSON_PARTIAL_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)
_MARKDOWN_PATTERNS = [
    re.compile(r"^#\s+.+", re.MULTILINE),
    re.compile(r"^[*+-]\s+.+", re.MULTILINE),
    re.compile(r"^\d+\.\s+.+", re.MULTILINE),
    re.compile(r"!?\[.*\]\(.+\)", re.MULTILINE),
    re.compile(r"^\s*`{3}[\s\S]*?`{3}", re.MULTILINE),
    re.compile(r"^\s*\|.+\|", re.MULTILINE),
    re.compile(r"^\s*>+.+", re.MULTILINE),
]
_ENGLISH_LETTER_PUNCT_RE = re.compile(r'^[a-zA-Z\s\.,!?;:\'"-]+$')
_ENGLISH_LETTER_RE = re.compile(r"[a-zA-Z]")
_EXTRACT_WORDS_RE = re.compile(
    r"[\w\u0400-\u04FF\u00C0-\u017F\u3040-\u30FF\u4E00-\u9FFF]+(?:['-][\w\u0400-\u04FF\u00C0-\u017F\u3040-\u30FF\u4E00-\u9FFF]+)*",
    re.UNICODE,
)


@functools.lru_cache(maxsize=2048)
def robust_detect_lang(text):
    _ = text
    return "en"


def normalize_lang_code(lang_code):
    return lang_code.lower().split("-")[0]


def count_words(text, language):
    if not text.strip():
        return 0
    _ = language
    return len(_EN_WORD_RE.findall(text))


def count_sentences(text, language):
    if not text.strip():
        return 0
    text = text.replace("\n", " ")
    _ = language
    sentences = [s for s in _EN_SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return len(sentences)


def detect_json(text):
    if _JSON_FULL_RE.fullmatch(text):
        try:
            json.loads(text)
            return True
        except Exception:
            pass
    elif _JSON_PARTIAL_RE.search(text):
        try:
            json_str = _JSON_PARTIAL_RE.search(text).group(1)
            json.loads(json_str)
            return True
        except Exception:
            pass
    return False


def detect_markdown(text):
    return any(p.search(text) for p in _MARKDOWN_PATTERNS)


def detect_xml_html(text):
    xml_declaration = r"<\?xml\s+version="
    xml_tag = r"<([a-z][a-z0-9]*)(?:\s+[^>]*)?>.*?</\1>"

    html5_self_closing = r"<(img|br|hr|input|meta|link|source|track|embed|wbr)\b[^>]*/?>"
    html5_attrs = r"\s(data-|aria-)[a-z-]+="

    html_doctype = r"<!DOCTYPE\s+html>"
    html_tags = r"<(html|head|body|div|span|a|p)\b"
    html_self_closing = r"<(br|hr|img|input|meta|link)\b[^>]*>"
    html_attr = r'\s(id|class|style)=["\'][^"\']*["\']'

    if re.search(xml_declaration, text, re.IGNORECASE):
        return "xml"
    if re.search(html_doctype, text, re.IGNORECASE):
        return "html"

    xml_score = 0
    html_score = 0

    if re.search(xml_tag, text, re.DOTALL | re.IGNORECASE):
        xml_score += 2
    if "<?xml" in text.lower():
        xml_score += 3

    if re.search(html5_self_closing, text, re.IGNORECASE):
        html_score += 2
    if re.search(html5_attrs, text, re.IGNORECASE):
        html_score += 1

    if re.search(html_tags, text, re.IGNORECASE):
        html_score += 2
    if re.search(html_self_closing, text, re.IGNORECASE):
        html_score += 1
    if re.search(html_attr, text):
        html_score += 1
    if "<html" in text.lower():
        html_score += 3

    if xml_score > html_score and xml_score >= 2:
        return "xml"
    elif html_score > xml_score and html_score >= 2:
        return "html"
    elif xml_score == html_score and xml_score >= 2:
        return "html"
    return None


def detect_text_format(text):
    text = text.strip()
    if not text:
        return 0
    if detect_json(text):
        return 1
    if detect_markdown(text):
        return 2
    xml_html_type = detect_xml_html(text)
    if xml_html_type == "xml":
        return 3
    elif xml_html_type == "html":
        return 4
    return 0


def count_substring(main_str, sub_str):
    if not sub_str:
        return 0
    return main_str.count(sub_str)


def extract_first_last_words(text: str) -> Tuple[str, str]:
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text.strip())
    if not text:
        return "", ""
    words = _EXTRACT_WORDS_RE.findall(text)
    if not words:
        return "", ""
    first_word = words[0]
    last_word = words[-1]
    punctuation = r""".!?,;:'"‘’“”«»„‟()[]{}<>~`@#$%^&*_\-+=|\\/…¿¡°•·"""
    first_word = first_word.strip(punctuation)
    last_word = last_word.strip(punctuation)
    return first_word, last_word


def check_english_uppercase(text):
    if not text.strip():
        return 0
    if not _ENGLISH_LETTER_PUNCT_RE.fullmatch(text):
        return 0
    letters = _ENGLISH_LETTER_RE.findall(text)
    if not letters:
        return 0
    return 1 if all(c.isupper() for c in letters) else 0


def check_english_lowercase(text):
    if not text.strip():
        return False
    if not _ENGLISH_LETTER_PUNCT_RE.fullmatch(text):
        return False
    letters = _ENGLISH_LETTER_RE.findall(text)
    if not letters:
        return False
    return all(c.islower() for c in letters)


def contains_no_punctuation(text, punctuation_chars=None):
    if punctuation_chars is None:
        punctuation_chars = {"，", ",", "﹐"}
    return not any(punc in text for punc in punctuation_chars)


def evaluate_start_with(response, strat_with_word):
    first_word, _ = extract_first_last_words(response)
    return first_word == strat_with_word


def evaluate_end_with(response, end_with_word):
    _, last_word = extract_first_last_words(response)
    return end_with_word == last_word


def evaluate_keyword(response, keyword, num):
    count_num = count_substring(response, keyword)
    return count_num == num


def evaluate_format(response, format_type):
    format_id = detect_text_format(response)
    id2type = {0: "NO_FORMAT", 1: "JSON", 2: "MARKDOWN", 3: "XML", 4: "HTML"}
    return id2type[format_id] == format_type


def evaluate_word_length(
    response, word_num_bottom_top, around_word_num, word_length_bar, word_length_template_type
):
    lang_full = robust_detect_lang(response)
    lang_short = normalize_lang_code(lang_full)
    word_num = count_words(response, lang_short)
    if word_length_template_type == 0:
        low = int(around_word_num * 0.8)
        high = int(around_word_num * 1.2)
        return low <= word_num <= high
    if word_length_template_type == 1:
        return word_num <= word_length_bar
    return word_num_bottom_top[0] <= word_num <= word_num_bottom_top[1]


def evaluate_sentence_length(
    response, sentence_length_template_type, sentence_length_target, sentence_length_bottom_top
):
    lang_full = robust_detect_lang(response)
    lang_short = normalize_lang_code(lang_full)
    sentence_num = count_sentences(response, lang_short)
    if sentence_length_template_type == 0:
        return sentence_num == sentence_length_target
    if sentence_length_template_type == 1:
        return max(sentence_length_target - 2, 0) <= sentence_num <= sentence_length_target + 2
    if sentence_length_template_type == 2:
        return sentence_num <= sentence_length_target + 2
    return sentence_length_bottom_top[0] <= sentence_num <= sentence_length_bottom_top[1]

