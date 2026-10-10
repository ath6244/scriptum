"""RQ4: text transformations for the ablations, and descriptive style features.

Each transformation removes one kind of signal, so retraining on transformed text shows how much
the classifiers depend on it. Train, val and test are always transformed the same way.
"""

import re

import numpy as np

# ---------------------------------------------------------------------------
# Markdown removal
# ---------------------------------------------------------------------------
# (pattern, replacement), applied in this order. MULTILINE makes ^ and $ match at every line.
MARKDOWN_RULES = [
    (re.compile(r"^\s*```.*$", re.MULTILINE), ""),                         # code fence lines (code itself is kept)
    (re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$", re.MULTILINE), ""),         # horizontal rules: ---, ***, ___
    (re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$", re.MULTILINE), ""),  # table separator rows
    (re.compile(r"\|"), " "),                                              # remaining table pipes
    (re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE), ""),                  # headers: "## Title" -> "Title"
    (re.compile(r"^\s*[-*+•]\s+", re.MULTILINE), ""),                      # bullet markers
    (re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE), ""),                     # numbered-list markers: "1. " / "1) "
    (re.compile(r"(\*\*|__)(.+?)\1"), r"\2"),                              # bold: **x** / __x__ -> x
    (re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])"), r"\1"),     # italics: *x* -> x
    (re.compile(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])"), r"\1"),       # italics: _x_ -> x (not snake_case)
    (re.compile(r"`([^`\n]+)`"), r"\1"),                                   # inline code: `x` -> x
]


def strip_markdown(text):
    for pattern, replacement in MARKDOWN_RULES:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Emoji and punctuation normalization
# ---------------------------------------------------------------------------
# Emoji blocks plus the invisible joiners/selectors that build multi-part emoji.
ANY_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF️‍⃣]")
PUNCT_RULES = [
    (re.compile(r"[‐-―−]"), "-"),     # hyphen/en-dash/em-dash/minus variants -> "-"
    (re.compile(r"[‘’‚‛]"), "'"),  # curly single quotes -> '
    (re.compile(r"[“”„‟]"), '"'),  # curly double quotes -> "
    (re.compile(r"…"), "."),                     # ellipsis character -> .
    (re.compile(r"([^\w\s])\1+"), r"\1"),             # repeated punctuation: "..." -> ".", "!!!" -> "!", "--" -> "-"
    (re.compile(r"\s+"), " "),                        # all whitespace, including newlines, -> one space
]


def strip_emoji_and_normalize_punct(text):
    text = ANY_EMOJI.sub("", text)
    for pattern, replacement in PUNCT_RULES:
        text = pattern.sub(replacement, text)
    return text.strip()


# ---------------------------------------------------------------------------
# Opener removal and length matching
# ---------------------------------------------------------------------------
# The first sentence ends at the first . ! ? followed by whitespace, or at the first line break.
FIRST_SENTENCE = re.compile(r"^\s*.*?(?:[.!?](?=\s)|\n|$)", re.DOTALL)


def drop_first_sentence(text):
    return FIRST_SENTENCE.sub("", text, count=1).lstrip()


WORD = re.compile(r"\S+")


def truncate_words(text, n_words):
    """Keep the first n_words words, cutting the original string so line breaks and formatting survive."""
    for i, match in enumerate(WORD.finditer(text)):
        if i == n_words - 1:
            return text[:match.end()]
    return text


# ---------------------------------------------------------------------------
# Descriptive features
# ---------------------------------------------------------------------------
LOWER_WORD = re.compile(r"\w+")
CODE_BLOCK = re.compile(r"```.*?(```|$)", re.DOTALL)
SENTENCE_END = re.compile(r"[.!?]+(?=\s)|\n+")
HEADER = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
BULLET = re.compile(r"^\s*([-*+•]|\d+[.)])\s+", re.MULTILINE)  # bullet or numbered item
BOLD = re.compile(r"(\*\*|__)(.+?)\1")
FENCE = re.compile(r"^\s*```", re.MULTILINE)
# Common emoji blocks: pictographs, emoticons, transport, supplemental symbols, dingbats, misc symbols.
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")
REFUSAL = re.compile(r"\b(I can't|I can’t|I cannot|I'm unable|I’m unable|I am unable|I'm not able|I’m not able|"
                     r"I am not able|I won't be able|I'm sorry, but|I’m sorry, but)", re.IGNORECASE)
TTR_WINDOW = 200  # words
REFUSAL_CHARS = 200  # refusals happen at the start of a reply


def type_token_ratio(text, window=TTR_WINDOW):
    """Distinct words / words in the first `window` words. NaN if the text is shorter than the window,
    because TTR falls as texts get longer and would otherwise just measure length."""
    words = LOWER_WORD.findall(text.lower())
    return len(set(words[:window])) / window if len(words) >= window else np.nan


def mean_sentence_length(text):
    """Mean words per sentence, ignoring code blocks."""
    prose = CODE_BLOCK.sub(" ", text)
    sentences = [s for s in SENTENCE_END.split(prose) if s.strip()]
    lengths = [len(s.split()) for s in sentences]
    return np.mean(lengths) if lengths else np.nan


def style_features(text):
    return {
        "words": len(text.split()),
        "ttr_200": type_token_ratio(text),
        "mean_sentence_len": mean_sentence_length(text),
        "headers": len(HEADER.findall(text)),
        "bullets": len(BULLET.findall(text)),
        "bold": len(BOLD.findall(text)),
        "code_blocks": len(FENCE.findall(text)) // 2,  # a block has an opening and a closing fence
        "emoji": len(EMOJI.findall(text)),
        "refusal": bool(REFUSAL.search(text[:REFUSAL_CHARS])),
    }
