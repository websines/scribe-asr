"""Fast regex/dict code-term replacement on ASR transcripts."""

import re
from .config import CODE_TERMS


def _build_pattern() -> re.Pattern:
    sorted_keys = sorted(CODE_TERMS.keys(), key=len, reverse=True)
    escaped = [re.escape(k) for k in sorted_keys]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


_PATTERN = _build_pattern()


def fix_code_terms(text: str) -> str:
    """Replace common ASR mishearings with correct code terms. ~0.1ms."""
    if not text:
        return text
    return _PATTERN.sub(lambda m: CODE_TERMS.get(m.group(0).lower(), m.group(0)), text)
