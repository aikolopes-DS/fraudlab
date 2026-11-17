"""Deterministic GPT-2 numeric keys, with no labels or learned statistics.

Hypothesis: azizdzhon's numeric BPE notebook and Shallow-Tree Stack v31.
Implementation: our own feature extraction using OpenAI's tokenizer API.
No notebook code, merge tables, predictions or fitted statistics are reused.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import version

import numpy as np
import pandas as pd

COLUMNS = ("amount_usd", "account_balance_usd")
SUFFIXES = ("__gpt2_first_key", "__gpt2_second_key", "__gpt2_second_cents_key")
TOKENIZER_VERSION = "0.14.0"
ENCODING = "gpt2"
ENCODING_FILES = (
    {
        "url": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
        "sha256": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    },
    {
        "url": "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
        "sha256": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    },
)


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    if version("tiktoken") != TOKENIZER_VERSION:
        raise RuntimeError(f"This hypothesis requires tiktoken=={TOKENIZER_VERSION}.")
    # The pinned implementation verifies both official asset hashes above.
    return tiktoken.get_encoding(ENCODING)


def _cents(series: pd.Series) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        raise ValueError(f"{series.name} must contain numeric USD amounts.")
    values = series.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{series.name} requires finite, nonnegative USD amounts.")
    if (values > (2**53 - 1) / 100).any():
        raise ValueError(f"{series.name} overflows the supported USD cent range.")
    return np.rint(values * 100).astype(np.int64)


def bpe_keys(
    frame: pd.DataFrame,
    excluded: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Return six categorical keys; excluded columns produce no keys.

    USD is quantized with np.rint(float64_values * 100), matching value_digits.
    The integer text is one ASCII space followed by the unsigned base-ten integer.
    Token IDs are categorical labels; an absent second token is -1.
    Integers with more than two tokens intentionally retain only the first two.
    """
    result = pd.DataFrame(index=frame.index)
    for name in COLUMNS:
        if name in excluded:
            continue
        if name not in frame:
            raise ValueError(f"Missing BPE input column: {name}")
        cents = _cents(frame[name])
        integers, inverse = np.unique(cents // 100, return_inverse=True)
        tokens = [_encoding().encode_ordinary(f" {int(value)}") for value in integers]
        first = np.asarray([row[0] for row in tokens], dtype=np.int32)[inverse]
        second = np.asarray(
            [row[1] if len(row) > 1 else -1 for row in tokens], dtype=np.int32
        )[inverse]
        result[name + SUFFIXES[0]] = first.astype(str)
        result[name + SUFFIXES[1]] = second.astype(str)
        result[name + SUFFIXES[2]] = np.char.add(
            np.char.add(second.astype(str), ":"), (cents % 100).astype(str)
        )
    return result


def manifest() -> dict:
    """Pin the tokenizer and official vocabulary assets in experiment evidence."""
    encoding = _encoding()
    return {
        "tiktoken": TOKENIZER_VERSION,
        "encoding": encoding.name,
        "vocabulary_size": encoding.max_token_value + 1,
        "encoding_files": list(ENCODING_FILES),
        "input_columns": list(COLUMNS),
        "output_suffixes": list(SUFFIXES),
        "representation": "cents=np.rint(float64_USD*100); ' ' + str(cents // 100)",
        "rounding": "NumPy rint; nearest representable cent, halfway ties to even",
        "second_token_missing": "-1",
        "third_key": "second_token_id + ':' + str(cents % 100)",
        "more_than_two_tokens": "retain first two; discard later integer tokens",
        "label_usage": "none in extraction; fold-local cross-fitted TE downstream",
    }
