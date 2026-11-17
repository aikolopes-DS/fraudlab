"""Complete-number GPT-2 keys and integer-prefix views, without labels.

Inspired by megayak's numeric-token views and goodpjw2008's Part 12 notebook
(version 16). Independently implemented; no notebook code or predictions are
copied. Canonical float64 text intentionally differs from original CSV lexemes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bpe import ENCODING, ENCODING_FILES, TOKENIZER_VERSION, _encoding
from .pooled import COLUMNS, canonical_decimal

SUFFIXES = (
    "__fulltoken_first_key",
    "__fulltoken_second_key",
    "__fulltoken_last_key",
    "__fulltoken_length_key",
    "__integer_tens_key",
    "__integer_length_first_key",
    "__integer_length_two_key",
)


def token_keys(frame: pd.DataFrame, excluded: tuple[str, ...] = ()) -> pd.DataFrame:
    """Return seven categorical keys per allowed continuous column.

    Inputs must also fit the exact-cent range inherited by value_digits. The
    complete decimal text retains fractional cents; there is no cent rounding.
    A missing second token is -1. Prefixes come from floor(x), with zero length1.
    """
    result = pd.DataFrame(index=frame.index)
    encoding = _encoding()
    for name in COLUMNS:
        if name in excluded:
            continue
        if name not in frame:
            raise ValueError(f"Missing full-token input column: {name}")
        series = frame[name]
        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_complex_dtype(series)
        ):
            raise ValueError(f"{name} requires numeric full-token inputs.")
        values = series.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} requires finite nonnegative values.")
        if (values > (2**53 - 1) / 100).any():
            raise ValueError(
                f"{name} exceeds the exact-cent range inherited from value_digits."
            )
        unique, inverse = np.unique(values, return_inverse=True)
        keys = []
        for value in unique:
            text = canonical_decimal(value)
            pieces = encoding.encode_ordinary(" " + text)
            integer = text.split(".", 1)[0]
            keys.append(
                (
                    str(pieces[0]),
                    str(pieces[1] if len(pieces) > 1 else -1),
                    str(pieces[-1]),
                    str(len(pieces)),
                    str(int(integer) // 10),
                    f"{len(integer)}:{integer[:1]}",
                    f"{len(integer)}:{integer[:2]}",
                )
            )
        columns = np.asarray(keys, dtype=object).reshape(-1, len(SUFFIXES))[inverse]
        for index, suffix in enumerate(SUFFIXES):
            result[name + suffix] = columns[:, index]
    return result


def manifest() -> dict:
    encoding = _encoding()
    return {
        "algorithm": "complete_numeric_token_and_integer_prefix_keys",
        "tiktoken": TOKENIZER_VERSION,
        "encoding": ENCODING,
        "vocabulary_size": encoding.max_token_value + 1,
        "encoding_files": list(ENCODING_FILES),
        "input_columns": list(COLUMNS),
        "output_suffixes": list(SUFFIXES),
        "output_count": len(COLUMNS) * len(SUFFIXES),
        "base_feature_set": "value_digits",
        "canonical_text": "float64 -> Decimal(str(float(value))) -> fixed decimal notation, at least .0; one leading ASCII space",
        "zero": "-0.0 normalized to 0.0",
        "fractional_cents": "retained; no cent rounding or decimal padding",
        "csv_lexemes": "not preserved; trailing zeros and exponent spelling may differ",
        "second_token_missing": "-1",
        "integer_keys": "floor(x/10), integer-digit-length:first-digit, integer-digit-length:first-two-digits; floor(0) has length1",
        "numeric_range": "finite, nonnegative, at most (2**53-1)/100 to support inherited exact-cent views",
        "category_semantics": "all seven keys are categorical labels, not ordered token magnitudes",
        "label_usage": "none; only the six declared input columns are read",
        "inspiration": [
            "https://www.kaggle.com/code/megayak/s6e11-per-value-digit-te-shallow-tree-stack",
            "https://www.kaggle.com/code/goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298",
        ],
    }
