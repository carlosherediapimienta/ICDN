"""Stable mapping between raw identifiers and the contiguous codes used by embeddings."""

import pandas as pd


class LabelEncoder:
    """Maps arbitrary labels to contiguous integer codes.

    The mapping is fitted once on the training panel and stored inside the
    checkpoint, so predictions on new data reuse exactly the same codes.

    Args:
        reserve_zero: when True code 0 is reserved for unseen labels, which is
            what embeddings with ``padding_idx=0`` expect.
    """

    def __init__(self, reserve_zero: bool = False):
        self.reserve_zero = reserve_zero
        self.classes_: list = []
        self._mapping: dict = {}

    def fit(self, values) -> "LabelEncoder":
        labels = pd.unique(pd.Series(values).dropna())
        self.classes_ = sorted(labels.tolist(), key=repr)
        offset = 1 if self.reserve_zero else 0
        self._mapping = {label: i + offset for i, label in enumerate(self.classes_)}
        return self

    def transform(self, values, strict: bool = True) -> pd.Series:
        series = pd.Series(values)
        codes = series.map(self._mapping)
        unknown = series[codes.isna()].unique().tolist()
        if unknown:
            if strict and not self.reserve_zero:
                raise ValueError(
                    f"unseen labels {unknown[:5]} cannot be encoded. "
                    f"Fit the model on data that covers them, or drop those rows."
                )
            codes = codes.fillna(0)
        return codes.astype(int)

    @property
    def size(self) -> int:
        """Number of embedding rows required, including the reserved code."""
        return len(self.classes_) + (1 if self.reserve_zero else 0)

    def to_dict(self) -> dict:
        return {"reserve_zero": self.reserve_zero, "classes": self.classes_}

    @classmethod
    def from_dict(cls, payload: dict) -> "LabelEncoder":
        encoder = cls(reserve_zero=payload["reserve_zero"])
        encoder.classes_ = list(payload["classes"])
        offset = 1 if encoder.reserve_zero else 0
        encoder._mapping = {label: i + offset for i, label in enumerate(encoder.classes_)}
        return encoder
