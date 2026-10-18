"""Patient-level splits.

A study id hashes to exactly one of train, val, or test. Masks from the same
patient never cross those boundaries.
"""

from __future__ import annotations

import hashlib


def assign_split(patient_id: str, seed: int = 2026, train: float = 0.70, val: float = 0.15) -> str:
    digest = hashlib.sha256(f"{seed}:{patient_id}".encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / 0xFFFFFFFF
    if unit < train:
        return "train"
    if unit < train + val:
        return "val"
    return "test"


def assert_no_patient_leakage(patient_ids: list[str], splits: list[str]) -> None:
    seen: dict[str, str] = {}
    for patient_id, split in zip(patient_ids, splits):
        previous = seen.get(patient_id)
        if previous is not None and previous != split:
            raise ValueError(f"patient {patient_id} appears in both {previous} and {split}")
        seen[patient_id] = split
