"""Fuse candidate lung masks.

Majority vote and probability averaging are the default. STAPLE is the
classical multi-rater baseline. The method name is stored with every silver mask.
"""

from __future__ import annotations

import numpy as np


def majority_vote(masks: list[np.ndarray], min_votes: int | None = None) -> np.ndarray:
    stack = np.stack([np.asarray(mask) > 0 for mask in masks], axis=0)
    needed = int(np.ceil(stack.shape[0] / 2)) if min_votes is None else int(min_votes)
    if stack.shape[0] >= 3 and min_votes is None:
        needed = 2
    return stack.sum(axis=0) >= needed


def probability_average(probabilities: list[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    stacked = np.stack([np.asarray(prob, dtype=np.float32) for prob in probabilities], axis=0)
    return stacked.mean(axis=0) >= threshold


def staple(masks: list[np.ndarray], iterations: int = 20) -> np.ndarray:
    """Binary STAPLE (Warfield). Returns the MAP label after a short EM run."""
    stack = np.stack([np.asarray(mask) > 0 for mask in masks], axis=0).astype(np.float64)
    raters, height, width = stack.shape
    observations = stack.reshape(raters, -1)
    sensitivity = np.full(raters, 0.9)
    specificity = np.full(raters, 0.9)
    prior = float(np.clip(observations.mean(), 0.05, 0.95))
    posterior = np.full(observations.shape[1], prior)
    for _ in range(iterations):
        log_true = np.full(observations.shape[1], np.log(prior + 1e-8))
        log_false = np.full(observations.shape[1], np.log(1.0 - prior + 1e-8))
        for rater in range(raters):
            positive = observations[rater] > 0.5
            log_true = log_true + np.where(
                positive, np.log(sensitivity[rater] + 1e-8), np.log(1.0 - sensitivity[rater] + 1e-8)
            )
            log_false = log_false + np.where(
                positive, np.log(1.0 - specificity[rater] + 1e-8), np.log(specificity[rater] + 1e-8)
            )
        shift = np.maximum(log_true, log_false)
        true_score = np.exp(log_true - shift)
        false_score = np.exp(log_false - shift)
        posterior = true_score / (true_score + false_score)
        for rater in range(raters):
            positive = observations[rater]
            sensitivity[rater] = float((posterior * positive).sum() / (posterior.sum() + 1e-8))
            specificity[rater] = float(((1.0 - posterior) * (1.0 - positive)).sum() / ((1.0 - posterior).sum() + 1e-8))
        prior = float(np.clip(posterior.mean(), 1e-3, 1.0 - 1e-3))
    return posterior.reshape(height, width) >= 0.5


def fuse(
    masks: list[np.ndarray],
    method: str = "majority",
    probabilities: list[np.ndarray] | None = None,
    threshold: float = 0.5,
) -> tuple[np.ndarray, str]:
    if method == "majority":
        return majority_vote(masks), "majority"
    if method == "probability":
        if not probabilities:
            raise ValueError("Probability averaging needs a probability map from each model.")
        return probability_average(probabilities, threshold=threshold), "probability"
    if method == "staple":
        return staple(masks), "staple"
    raise ValueError(f"Unknown fusion method: {method}")
