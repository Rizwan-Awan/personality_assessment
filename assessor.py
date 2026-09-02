"""Personality scoring + lightweight ML assessment layer."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from questions import TRAITS, QUESTIONS
from contradiction import RESPONSE_SCORE


def deterministic_scores(questions, answers):
    totals = {trait: 0.0 for trait in TRAITS}
    counts = {trait: 0 for trait in TRAITS}
    for q in questions:
        if q["id"] not in answers:
            continue
        # Use .get() with a default of 3 so missing keys never throw a KeyError
        value = RESPONSE_SCORE.get(answers[q["id"]], 3)
        score = 6 - value if q.get("reverse") else value
        totals[q["trait"]] += score
        counts[q["trait"]] += 1
    return {
        trait: round(((totals[trait] / counts[trait] - 1) / 4) * 100, 1) if counts[trait] else 0.0
        for trait in TRAITS
    }


def _synthetic_training_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    n_q = len(QUESTIONS)
    X = rng.integers(1, 6, size=(n, n_q)).astype(float)
    y = np.zeros((n, len(TRAITS)), dtype=float)

    trait_names = list(TRAITS)
    for j, trait in enumerate(trait_names):
        indices = [i for i, q in enumerate(QUESTIONS) if q["trait"] == trait]
        transformed = np.empty((n, len(indices)), dtype=float)
        for k, idx in enumerate(indices):
            raw = X[:, idx]
            transformed[:, k] = 6 - raw if QUESTIONS[idx].get("reverse") else raw
        latent = transformed.mean(axis=1)
        y[:, j] = ((latent - 1) / 4) * 100 + rng.normal(0, 3.0, size=n)
    return X, y


_MODEL = None


def ml_scores(questions, answers):
    global _MODEL
    if _MODEL is None:
        X, y = _synthetic_training_data()
        _MODEL = RandomForestRegressor(
            n_estimators=160,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=5,
        )
        _MODEL.fit(X, y)

    vector = [RESPONSE_SCORE.get(answers.get(q["id"]), 3) for q in QUESTIONS]
    pred = _MODEL.predict(np.array(vector, dtype=float).reshape(1, -1))[0]
    return {trait: round(float(np.clip(value, 0, 100)), 1) for trait, value in zip(TRAITS, pred)}


def combine_scores(rule_scores, ml_model_scores, alpha=0.65):
    return {
        k: round(alpha * rule_scores[k] + (1 - alpha) * ml_model_scores[k], 1)
        for k in rule_scores
    }


def overall_summary(scores):
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    strengths = ordered[:3]
    growth = ordered[-3:]
    return strengths, growth
