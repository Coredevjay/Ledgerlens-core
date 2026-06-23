"""Continuous runner that drives the red team loop against the live model.

Loads seed wash-trade feature vectors, attacks the current model with the
:class:`~detection.red_team.attacker.GeneticAttacker`, logs successful evasions,
and periodically evaluates the automated hardening trigger.  Designed to run on a
background thread so it never blocks inference.
"""

from __future__ import annotations

import json
import logging
import threading

import numpy as np

from detection.red_team import EVASION_THRESHOLD, N_EVASION_TRIGGER
from detection.red_team.attacker import GeneticAttacker, evaluate_score
from detection.red_team.evasion_logger import log_evasion, maybe_trigger_hardening

logger = logging.getLogger("ledgerlens.red_team.runner")


def load_random_seeds(seed_dataset_path: str, n: int, rng=None) -> list[dict]:
    """Load up to ``n`` random seed feature dicts from a JSON dataset.

    The dataset is a JSON file containing a list of ``{feature: value}`` objects
    (feature vectors of known wash trades).  Fewer than ``n`` rows are returned
    when the dataset is smaller.
    """
    rng = rng if rng is not None else np.random.default_rng()
    with open(seed_dataset_path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise ValueError("seed dataset must be a JSON list of feature objects")
    if not rows:
        return []
    n = min(n, len(rows))
    idx = rng.choice(len(rows), size=n, replace=False)
    return [rows[int(i)] for i in idx]


def run_red_team_loop(
    model,
    seed_dataset_path: str,
    feature_constraints: dict,
    poll_interval_seconds: int = 300,
    n_seeds_per_round: int = 20,
    n_generations: int = 100,
    threshold: float = EVASION_THRESHOLD,
    n_trigger: int = N_EVASION_TRIGGER,
    retrain_callback=None,
    stop_event: threading.Event | None = None,
    max_iterations: int | None = None,
    db_path: str | None = None,
    seed: int | None = None,
) -> int:
    """Continuously attack the current model and log evasions.

    Each round samples ``n_seeds_per_round`` seeds, evolves an attack per seed,
    logs any evasion that beats ``threshold``, then checks the hardening trigger.

    The loop terminates when ``stop_event`` is set or after ``max_iterations``
    rounds (whichever comes first); leave both unset for a truly continuous loop.
    Returns the number of rounds executed.

    Pacing uses ``stop_event.wait(poll_interval_seconds)`` rather than a bare
    sleep, so a background loop can be cancelled promptly without blocking.
    """
    rng = np.random.default_rng(seed)
    feature_names = list(feature_constraints.keys())
    rounds = 0

    while not (stop_event is not None and stop_event.is_set()):
        seeds = load_random_seeds(seed_dataset_path, n_seeds_per_round, rng)
        for seed_features in seeds:
            seed_array = np.array([seed_features.get(f, 0.0) for f in feature_names], dtype=float)
            attacker = GeneticAttacker(
                model, feature_constraints, seed=int(rng.integers(0, 2**31 - 1))
            )
            best, score = attacker.evolve(seed_array, n_generations=n_generations)
            if score < threshold:
                log_evasion(
                    original_features=seed_features,
                    evasion_features=attacker.to_dict(best),
                    original_score=evaluate_score(model, seed_features),
                    evasion_score=score,
                    attacker_generation=getattr(attacker, "last_generation", n_generations),
                    threshold=threshold,
                    db_path=db_path,
                )

        maybe_trigger_hardening(
            n_trigger=n_trigger,
            threshold=threshold,
            retrain_callback=retrain_callback,
            db_path=db_path,
        )

        rounds += 1
        if max_iterations is not None and rounds >= max_iterations:
            break
        if stop_event is not None:
            if stop_event.wait(poll_interval_seconds):
                break
        else:  # pragma: no cover - only hit by a genuinely unbounded loop
            import time

            time.sleep(poll_interval_seconds)

    return rounds


def start_red_team_loop(*args, **kwargs) -> threading.Thread:
    """Start :func:`run_red_team_loop` on a daemon thread and return it.

    Accepts the same arguments as :func:`run_red_team_loop`.  The thread is a
    daemon so it never keeps the process alive, satisfying the requirement that
    the red team loop run in the background without blocking inference.
    """
    thread = threading.Thread(
        target=run_red_team_loop, args=args, kwargs=kwargs, name="red-team-loop", daemon=True
    )
    thread.start()
    return thread
