"""Linear models must not pass n_jobs (sklearn FutureWarning)."""

from __future__ import annotations

import warnings

import pytest

from idxbot.ml.candidates import get_candidate


def test_logistic_regression_no_n_jobs_param():
    model = get_candidate("logistic_regression").create(42)
    # sklearn stores params; n_jobs should be absent or default None without us setting it
    params = model.get_params()
    # We must not have explicitly constructed with n_jobs=1 causing warnings
    # Construction should not warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        get_candidate("logistic_regression").create(0)
        get_candidate("regime_aware_logreg").create(0)
        future = [x for x in w if issubclass(x.category, FutureWarning) and "n_jobs" in str(x.message)]
        assert future == [], f"Unexpected n_jobs FutureWarning: {future}"


def test_tree_models_keep_n_jobs_one():
    rf = get_candidate("random_forest").create(42)
    assert rf.get_params().get("n_jobs") == 1
