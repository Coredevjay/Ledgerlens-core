import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

import run_pipeline
from detection.feature_engineering import FEATURE_NAMES
from detection.storage import get_latest_scores
from ingestion.synthetic_data import generate_synthetic_dataset


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    X = [[0] * len(FEATURE_NAMES), [1] * len(FEATURE_NAMES)]
    y = [0, 1]
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    model_path = tmp_path / "models"
    model_path.mkdir()
    for name in ("random_forest", "xgboost", "lightgbm"):
        joblib.dump(model, model_path / f"{name}.joblib")

    import config.settings as settings_module

    object.__setattr__(settings_module.settings, "model_dir", str(model_path))
    object.__setattr__(settings_module.settings, "db_path", str(tmp_path / "ledgerlens.db"))
    return str(model_path)


def test_run_persists_scores(model_dir, monkeypatch):
    trades, account_metadata, events, _labels = generate_synthetic_dataset(
        n_normal_accounts=3, n_wash_rings=1, ring_size=2, trades_per_normal=4, trades_per_wash=4
    )

    monkeypatch.setattr(run_pipeline, "load_historical_trades", lambda **kwargs: trades)
    monkeypatch.setattr(run_pipeline, "load_account_metadata", lambda accounts: account_metadata)
    monkeypatch.setattr(run_pipeline, "load_order_book_events", lambda account: [])

    scores = run_pipeline.run(asset_pairs=[(None, "USDC:ISSUER")])

    accounts = pd.unique(trades[["base_account", "counter_account"]].values.ravel())
    assert len(scores) == len(accounts)
    for s in scores:
        assert 0 <= s.score <= 100

    import config.settings as settings_module

    stored = get_latest_scores(db_path=settings_module.settings.db_path)
    assert len(stored) == len(scores)
