import pytest

pytest.importorskip("imblearn")

import numpy as np
import pandas as pd

from src.modeling import WCGSurvivalEncoder
from src.stacking import generate_meta_oof_predictions


class RecordingLogisticRegression:
    fit_calls = []
    predict_calls = []

    def __init__(self, *args, **kwargs):
        self.offset = 0.0

    def fit(self, X, y):
        X = X.copy()
        y = pd.Series(y).copy()
        self.__class__.fit_calls.append((X.index.tolist(), y.index.tolist()))
        self.offset = float(X.iloc[:, 0].mean()) if len(X) else 0.0
        return self

    def predict_proba(self, X):
        X = X.copy()
        self.__class__.predict_calls.append(X.index.tolist())
        probs = np.clip(X.iloc[:, 0].to_numpy(dtype=float) / 10.0 + self.offset / 100.0, 0.0, 1.0)
        return np.column_stack([1.0 - probs, probs])


@pytest.fixture(autouse=True)
def _reset_recording_lr():
    RecordingLogisticRegression.fit_calls = []
    RecordingLogisticRegression.predict_calls = []
    yield


def test_generate_meta_oof_predictions_is_fold_local(monkeypatch):
    monkeypatch.setattr('src.stacking.LogisticRegression', RecordingLogisticRegression)

    blend_oof = pd.DataFrame(
        {'base_a': np.linspace(0.1, 0.9, 10), 'base_b': np.linspace(0.9, 0.1, 10)},
        index=pd.Index(range(100, 110), name='row_id'),
    )
    y = pd.Series([0, 1] * 5, index=blend_oof.index)

    meta_oof, fold_scores, final_model = generate_meta_oof_predictions(blend_oof, y, seed=42)

    assert meta_oof.shape == (len(blend_oof),)
    assert not np.isnan(meta_oof).any()
    assert len(fold_scores) == 5
    assert isinstance(final_model, RecordingLogisticRegression)

    fit_lengths = [len(fit_idx) for fit_idx, _ in RecordingLogisticRegression.fit_calls]
    predict_lengths = [len(valid_idx) for valid_idx in RecordingLogisticRegression.predict_calls]
    assert fit_lengths[:-1] == [8] * 5
    assert predict_lengths[:-1] == [2] * 5
    assert fit_lengths[-1] == 10

    covered = sorted(idx for fold in RecordingLogisticRegression.predict_calls[:-1] for idx in fold)
    assert covered == sorted(blend_oof.index.tolist())


def test_wcg_encoder_rejects_non_unique_index():
    X = pd.DataFrame(
        {
            'Last_Name': ['Smith', 'Smith', 'Jones'],
            'AdjFare': [10.0, 10.0, 30.0],
            'Ticket': ['A/5', 'A/5', 'PC 1'],
            'Fare': [10.0, 10.0, 30.0],
            'Pclass': [3, 3, 1],
            'Sex': ['male', 'female', 'female'],
            'Age': [22.0, 24.0, 35.0],
        },
        index=[0, 0, 1],
    )
    y = pd.Series([0, 1, 1], index=X.index)

    encoder = WCGSurvivalEncoder()
    with pytest.raises(ValueError, match='unique index'):
        encoder.fit(X, y)
