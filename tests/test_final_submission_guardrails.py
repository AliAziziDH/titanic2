import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.final_submission import run_final_submission

def test_final_submission_rejects_invalid_survivor_count(monkeypatch):
    monkeypatch.setenv("ENABLE_KAGGLE_SUBMISSION", "0")
    
    # Mock data loading
    test_df = pd.DataFrame({
        "PassengerId": range(892, 1310),
        "Pclass": [3] * 418,
        "Name": ["A, B"] * 418,
        "Sex": ["male"] * 418,
        "Age": [22] * 418,
        "SibSp": [0] * 418,
        "Parch": [0] * 418,
        "Ticket": ["123"] * 418,
        "Fare": [7.25] * 418,
        "Cabin": [np.nan] * 418,
        "Embarked": ["S"] * 418,
        "Title": ["Mr"] * 418
    })
    
    # Fake model that predicts all 0s
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.zeros((418, 2))
    
    with patch('src.final_submission._load_individual_models') as mock_indiv:
        mock_indiv.return_value = {"CatBoost": [mock_model]}
        
        with patch('src.final_submission._load_modeling_data') as mock_load:
            mock_load.return_value = (test_df, test_df)
            with patch('src.final_submission.pd.read_csv', return_value=test_df):
                with patch('src.final_submission.WCGPostProcessor') as mock_proc:
                    instance = mock_proc.return_value
                    # return 0 survivors for final blend
                    instance.transform.return_value = np.zeros(418)

                    with patch('src.final_submission._submit_to_kaggle'):
                        # We expect it to raise ValueError for final_blend, NOT for CatBoost
                        with pytest.raises(ValueError, match="CRITICAL ERROR: Submission final_blend has (0|418) survivors"):
                            run_final_submission()
