from __future__ import annotations

import importlib.util
import pathlib


def pytest_ignore_collect(collection_path, config):
    path = pathlib.Path(str(collection_path))
    if path.name == 'test_stacking_oof_guardrails.py' and importlib.util.find_spec('imblearn') is None:
        return True
    return False
