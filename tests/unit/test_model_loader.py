from unittest.mock import patch, MagicMock

import pytest

from src.serving.model_loader import (
    load_model,
    _load_local,
    get_model_info,
)


def test_get_model_info_returns_dict():
    result = get_model_info()

    assert isinstance(result, dict)


def test_load_model_falls_back_to_local():
    mock_model = MagicMock()

    with patch(
        "src.serving.model_loader._load_local",
        return_value=(mock_model, {"source": "local"}),
    ):
        model, info = load_model()

        assert model is not None
        assert info["source"] == "local"


def test_load_local_file_not_found():
    with patch(
        "src.serving.model_loader.Path.exists",
        return_value=False,
    ):
        with pytest.raises(FileNotFoundError):
            _load_local()