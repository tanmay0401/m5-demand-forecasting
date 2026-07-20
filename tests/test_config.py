"""Tests for the OmegaConf-based config composer (Hydra-free entry points)."""

from __future__ import annotations

import pytest
from omegaconf import errors

from m5forecast.utils.config import load_config


def test_defaults_compose():
    cfg = load_config()
    assert cfg.model.name == "lightgbm"
    assert cfg.data.horizon == 28
    assert cfg.seed == 42


def test_group_swap():
    cfg = load_config(["model=tft"])
    assert cfg.model.name == "tft"
    assert cfg.data.n_series == 30490  # other groups untouched


def test_dotlist_override():
    cfg = load_config(["training.horizon=14", "seed=7"])
    assert cfg.training.horizon == 14
    assert cfg.seed == 7


def test_unknown_key_raises():
    with pytest.raises(errors.ConfigAttributeError):
        _ = load_config().model.no_such_key
