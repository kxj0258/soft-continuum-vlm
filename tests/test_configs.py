from pathlib import Path

import pytest

from soft_continuum_vlm.utils.config import load_yaml_config
from soft_continuum_vlm.utils.paths import project_root


def test_all_checked_in_yaml_configs_are_mappings() -> None:
    root = project_root()
    config_paths = sorted((root / "configs").rglob("*.yaml"))
    assert config_paths
    for path in config_paths:
        loaded = load_yaml_config(path)
        assert isinstance(loaded, dict), path
        assert loaded, path


def test_load_yaml_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_yaml_config(tmp_path / "missing.yaml")
