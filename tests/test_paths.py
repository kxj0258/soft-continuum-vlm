from pathlib import Path

import pytest

from soft_continuum_vlm.utils.paths import feagine_root, project_root, workspace_root


def test_project_root_finds_pyproject() -> None:
    root = project_root()
    assert root.name == "soft-continuum-vlm"
    assert (root / "pyproject.toml").exists()


def test_workspace_root_is_project_parent() -> None:
    assert workspace_root() == project_root().parent


def test_feagine_root_uses_environment_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "external-feagine"
    fake.mkdir()
    monkeypatch.setenv("FEAGINE_SIM_ROOT", str(fake))
    assert feagine_root() == fake.resolve()


def test_feagine_root_reports_checked_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "soft-continuum-vlm"
    project.mkdir()
    monkeypatch.delenv("FEAGINE_SIM_ROOT", raising=False)
    with pytest.raises(FileNotFoundError) as exc_info:
        feagine_root(project=project)
    message = str(exc_info.value)
    assert "FEAGINE_SIM_ROOT" in message
    assert "feagine-simulation" in message
    assert "feagine_simulation" in message
