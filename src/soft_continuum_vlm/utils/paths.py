from __future__ import annotations

import os
from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"Could not find project root from {current}")


def workspace_root(start: Path | None = None) -> Path:
    return project_root(start=start).parent


def feagine_root(project: Path | None = None) -> Path:
    checked: list[Path] = []

    override = os.environ.get("FEAGINE_SIM_ROOT")
    if override:
        override_path = Path(override).expanduser().resolve()
        checked.append(override_path)
        if override_path.exists():
            return override_path

    root = (project or project_root()).resolve()
    for name in ("feagine_simulation", "feagine-simulation"):
        candidate = root.parent / name
        checked.append(candidate)
        if candidate.exists():
            return candidate.resolve()

    checked_text = "\n".join(f"- {path}" for path in checked)
    raise FileNotFoundError(
        "Feagine simulation root was not found. Set FEAGINE_SIM_ROOT or place "
        "feagine_simulation or feagine-simulation next to this project. Checked paths:\n"
        f"{checked_text}"
    )
