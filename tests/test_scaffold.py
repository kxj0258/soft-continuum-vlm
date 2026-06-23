from pathlib import Path

import soft_continuum_vlm


def test_package_marker_imports() -> None:
    assert soft_continuum_vlm.__name__ == "soft_continuum_vlm"
    assert Path(soft_continuum_vlm.__file__).name == "__init__.py"
