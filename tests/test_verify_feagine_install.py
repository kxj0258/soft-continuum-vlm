from __future__ import annotations

from scripts import verify_feagine_install


class _FakeMujoco:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def robot_asset_path(self, *, preset_id: str, model_type: str) -> str:
        self.calls.append((preset_id, model_type))
        if preset_id == "a03_type_2":
            return "C:/fake/presets/a03_type_2/feagine.xml"
        raise FileNotFoundError(preset_id)


class _FallbackMujoco:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def robot_asset_path(self, *, preset_id: str, model_type: str) -> str:
        self.calls.append((preset_id, model_type))
        if preset_id == "a03":
            return "C:/fake/presets/a03/feagine.xml"
        raise FileNotFoundError(preset_id)


def test_print_asset_path_prefers_project_default_a03_type_2() -> None:
    fake = _FakeMujoco()

    asset_path = verify_feagine_install._print_asset_path(fake)

    assert fake.calls == [("a03_type_2", "mjcf")]
    assert str(asset_path).replace("\\", "/").endswith("/a03_type_2/feagine.xml")


def test_print_asset_path_falls_back_to_legacy_a03() -> None:
    fake = _FallbackMujoco()

    asset_path = verify_feagine_install._print_asset_path(fake)

    assert fake.calls == [("a03_type_2", "mjcf"), ("a03", "mjcf")]
    assert str(asset_path).replace("\\", "/").endswith("/a03/feagine.xml")
