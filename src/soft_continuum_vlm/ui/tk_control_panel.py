from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Any, Sequence

from soft_continuum_vlm.ui.manual_control import ManualFeagineController


SECTION_PRESETS_DEG: dict[str, list[float]] = {
    "Straight": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Bend +Y": [20.0, 90.0, 15.0, 90.0, 10.0, 90.0],
    "Bend -Y": [20.0, -90.0, 15.0, -90.0, 10.0, -90.0],
    "X Gradient": [20.0, 0.0, 10.0, 0.0, 10.0, 180.0],
}

SECTION_MAGNITUDE_LIMIT_DEG = 135.0
SECTION_DIRECTION_LIMIT_DEG = 180.0
GRASPER_ROTATION_LIMIT_DEG = 180.0


class TkFeagineControlPanel:
    def __init__(
        self,
        controller: ManualFeagineController | None = None,
        *,
        title: str = "Feagine Manual Control",
        live_update: bool = True,
    ) -> None:
        self.controller = controller or ManualFeagineController()
        self.live_update = bool(live_update)
        self._root = tk.Tk()
        self._root.title(title)
        self._root.protocol("WM_DELETE_WINDOW", self._request_quit)
        self._closed = False
        self._syncing = False
        self._tip_position: Sequence[float] | None = None

        self._section_vars: list[tk.DoubleVar] = []
        self._section_entry_vars: list[tk.StringVar] = []
        self._grip_var = tk.DoubleVar(value=float(self.controller.state.grip_command))
        self._rotation_var = tk.DoubleVar(value=float(self.controller.state.grasper_rotation))
        self._rotation_entry_var = tk.StringVar(value="0.000")
        self._status_var = tk.StringVar(value="")

        self._build()
        self._refresh_from_controller()
        self._update_status()

    def update(self) -> None:
        if self._closed:
            return
        try:
            self._root.update_idletasks()
            self._root.update()
        except tk.TclError:
            self._closed = True
            self.controller.state.quit_requested = True

    def action(self) -> dict[str, Any]:
        if not self.live_update:
            self._sync_controller_from_vars()
        action = self.controller.action()
        return {
            "section_angles": list(action["section_angles"]),
            "grip_command": float(action["grip_command"]),
            "grasper_rotation": float(action["grasper_rotation"]),
        }

    def set_tip_position(self, tip_position: Sequence[float] | None) -> None:
        self._tip_position = tip_position
        self._update_status()

    def should_quit(self) -> bool:
        return bool(self._closed or self.controller.state.quit_requested)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._root.destroy()
        except tk.TclError:
            pass

    def set_section_angle(self, index: int, value: float) -> None:
        self._set_section_angle(index, float(value))

    def set_section_angle_from_text(self, index: int, text: str) -> None:
        try:
            value = float(text)
        except ValueError:
            current_rad = float(self.controller.state.section_angles[index])
            self._section_entry_vars[index].set(f"{self._radians_to_ui(index, current_rad):.1f}")
            return
        self._set_section_angle(index, value)

    def set_grip_command(self, value: float) -> None:
        grip = self._clip(float(value), float(self.controller.config.min_grip_command), float(self.controller.config.max_grip_command))
        self._grip_var.set(grip)
        self.controller.state.grip_command = grip
        self._update_status()

    def set_grasper_rotation(self, value: float) -> None:
        rotation_deg = self._clip(
            float(value),
            -GRASPER_ROTATION_LIMIT_DEG,
            GRASPER_ROTATION_LIMIT_DEG,
        )
        rotation_rad = math.radians(rotation_deg)
        self._rotation_var.set(rotation_deg)
        self._rotation_entry_var.set(f"{rotation_deg:.1f}")
        self.controller.state.grasper_rotation = rotation_rad
        self._update_status()

    def set_grasper_rotation_from_text(self, text: str) -> None:
        try:
            value = float(text)
        except ValueError:
            rotation_deg = math.degrees(float(self.controller.state.grasper_rotation))
            self._rotation_entry_var.set(f"{rotation_deg:.1f}")
            return
        self.set_grasper_rotation(value)

    def apply_preset(self, name: str) -> None:
        if name not in SECTION_PRESETS_DEG:
            raise ValueError(f"Unknown Feagine section preset: {name}")
        for index, value in enumerate(SECTION_PRESETS_DEG[name]):
            self._set_section_angle(index, value)

    def _build(self) -> None:
        main = ttk.Frame(self._root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        self._root.columnconfigure(0, weight=1)
        self._root.rowconfigure(0, weight=1)

        sections = ttk.LabelFrame(main, text="Section Angles", padding=6)
        sections.grid(row=0, column=0, sticky="ew")
        sections.columnconfigure(1, weight=1)
        for index in range(6):
            self._add_section_row(sections, index)

        grip = ttk.LabelFrame(main, text="Gripper", padding=6)
        grip.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        grip.columnconfigure(0, weight=1)
        tk.Scale(
            grip,
            from_=float(self.controller.config.min_grip_command),
            to=float(self.controller.config.max_grip_command),
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self._grip_var,
            command=self._on_grip_slider,
        ).grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Button(grip, text="Open", command=lambda: self.set_grip_command(0.0)).grid(row=1, column=0, sticky="ew")
        ttk.Button(grip, text="Close", command=lambda: self.set_grip_command(1.0)).grid(row=1, column=1, sticky="ew")

        rotation = ttk.LabelFrame(main, text="Grasper Rotation", padding=6)
        rotation.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        rotation.columnconfigure(1, weight=1)
        ttk.Label(rotation, text="rotation").grid(row=0, column=0, sticky="w")
        tk.Scale(
            rotation,
            from_=-GRASPER_ROTATION_LIMIT_DEG,
            to=GRASPER_ROTATION_LIMIT_DEG,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self._rotation_var,
            command=self._on_rotation_slider,
        ).grid(row=0, column=1, sticky="ew")
        entry = ttk.Entry(rotation, textvariable=self._rotation_entry_var, width=8)
        entry.grid(row=0, column=2, padx=(6, 0))
        entry.bind("<Return>", lambda _event: self.set_grasper_rotation_from_text(self._rotation_entry_var.get()))

        buttons = ttk.Frame(main)
        buttons.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        for column, (label, command) in enumerate(
            [
                ("Reset Command", self._reset_command),
                ("Straight", lambda: self.apply_preset("Straight")),
                ("Bend +Y", lambda: self.apply_preset("Bend +Y")),
                ("Bend -Y", lambda: self.apply_preset("Bend -Y")),
                ("X Gradient", lambda: self.apply_preset("X Gradient")),
                ("Pause / Resume", self._toggle_pause),
                ("Quit", self._request_quit),
            ]
        ):
            ttk.Button(buttons, text=label, command=command).grid(row=column // 4, column=column % 4, sticky="ew", padx=2, pady=2)
            buttons.columnconfigure(column % 4, weight=1)

        ttk.Label(main, textvariable=self._status_var, justify=tk.LEFT).grid(row=4, column=0, sticky="w", pady=(8, 0))

    def _add_section_row(self, parent: ttk.Frame, index: int) -> None:
        value_rad = float(self.controller.state.section_angles[index])
        value_deg = self._radians_to_ui(index, value_rad)
        slider_var = tk.DoubleVar(value=value_deg)
        entry_var = tk.StringVar(value=f"{value_deg:.1f}")
        self._section_vars.append(slider_var)
        self._section_entry_vars.append(entry_var)
        lower, upper = self._section_bounds(index)

        ttk.Label(parent, text=self._section_label(index)).grid(row=index, column=0, sticky="w")
        tk.Scale(
            parent,
            from_=lower,
            to=upper,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=slider_var,
            command=lambda raw, axis=index: self._on_section_slider(axis, raw),
        ).grid(row=index, column=1, sticky="ew")
        entry = ttk.Entry(parent, textvariable=entry_var, width=8)
        entry.grid(row=index, column=2, padx=(6, 0))
        entry.bind("<Return>", lambda _event, axis=index: self.set_section_angle_from_text(axis, self._section_entry_vars[axis].get()))

    def _on_section_slider(self, index: int, raw_value: str) -> None:
        if self._syncing:
            return
        self._set_section_angle(index, float(raw_value))

    def _on_grip_slider(self, raw_value: str) -> None:
        if self._syncing:
            return
        self.set_grip_command(float(raw_value))

    def _on_rotation_slider(self, raw_value: str) -> None:
        if self._syncing:
            return
        self.set_grasper_rotation(float(raw_value))

    def _set_section_angle(self, index: int, value: float) -> None:
        lower, upper = self._section_bounds(index)
        clipped_deg = self._clip(float(value), lower, upper)
        clipped_rad = self._ui_to_radians(index, clipped_deg)
        self._syncing = True
        try:
            self._section_vars[index].set(clipped_deg)
            self._section_entry_vars[index].set(f"{clipped_deg:.1f}")
        finally:
            self._syncing = False
        self.controller.state.section_angles[index] = clipped_rad
        self._update_status()

    def _sync_controller_from_vars(self) -> None:
        for index, var in enumerate(self._section_vars):
            lower, upper = self._section_bounds(index)
            value_deg = self._clip(
                float(var.get()),
                lower,
                upper,
            )
            self.controller.state.section_angles[index] = self._ui_to_radians(index, value_deg)
        self.controller.state.grip_command = self._clip(
            float(self._grip_var.get()),
            float(self.controller.config.min_grip_command),
            float(self.controller.config.max_grip_command),
        )
        rotation_deg = self._clip(
            float(self._rotation_var.get()),
            -GRASPER_ROTATION_LIMIT_DEG,
            GRASPER_ROTATION_LIMIT_DEG,
        )
        self.controller.state.grasper_rotation = math.radians(rotation_deg)
        self._update_status()

    def _refresh_from_controller(self) -> None:
        self._syncing = True
        try:
            for index, value in enumerate(self.controller.state.section_angles[:6]):
                value_deg = self._radians_to_ui(index, float(value))
                self._section_vars[index].set(value_deg)
                self._section_entry_vars[index].set(f"{value_deg:.1f}")
            self._grip_var.set(float(self.controller.state.grip_command))
            rotation_deg = math.degrees(float(self.controller.state.grasper_rotation))
            self._rotation_var.set(rotation_deg)
            self._rotation_entry_var.set(f"{rotation_deg:.1f}")
        finally:
            self._syncing = False

    def _reset_command(self) -> None:
        self.controller.reset()
        self._refresh_from_controller()
        self._update_status()

    def _toggle_pause(self) -> None:
        self.controller.state.paused = not self.controller.state.paused
        self._update_status()

    def _request_quit(self) -> None:
        self.controller.state.quit_requested = True
        self.close()

    def _update_status(self) -> None:
        angles_rad = [float(value) for value in self.controller.state.section_angles[:6]]
        angles_deg = [self._radians_to_ui(index, value) for index, value in enumerate(angles_rad)]
        angles_deg_text = ", ".join(f"{value:.1f}" for value in angles_deg)
        angles_rad_text = ", ".join(f"{value:.3f}" for value in angles_rad)
        grasper_rotation_rad = float(self.controller.state.grasper_rotation)
        grasper_rotation_deg = math.degrees(grasper_rotation_rad)
        tip_text = "None"
        if self._tip_position is not None:
            tip_text = "[" + ", ".join(f"{float(value):.3f}" for value in self._tip_position[:3]) + "]"
        self._status_var.set(
            "section_angles_deg=["
            + angles_deg_text
            + "]\n"
            + "section_angles_rad=["
            + angles_rad_text
            + "]\n"
            + f"grip_command={self.controller.state.grip_command:.3f}\n"
            + f"grasper_rotation_deg={grasper_rotation_deg:.1f}\n"
            + f"grasper_rotation_rad={grasper_rotation_rad:.3f}\n"
            + f"tip_position={tip_text}\n"
            + f"paused={self.controller.state.paused}"
        )

    @staticmethod
    def _section_bounds(index: int) -> tuple[float, float]:
        if index % 2 == 0:
            return 0.0, SECTION_MAGNITUDE_LIMIT_DEG
        return -SECTION_DIRECTION_LIMIT_DEG, SECTION_DIRECTION_LIMIT_DEG

    @staticmethod
    def _section_label(index: int) -> str:
        if index % 2 == 0:
            return f"section {index} bend (deg)"
        return f"section {index} direction (deg)"

    @staticmethod
    def _ui_to_radians(index: int, value_deg: float) -> float:
        if index % 2 == 0:
            return math.radians(max(0.0, value_deg))
        return math.radians(value_deg)

    @staticmethod
    def _radians_to_ui(index: int, value_rad: float) -> float:
        value_deg = math.degrees(value_rad)
        if index % 2 == 0:
            return max(0.0, value_deg)
        return value_deg

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return max(float(lower), min(float(upper), float(value)))
