# outputs/

`outputs/` stores local generated artifacts. Most files under this directory are intentionally ignored by Git because they are run outputs, diagnostics, checkpoints, metrics, or figures.

Tracked file:

- `outputs/README.md`

Common generated subdirectories:

- `outputs/scenes/`: generated MuJoCo scene XML files, such as `feagine_tabletop_scene_reachable.xml`.
- `outputs/diagnostics/`: JSON diagnostics from scene validation, reachability scans, contact scans, and reachable red-pick runs.
- `outputs/scene_inspection/`: Feagine/MuJoCo body, geom, site, and observation inspection reports.
- `outputs/rollouts/`: rollout summaries and optional trajectories.
- `outputs/metrics/`: adapter training and policy evaluation JSON/CSV metrics.
- `outputs/figures/`: figures exported from saved metrics.
- `outputs/checkpoints/`: local adapter checkpoints.
- `outputs/calibration/`: local Jacobian or controller calibration reports.

Important current local scene:

```powershell
python scripts/generate_feagine_tabletop_scene.py --variant reachable --output outputs/scenes/feagine_tabletop_scene_reachable.xml
```

Important current diagnostic:

```powershell
python scripts/run_reachable_red_pick_retract_lift.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --output outputs/diagnostics/reachable_red_pick_retract_lift.json
```

Important current viewer demo:

```powershell
python scripts/view_reachable_red_pick_demo.py --scene outputs/scenes/feagine_tabletop_scene_reachable.xml --trial gentle_retract_scale_down --realtime-factor 0.5
```

Notes:

- Generated outputs should be reproducible from scripts and config.
- Paper figures must be generated from saved JSON/CSV metrics, not hand-filled values.
- Current reachable red-pick diagnostics do not claim stable grasp or object lifting.
