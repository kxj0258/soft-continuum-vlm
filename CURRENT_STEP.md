# CURRENT_STEP.md

Current focus: converge the real Feagine MuJoCo tabletop environment before any learning, VLM, VLA, OpenVLA, Octo, adapter training, or dataset collection work.

## Active Route

The current route is B + A:

1. DEMO-1: replay the REAL-PICK-4 reachable red-pick retract/lift-like diagnostic in the MuJoCo viewer for manual contact observation.
2. SCENE-GRIPPER-1: attribute contacts to concrete geom/body pairs so we can tell whether red-object motion comes from finger enclosure, arm/body pushing, pedestal blocking, or sweeping contact.

## Implemented Real-Scene Diagnostics

- `scripts/generate_feagine_tabletop_scene.py`
- `scripts/validate_reachable_scene.py`
- `scripts/inspect_tabletop_scene_state.py`
- `scripts/diagnose_tabletop_contacts.py`
- `scripts/scan_feagine_reachability.py`
- `scripts/scan_grasper_red_contact.py`
- `scripts/scan_grasper_red_contact_local.py`
- `scripts/sweep_preclose_gripper_close_contact.py`
- `scripts/tune_reachable_red_pick_parameters.py`
- `scripts/run_reachable_red_pick_attempt.py`
- `scripts/run_reachable_red_pick_hold_retract.py`
- `scripts/run_reachable_red_pick_retract_lift.py`
- `scripts/view_reachable_red_pick_demo.py`
- `scripts/control_feagine_tabletop_panel_ui.py`

## Current REAL-PICK Finding

The reachable red-object scene can produce red-grasper contact and red-object planar motion, but the post-close diagnostic does not support a stable grasp or object lifting claim.

Current key diagnostic result:

- best follow-score trial: `gentle_retract_scale_down`
- max post-close contact ratio: `0.47692307692307695`
- max post-close red motion: `0.04763757862586862`
- max post-close red z delta: `0.00098981615347854`
- min post-close pedestal contact ratio: `1.0`
- max red-grasper normal force: `780.2823229920069`
- judgment: `[FAIL] no useful post-close following observed`

## Do Not Work On Yet

- OpenVLA
- Octo
- real VLM planner
- VLA adapter
- adapter training for new conclusions
- dataset collection
- large model downloads
- paper claims from current reachable diagnostics

## Next Step

Proceed to SCENE-GRIPPER-1: contact geometry attribution for `red_object`, `red_pedestal`, Feagine grasper fingers, and nearby arm/body geoms.
