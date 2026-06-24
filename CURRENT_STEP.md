# CURRENT_STEP.md

Current focus: converge the real Feagine MuJoCo environment before any learning or VLM/VLA work.

Do not work on:
- OpenVLA
- Octo
- real VLM
- adapter training
- dataset generation
- all four tasks at once
- paper figures

Current milestone order:
1. Verify Feagine runtime imports and preset resolution.
2. Inspect MuJoCo model names and Feagine robot methods.
3. Stabilize FeagineMujocoEnv reset/step.
4. Add structured robot_state with qpos/qvel/section_angles/grip/grasper_rotation.
5. Identify reliable tip pose source.
6. Verify section_angles physically affect the model.
7. Calibrate local Jacobian.
8. Replace zero PccIkController.
9. Implement one pick task expert.
