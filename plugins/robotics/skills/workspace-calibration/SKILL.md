---
name: workspace-calibration
description: >-
  Lazy cross-session robot workspace calibration (.robot_calibration.json) and
  tested helpers for surface-height probing, wrist hand-eye checks or local
  Jacobians, and camera staleness. Use when calibrating, verifying, or reusing
  spatial, camera, surface-height, or gripper calibration before robotic
  manipulation. Don't use for non-robot camera calibration or non-robotics
  tasks.
metadata:
  publisher: google
---

# Lazy Workspace Calibration ([`calibration.py`](scripts/calibration.py))

Use bundled [`calibration.py`](scripts/calibration.py) (`numpy opencv-python`)
for lazy, capability-keyed memoization in `.robot_calibration.json` (following
[`physical_robot_safety.md`](../../rules/physical_robot_safety.md)). **Probe
only what the current task actually needs:** in simulation, read exact model
parameters directly (never share `.robot_calibration.json` between sim and
real); skip fixed-camera calibration whenever a wrist camera alone sees the
workspace; skip `probe_surface_z` on stiff arms when RGB-D depth already gives
clean target Z.

1. **Cache, Intrinsics & Staleness**:
   - Store poses in the **robot base frame** and camera extrinsics in **OpenCV
     optical axes** (`+X` right, `+Y` down, `+Z` forward).
   - Call `load_calibration` / `save_calibration`, rescale `K` on resolution
     changes with `scale_intrinsics(K, from_wh, to_wh)`, and guard fixed-camera
     `T_world_cam` / `H` with `border_fingerprint(img)` and `is_border_stale`.
2. **Surface Height (`probe_surface_z`)**:
   - Never roll an ad-hoc `|z_fk - z_cmd|` probe from scratch (gravity sag causes
     mid-air false triggers). Call `probe_surface_z(move_z_fn, read_z_fk_fn,
     z_start, z_min, mode="compliant" | "stiff", read_force_fn=...)` over a
     clear patch with closed jaws; persist both `surface_z` (vision plane) and
     `surface_z_cmd` (table-level command baking in compliance).
3. **Hand-Eye (`verify_wrist_extrinsic_sign` / `fit_wrist_jacobian`) & Gripper**:
   - *CAD `T_ee_cam` prior exists*: Check axis sign once when populating the
     cache (or if the mount was bumped) via `verify_wrist_extrinsic_sign(...)`.
   - *Extrinsics unknown*: Execute two orthogonal hover nudges and call
     `fit_wrist_jacobian(duv_dx, duv_dy, step_x_m, step_y_m)` to get `J_xy_uv`,
     then step with `jacobian_step_xy(J_xy_uv, uv_target, uv_ref)`.
   - Record `closing_axis_ee`, `open_cmd`, `closed_cmd`, and
     `gripper_empty_closed` once in `.robot_calibration.json`.
   - *Fixed camera `T_world_cam` (eye-to-hand)*: with a gripper-tip marker at
     known FK positions, call `solve_fixed_camera_extrinsic_pnp(points_world,
     pixels_uv, K)`; with a calibration target held in the gripper, call
     `solve_fixed_camera_extrinsic_handeye(t_world_ee_list,
     t_cam_target_list)` using varied end-effector orientations. Both return
     `T_world_cam` (OpenCV optical axes) plus residuals and `valid`; persist it
     only when `valid` is true. Verify with `python3 scripts/calibration_test.py`.
   - Use fixed-camera `fit_plane_homography` only as a cached coarse seed for a
     wrist camera (or fallback when no wrist camera, depth, or `T_world_cam`
     exists).
