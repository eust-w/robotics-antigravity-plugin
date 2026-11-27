# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fixed-camera extrinsic calibration from a colored marker on the gripper.

Example pipeline around `calibration.solve_fixed_camera_extrinsic_pnp`:

1. Stick a small colored sticker rigidly on the gripper body.
2. Move the arm through 6+ poses spread over the workspace and heights. For
   each pose save one camera frame and the 4x4 pose of the gripper's tool
   frame in the robot base frame (from your forward kinematics, e.g. LeRobot's
   `RobotKinematics.forward_kinematics(joints_deg)` for the SO-101 URDF
   `gripper_frame_link`; verify the return convention for your version).
3. Run this script: it detects the sticker pixel (HSV mask + centroid), turns
   the sticker offset (meters, in the tool-frame axes) into a base-frame point
   `p = T[:3, 3] + T[:3, :3] @ offset`, solves `T_world_cam`, and writes an
   overlay image per frame (green = detected, red = re-projected).

Samples file (`samples.json`; image paths are relative to the file):
  {"samples": [{"image": "f00.png", "t_world_frame": [[...4x4...]]}, ...]}
Intrinsics file: {"camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]],
                  "dist_coeffs": [k1, k2, p1, p2, k3]}
The intrinsics must belong to the same image resolution as the frames.

Usage:
  python3 fixed_camera_marker_example.py --demo --out-dir /tmp/fcm_demo
  python3 fixed_camera_marker_example.py --samples samples.json \
      --intrinsics intrinsics.json --offset -0.0269 0.0062 -0.0931 \
      --out-dir out [--write-calibration .robot_calibration.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import calibration  # pylint: disable=g-import-not-at-top

# OpenCV HSV ranges (H 0-179). Defaults suit a saturated blue sticker.
DEFAULT_HSV_LO = (100, 120, 60)
DEFAULT_HSV_HI = (130, 255, 255)


def detect_marker_pixel(
    image_bgr: np.ndarray,
    hsv_lo: tuple[int, int, int] = DEFAULT_HSV_LO,
    hsv_hi: tuple[int, int, int] = DEFAULT_HSV_HI,
    *,
    min_area_px: float = 20.0,
) -> tuple[float, float] | None:
  """Returns the `(u, v)` centroid of the largest HSV-matching blob, or None."""
  hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(
      hsv, np.array(hsv_lo, dtype=np.uint8), np.array(hsv_hi, dtype=np.uint8)
  )
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
  contours, _ = cv2.findContours(
      mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  if not contours:
    return None
  best = max(contours, key=cv2.contourArea)
  if cv2.contourArea(best) < float(min_area_px):
    return None
  m = cv2.moments(best)
  if m["m00"] <= 0.0:
    return None
  return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def marker_world_point(
    t_world_frame: np.ndarray, offset_m: np.ndarray
) -> np.ndarray:
  """Marker position in the base frame from the tool-frame pose and offset."""
  t = np.asarray(t_world_frame, dtype=np.float64).reshape(4, 4)
  return t[:3, 3] + t[:3, :3] @ np.asarray(offset_m, dtype=np.float64)


def reproject(
    points_world: np.ndarray,
    t_world_cam: np.ndarray,
    k_mat: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
  """Projects base-frame points into the camera, including distortion."""
  t_cam_world = np.linalg.inv(t_world_cam)
  rvec, _ = cv2.Rodrigues(t_cam_world[:3, :3])
  uv, _ = cv2.projectPoints(
      np.asarray(points_world, dtype=np.float64).reshape(-1, 3),
      rvec,
      t_cam_world[:3, 3],
      k_mat,
      dist,
  )
  return uv.reshape(-1, 2)


def calibrate(
    t_world_frames: list[np.ndarray],
    pixels_uv: list[tuple[float, float]],
    k_mat: np.ndarray,
    dist: np.ndarray,
    offset_m: np.ndarray,
) -> dict[str, Any]:
  """Solves `T_world_cam` and adds per-point re-projection errors."""
  pts = np.stack([marker_world_point(t, offset_m) for t in t_world_frames])
  uv = np.asarray(pixels_uv, dtype=np.float64)
  res = calibration.solve_fixed_camera_extrinsic_pnp(pts, uv, k_mat, dist)
  proj = reproject(pts, res["T_world_cam"], k_mat, dist)
  res["points_world"] = pts
  res["reprojected_uv"] = proj
  res["per_point_err_px"] = np.linalg.norm(proj - uv, axis=1)
  return res


def offset_sensitivity(
    t_world_frames: list[np.ndarray],
    pixels_uv: list[tuple[float, float]],
    k_mat: np.ndarray,
    dist: np.ndarray,
    offset_m: np.ndarray,
    *,
    delta_m: float = 0.005,
) -> dict[str, float]:
  """Camera-position shift (m) when each offset axis is off by `+-delta_m`."""
  base = calibrate(t_world_frames, pixels_uv, k_mat, dist, offset_m)
  out = {}
  for axis, name in enumerate("xyz"):
    shifts = []
    for sign in (-1.0, 1.0):
      off = np.asarray(offset_m, dtype=np.float64).copy()
      off[axis] += sign * delta_m
      try:
        alt = calibrate(t_world_frames, pixels_uv, k_mat, dist, off)
      except RuntimeError:
        shifts.append(float("nan"))
        continue
      shifts.append(
          float(
              np.linalg.norm(
                  alt["T_world_cam"][:3, 3] - base["T_world_cam"][:3, 3]
              )
          )
      )
    out[name] = float(np.nanmax(shifts)) if not np.all(np.isnan(shifts)) else float("nan")
  return out


def tool_tilt_deg(t_world_frame: np.ndarray) -> float:
  """Angle (deg) between the tool +z axis and straight down (base -Z)."""
  z_axis = np.asarray(t_world_frame, dtype=np.float64).reshape(4, 4)[:3, 2]
  cos_a = float(np.clip(z_axis @ np.array([0.0, 0.0, -1.0]), -1.0, 1.0))
  return float(np.degrees(np.arccos(cos_a)))


def residual_report(
    poses: list[np.ndarray],
    pixels_uv: list[tuple[float, float]],
    res: dict[str, Any],
    *,
    tilt_bins_deg: tuple[float, ...] = (10.0, 25.0),
) -> dict[str, Any]:
  """Per-pose signed residuals (detected - re-projected) grouped by tilt.

  Tilted poses carry most of the sensitivity to the principal point and to the
  marker offset, so a systematic bias in `dv` (or `du`) that appears in one
  tilt group hints at a wrong `cy` (or `cx`) or a wrong offset. The tool +z
  axis is assumed to point down at zero tilt; adjust if your frame differs.
  """
  uv = np.asarray(pixels_uv, dtype=np.float64)
  signed = uv - res["reprojected_uv"]
  tilts = np.array([tool_tilt_deg(t) for t in poses])
  edges = (0.0,) + tuple(tilt_bins_deg) + (180.0,)
  groups = []
  for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (tilts >= lo) & (tilts < hi)
    if not np.any(sel):
      continue
    groups.append({
        "tilt_range_deg": (lo, hi),
        "n": int(np.sum(sel)),
        "mean_du_px": float(np.mean(signed[sel, 0])),
        "mean_dv_px": float(np.mean(signed[sel, 1])),
        "rms_px": float(np.sqrt(np.mean(np.sum(signed[sel] ** 2, axis=1)))),
    })
  return {
      "per_pose": [
          {
              "index": i,
              "tilt_deg": float(tilts[i]),
              "du_px": float(signed[i, 0]),
              "dv_px": float(signed[i, 1]),
              "err_px": float(np.linalg.norm(signed[i])),
          }
          for i in range(len(poses))
      ],
      "groups": groups,
  }


def draw_overlay(
    image_bgr: np.ndarray,
    detected_uv: tuple[float, float],
    reprojected_uv: np.ndarray,
) -> np.ndarray:
  """Draws the detection (green) and the re-projected point (red)."""
  out = image_bgr.copy()
  d = (int(round(detected_uv[0])), int(round(detected_uv[1])))
  r = (int(round(reprojected_uv[0])), int(round(reprojected_uv[1])))
  cv2.circle(out, d, 9, (0, 255, 0), 2)
  cv2.drawMarker(out, r, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
  cv2.line(out, d, r, (0, 255, 255), 1)
  return out


def _demo_camera_pose() -> np.ndarray:
  """Camera ~1 m away, looking at the workspace (OpenCV optical axes)."""
  eye = np.array([0.9, -0.6, 0.9])
  z = np.array([0.35, 0.0, 0.05]) - eye
  z /= np.linalg.norm(z)
  x = np.cross(z, [0.0, 0.0, 1.0])
  x /= np.linalg.norm(x)
  t = np.eye(4)
  t[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
  t[:3, 3] = eye
  return t


def make_synthetic_frames(
    n_frames: int,
    k_mat: np.ndarray,
    dist: np.ndarray,
    offset_m: np.ndarray,
    size_wh: tuple[int, int] = (640, 480),
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
  """Renders blue-dot frames for random gripper poses (no hardware needed)."""
  rng = np.random.default_rng(seed)
  t_world_cam = _demo_camera_pose()
  images, poses = [], []
  while len(images) < n_frames:
    rvec = np.array([np.pi, 0.0, 0.0]) + rng.uniform(-0.5, 0.5, 3)
    t = np.eye(4)
    t[:3, :3], _ = cv2.Rodrigues(rvec)
    t[:3, 3] = [
        rng.uniform(0.25, 0.5), rng.uniform(-0.2, 0.2), rng.uniform(0.1, 0.35)
    ]
    uv = reproject(marker_world_point(t, offset_m), t_world_cam, k_mat, dist)[0]
    if not (20 < uv[0] < size_wh[0] - 20 and 20 < uv[1] < size_wh[1] - 20):
      continue
    img = np.full((size_wh[1], size_wh[0], 3), 90, dtype=np.uint8)
    img = cv2.add(
        img, rng.integers(0, 12, img.shape, dtype=np.uint8)
    )
    cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))), 6, (200, 60, 20), -1)
    images.append(img)
    poses.append(t)
  return images, poses, t_world_cam


def run(
    images: list[np.ndarray],
    poses: list[np.ndarray],
    k_mat: np.ndarray,
    dist: np.ndarray,
    offset_m: np.ndarray,
    hsv_lo: tuple[int, int, int],
    hsv_hi: tuple[int, int, int],
    out_dir: pathlib.Path,
) -> dict[str, Any]:
  """Detects, solves, writes overlays, and returns the result dict."""
  out_dir.mkdir(parents=True, exist_ok=True)
  kept_img, kept_pose, uvs = [], [], []
  for i, (img, pose) in enumerate(zip(images, poses)):
    uv = detect_marker_pixel(img, hsv_lo, hsv_hi)
    if uv is None:
      print(f"frame {i}: marker not found, skipped")
      continue
    kept_img.append(img)
    kept_pose.append(pose)
    uvs.append(uv)
  res = calibrate(kept_pose, uvs, k_mat, dist, offset_m)
  res["sensitivity_m_per_5mm"] = offset_sensitivity(
      kept_pose, uvs, k_mat, dist, offset_m
  )
  res["residuals"] = residual_report(kept_pose, uvs, res)
  for i, (img, uv) in enumerate(zip(kept_img, uvs)):
    cv2.imwrite(
        str(out_dir / f"overlay_{i:02d}.png"),
        draw_overlay(img, uv, res["reprojected_uv"][i]),
    )
  return res


def _print_report(res: dict[str, Any]) -> None:
  print(
      f"valid={res['valid']}  reproj_rms={res['reproj_rms_px']:.2f}px  "
      f"inliers={res['n_inliers']}/{res['n_points']}"
  )
  print("per-point error (px):", np.round(res["per_point_err_px"], 2).tolist())
  print("T_world_cam:\n", np.round(res["T_world_cam"], 4))
  print("per-pose residuals (detected - re-projected):")
  for r in res["residuals"]["per_pose"]:
    print(
        f"  pose {r['index']:2d}  tilt={r['tilt_deg']:5.1f} deg  "
        f"du={r['du_px']:+6.2f}  dv={r['dv_px']:+6.2f}  err={r['err_px']:5.2f} px"
    )
  print("grouped by tool tilt (a consistent dv/du sign hints at cy/cx or offset error):")
  for g in res["residuals"]["groups"]:
    lo, hi = g["tilt_range_deg"]
    print(
        f"  tilt [{lo:g}, {hi:g}) deg  n={g['n']}  "
        f"mean du={g['mean_du_px']:+.2f}  mean dv={g['mean_dv_px']:+.2f}  "
        f"rms={g['rms_px']:.2f} px"
    )
  s = res["sensitivity_m_per_5mm"]
  print(
      "camera position shift for a 5 mm offset error (m): "
      f"x={s['x']:.4f} y={s['y']:.4f} z={s['z']:.4f}"
  )


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  p.add_argument("--demo", action="store_true", help="Run on synthetic frames.")
  p.add_argument("--samples", type=pathlib.Path)
  p.add_argument("--intrinsics", type=pathlib.Path)
  p.add_argument(
      "--offset", type=float, nargs=3, default=[-0.0269, 0.0062, -0.0931],
      metavar=("X", "Y", "Z"),
      help="Marker offset in the tool-frame axes, meters.",
  )
  p.add_argument("--hsv-lo", type=int, nargs=3, default=list(DEFAULT_HSV_LO))
  p.add_argument("--hsv-hi", type=int, nargs=3, default=list(DEFAULT_HSV_HI))
  p.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("fcm_out"))
  p.add_argument("--write-calibration", type=pathlib.Path, default=None)
  a = p.parse_args()
  offset = np.array(a.offset, dtype=np.float64)

  if a.demo:
    k_mat = np.array(
        [[411.48, 0.0, 304.48], [0.0, 411.53, 280.74], [0.0, 0.0, 1.0]]
    )
    dist = np.array([-0.3208, 0.1201, 0.0015, -0.0011, -0.0219])
    images, poses, truth = make_synthetic_frames(12, k_mat, dist, offset)
  else:
    if a.samples is None or a.intrinsics is None:
      p.error("--samples and --intrinsics are required without --demo.")
    intr = json.loads(a.intrinsics.read_text(encoding="utf-8"))
    k_mat = np.asarray(intr["camera_matrix"], dtype=np.float64)
    dist = np.asarray(intr["dist_coeffs"], dtype=np.float64)
    spec = json.loads(a.samples.read_text(encoding="utf-8"))["samples"]
    images, poses, truth = [], [], None
    for s in spec:
      img = cv2.imread(str(a.samples.parent / s["image"]))
      if img is None:
        p.error(f"Cannot read image {s['image']}")
      images.append(img)
      poses.append(np.asarray(s["t_world_frame"], dtype=np.float64))

  res = run(
      images, poses, k_mat, dist, offset,
      tuple(a.hsv_lo), tuple(a.hsv_hi), a.out_dir,
  )
  _print_report(res)
  if truth is not None:
    err = np.linalg.norm(res["T_world_cam"][:3, 3] - truth[:3, 3])
    print(f"demo: camera position error vs ground truth = {err * 1000:.2f} mm")
  print(f"overlays written to {a.out_dir}")
  if a.write_calibration is not None:
    if not res["valid"]:
      print("Not writing calibration: result is not valid.")
      sys.exit(2)
    data = calibration.load_calibration(a.write_calibration)
    data.update({
        "T_world_cam": res["T_world_cam"],
        "camera_matrix": k_mat,
        "dist_coeffs": dist,
        "T_world_cam_source": "marker_pnp",
    })
    calibration.save_calibration(a.write_calibration, data)
    print(f"Saved T_world_cam to {a.write_calibration}")


if __name__ == "__main__":
  main()
