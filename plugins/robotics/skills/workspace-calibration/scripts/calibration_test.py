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
"""Synthetic round-trip tests for fixed-camera extrinsic solvers.

Run: `python -m unittest calibration_test` from this directory.
"""

import pathlib
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import calibration  # pylint: disable=g-import-not-at-top

K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])


def _pose(rvec, tvec):
  t = np.eye(4)
  t[:3, :3], _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
  t[:3, 3] = tvec
  return t


def _look_at_camera() -> np.ndarray:
  """Camera 1 m above and in front of the base, looking at the workspace."""
  eye = np.array([0.9, -0.6, 0.9])
  target = np.array([0.35, 0.0, 0.05])
  z = target - eye
  z /= np.linalg.norm(z)
  x = np.cross(z, [0.0, 0.0, 1.0])
  x /= np.linalg.norm(x)
  y = np.cross(z, x)
  t = np.eye(4)
  t[:3, :3] = np.stack([x, y, z], axis=1)
  t[:3, 3] = eye
  return t


def _angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
  return np.degrees(calibration._rot_angle_rad(r_a.T @ r_b))  # pylint: disable=protected-access


class FixedCameraExtrinsicTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = np.random.default_rng(0)
    self.t_world_cam = _look_at_camera()

  def test_pnp_round_trip(self):
    pts = np.column_stack([
        self.rng.uniform(0.2, 0.5, 30),
        self.rng.uniform(-0.2, 0.2, 30),
        self.rng.uniform(0.0, 0.3, 30),
    ])
    t_cam_world = np.linalg.inv(self.t_world_cam)
    rvec, _ = cv2.Rodrigues(t_cam_world[:3, :3])
    uv, _ = cv2.projectPoints(pts, rvec, t_cam_world[:3, 3], K, None)
    uv = uv.reshape(-1, 2) + self.rng.normal(0.0, 0.5, (30, 2))
    uv[3] += [150.0, -90.0]  # one gross outlier the RANSAC must reject.

    res = calibration.solve_fixed_camera_extrinsic_pnp(pts, uv, K)

    self.assertTrue(res["valid"])
    self.assertLess(res["reproj_rms_px"], 1.0)
    self.assertLess(
        np.linalg.norm(res["T_world_cam"][:3, 3] - self.t_world_cam[:3, 3]),
        0.01,
    )
    self.assertLess(
        _angle_deg(res["T_world_cam"][:3, :3], self.t_world_cam[:3, :3]), 0.5
    )

  def test_pnp_pixels_unproject_back_to_points(self):
    pts = np.column_stack([
        self.rng.uniform(0.2, 0.5, 12),
        self.rng.uniform(-0.2, 0.2, 12),
        self.rng.uniform(0.0, 0.3, 12),
    ])
    t_cam_world = np.linalg.inv(self.t_world_cam)
    rvec, _ = cv2.Rodrigues(t_cam_world[:3, :3])
    uv, _ = cv2.projectPoints(pts, rvec, t_cam_world[:3, 3], K, None)
    uv = uv.reshape(-1, 2)
    res = calibration.solve_fixed_camera_extrinsic_pnp(pts, uv, K)
    # Points on the table plane must unproject to their true XY.
    flat = pts.copy()
    flat[:, 2] = 0.0
    uv_f, _ = cv2.projectPoints(flat, rvec, t_cam_world[:3, 3], K, None)
    uv_f = uv_f.reshape(-1, 2)
    for p, (u, v) in zip(flat, uv_f):
      got = calibration._unproject_pixel_to_plane(  # pylint: disable=protected-access
          u, v, K, res["T_world_cam"], 0.0
      )
      np.testing.assert_allclose(got, p, atol=2e-3)

  def test_pnp_rejects_too_few_points(self):
    with self.assertRaises(ValueError):
      calibration.solve_fixed_camera_extrinsic_pnp(
          np.zeros((3, 3)), np.zeros((3, 2)), K
      )

  def test_handeye_round_trip(self):
    t_ee_target = _pose([0.1, -0.2, 0.3], [0.0, 0.02, 0.08])
    t_we, t_ct = [], []
    for _ in range(14):
      rot = self.rng.uniform(-0.6, 0.6, 3)
      pos = [
          self.rng.uniform(0.25, 0.5),
          self.rng.uniform(-0.2, 0.2),
          self.rng.uniform(0.1, 0.35),
      ]
      # Gripper pointing down (180 deg about X) plus varied tilts.
      t_ee = _pose([np.pi, 0.0, 0.0], pos) @ _pose(rot, [0.0, 0.0, 0.0])
      t_we.append(t_ee)
      noise = _pose(self.rng.normal(0.0, 0.002, 3), self.rng.normal(0.0, 0.0005, 3))
      t_ct.append(np.linalg.inv(self.t_world_cam) @ t_ee @ t_ee_target @ noise)

    res = calibration.solve_fixed_camera_extrinsic_handeye(t_we, t_ct)

    self.assertTrue(res["valid"], res)
    self.assertLess(
        np.linalg.norm(res["T_world_cam"][:3, 3] - self.t_world_cam[:3, 3]),
        0.01,
    )
    self.assertLess(
        _angle_deg(res["T_world_cam"][:3, :3], self.t_world_cam[:3, :3]), 1.0
    )
    self.assertLess(
        np.linalg.norm(res["T_ee_target"][:3, 3] - t_ee_target[:3, 3]), 0.01
    )

  def test_handeye_flags_corrupted_data(self):
    t_ee_target = _pose([0.1, -0.2, 0.3], [0.0, 0.02, 0.08])
    t_we, t_ct = [], []
    for _ in range(10):
      t_ee = _pose([np.pi, 0.0, 0.0], [0.4, 0.0, 0.2]) @ _pose(
          self.rng.uniform(-0.6, 0.6, 3), [0.0, 0.0, 0.0]
      )
      t_we.append(t_ee)
      t_ct.append(np.linalg.inv(self.t_world_cam) @ t_ee @ t_ee_target)
    # Unrelated detections: the constant-T_ee_target consistency must fail.
    t_ct = [_pose(self.rng.uniform(-1, 1, 3), self.rng.uniform(-0.5, 0.5, 3)
                  + [0, 0, 1]) for _ in t_ct]
    res = calibration.solve_fixed_camera_extrinsic_handeye(t_we, t_ct)
    self.assertFalse(res["valid"])

  def test_handeye_rejects_mismatched_lists(self):
    with self.assertRaises(ValueError):
      calibration.solve_fixed_camera_extrinsic_handeye(
          [np.eye(4)] * 6, [np.eye(4)] * 5
      )


if __name__ == "__main__":
  unittest.main()
