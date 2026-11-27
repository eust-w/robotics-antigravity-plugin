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
"""Tests for the colored-marker fixed-camera calibration example.

Run: `python3 fixed_camera_marker_example_test.py` from this directory.
"""

import pathlib
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fixed_camera_marker_example as ex  # pylint: disable=g-import-not-at-top

K = np.array([[411.48, 0.0, 304.48], [0.0, 411.53, 280.74], [0.0, 0.0, 1.0]])
DIST = np.array([-0.3208, 0.1201, 0.0015, -0.0011, -0.0219])
OFFSET = np.array([-0.0269, 0.0062, -0.0931])


class MarkerExampleTest(unittest.TestCase):

  def test_detect_marker_pixel_centroid(self):
    img = np.full((480, 640, 3), 90, dtype=np.uint8)
    cv2.circle(img, (321, 254), 6, (200, 60, 20), -1)  # BGR blue.
    uv = ex.detect_marker_pixel(img)
    self.assertIsNotNone(uv)
    self.assertAlmostEqual(uv[0], 321.0, delta=0.5)
    self.assertAlmostEqual(uv[1], 254.0, delta=0.5)

  def test_detect_marker_pixel_none_when_absent(self):
    img = np.full((480, 640, 3), 90, dtype=np.uint8)
    self.assertIsNone(ex.detect_marker_pixel(img))

  def test_pipeline_recovers_camera_pose(self):
    images, poses, truth = ex.make_synthetic_frames(12, K, DIST, OFFSET)
    with tempfile.TemporaryDirectory() as d:
      res = ex.run(
          images, poses, K, DIST, OFFSET,
          ex.DEFAULT_HSV_LO, ex.DEFAULT_HSV_HI, pathlib.Path(d),
      )
      self.assertEqual(len(list(pathlib.Path(d).glob("overlay_*.png"))), 12)
    self.assertTrue(res["valid"])
    self.assertLess(res["reproj_rms_px"], 1.0)
    self.assertLess(
        np.linalg.norm(res["T_world_cam"][:3, 3] - truth[:3, 3]), 0.01
    )

  def test_wrong_offset_shows_in_error_and_sensitivity(self):
    images, poses, truth = ex.make_synthetic_frames(12, K, DIST, OFFSET)
    uvs = [ex.detect_marker_pixel(i) for i in images]
    good = ex.calibrate(poses, uvs, K, DIST, OFFSET)
    bad = ex.calibrate(poses, uvs, K, DIST, OFFSET + [0.0, 0.0, 0.03])
    self.assertGreater(bad["reproj_rms_px"], 3.0 * good["reproj_rms_px"])
    self.assertGreater(
        np.linalg.norm(bad["T_world_cam"][:3, 3] - truth[:3, 3]),
        np.linalg.norm(good["T_world_cam"][:3, 3] - truth[:3, 3]),
    )
    sens = ex.offset_sensitivity(poses, uvs, K, DIST, OFFSET)
    self.assertTrue(all(v > 0.0 for v in sens.values()))

  def test_tool_tilt_deg(self):
    down = np.eye(4)
    down[:3, :3], _ = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))  # +z down.
    self.assertAlmostEqual(ex.tool_tilt_deg(down), 0.0, places=6)
    tilted = down.copy()
    rot_x30, _ = cv2.Rodrigues(np.array([np.radians(30.0), 0.0, 0.0]))
    tilted[:3, :3] = rot_x30 @ down[:3, :3]
    self.assertAlmostEqual(ex.tool_tilt_deg(tilted), 30.0, places=4)

  def test_residual_report_groups_and_signs(self):
    images, poses, _ = ex.make_synthetic_frames(12, K, DIST, OFFSET)
    uvs = [ex.detect_marker_pixel(i) for i in images]
    res = ex.calibrate(poses, uvs, K, DIST, OFFSET)
    rep = ex.residual_report(poses, uvs, res)
    self.assertEqual(len(rep["per_pose"]), 12)
    self.assertEqual(sum(g["n"] for g in rep["groups"]), 12)
    # Signed residual must equal detected - re-projected for every pose.
    for r in rep["per_pose"]:
      i = r["index"]
      self.assertAlmostEqual(r["du_px"], uvs[i][0] - res["reprojected_uv"][i][0])
      self.assertAlmostEqual(r["dv_px"], uvs[i][1] - res["reprojected_uv"][i][1])


if __name__ == "__main__":
  unittest.main()
