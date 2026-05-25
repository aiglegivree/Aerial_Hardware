#!/usr/bin/env python3
"""
Lap-one Evan controller for Crazyflie hardware.

This is the hardware-oriented version of the first-lap behavior in evan.py:
search for bright gate light in grayscale AI-deck frames, align, localize the
gate from a small number of bearing observations, cross it, then keep searching.

The real arena geometry is intentionally not hard-coded here. If you know the
map center/size, add it to the optional MAP_* settings below and tighten the
search/reposition behavior.

Keys:
  Q      immediate stop setpoint and close
  Space  immediate stop setpoint and close
"""

import contextlib
import math
import os
import socket
import struct
import sys
import threading
import time
from collections import deque

import cflib.crtp
import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

from cv_detection import DIST, K, detect_gate, find_corners_polydp, order_corners, render_detection


# Connection defaults from crazyflie_fpv_example_1.py. Override with CFLIB_URI.
URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E708")
AIDECK_IP = "192.168.4.1"
AIDECK_PORT = 5000
LOCAL_PORT = 5001
START_MAGIC = b"FER"

CPX_HEADER_SIZE = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE = 11
IMG_WIDTH = 324
IMG_HEIGHT = 244
MIN_JPEG_BYTES = 5000

# Flight tuning. Start gentle; increase only after confirming pose quality.
CRUISE_ALT = 0.85
TAKEOFF_DURATION = 3.0
LAND_DURATION = 3.0
SETPOINT_PERIOD = 0.05
NUM_GATES = 5

SEARCH_YAW_RATE_DEG = 12.0
SEARCH_FULL_TURN_DEG = 390.0
SEARCH_STEP_AFTER_FULL_TURN = 0.35
ALIGN_GAIN_DEG_PER_NORM = 18.0
ALIGN_DEADBAND = 0.10
APPROACH_NUDGE = 0.12

TRIANGULATION_FRAMES = 3
MIN_OBS_BASELINE = 0.18
MAX_OBS_RANGE = 3.5
MIN_DETECTION_AREA_REL = 0.006
LATERAL_SAMPLE_DIST = 0.22
DETECTION_THRESHOLD = 155
DETECTION_MIN_AREA = 120
DETECTION_MEDIAN_KSIZE = 3
DETECTION_CLOSE_SHORT = 7
DETECTION_CLOSE_LONG = 3
DETECTION_DILATE_EXTRA = 1
DETECTION_BORDER_PAD = 0
DETECTION_CLOSE_COVERAGE = 1.00

APPROACH_DIST = 0.45
EXIT_DIST = 0.65
PASS_TOL = 0.13
ADVANCE_AFTER_GATE_TIME = 1.0
ADVANCE_AFTER_GATE_DIST = 0.35

# Hardware arena geometry. Tune MAP_RADIUS after flight logs if the gate ring is
# tighter or wider, but 1.2 m is a safe first estimate for this course.
MAP_CENTER_XY = np.array([1.2, 0.0], dtype=float)
MAP_RADIUS = 1.2

# Camera model. cv_detection returns undistorted normalized image coordinates.
R_CAM_TO_BODY = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)
CAMERA_OFFSET_BODY = np.array([0.03, 0.0, 0.01], dtype=float)


@contextlib.contextmanager
def muted_stderr():
    saved = os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(null)
        os.close(saved)


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def quat_to_matrix(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


class StateBuffer:
    def __init__(self, max_len=100):
        self._buf = deque(maxlen=max_len)
        self._lock = threading.Lock()

    def push(self, state):
        with self._lock:
            self._buf.append((time.time(), dict(state)))

    @property
    def latest(self):
        with self._lock:
            return dict(self._buf[-1][1]) if self._buf else None


class UdpVideoThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    overlay_ready = QtCore.pyqtSignal(np.ndarray)
    status_ready = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self._lock = threading.Lock()
        self._frame = None
        self._frame_ts = 0.0
        self._last_detection = None
        self._stats = {
            "packets": 0,
            "headers": 0,
            "frames": 0,
            "decode_failures": 0,
            "timeouts": 0,
            "last_packet_ts": 0.0,
        }

    @property
    def latest_frame_with_ts(self):
        with self._lock:
            return self._frame, self._frame_ts

    @property
    def last_detection(self):
        with self._lock:
            return self._last_detection

    def set_detection(self, frame, result):
        if frame is None or result is None:
            return
        with self._lock:
            self._last_detection = result
        try:
            self.overlay_ready.emit(render_detection(frame, result))
        except Exception:
            self.frame_ready.emit(frame)

    def stop(self):
        self._running = False

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(("0.0.0.0", LOCAL_PORT))
        sock.settimeout(1.0)
        sock.sendto(START_MAGIC, (AIDECK_IP, AIDECK_PORT))

        buffer = bytearray()
        expected_size = 0
        receiving = False
        last_status = 0.0

        while self._running:
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                self._stats["timeouts"] += 1
                try:
                    sock.sendto(START_MAGIC, (AIDECK_IP, AIDECK_PORT))
                except OSError:
                    pass
                continue
            except OSError:
                break

            now = time.time()
            self._stats["packets"] += 1
            self._stats["last_packet_ts"] = now

            if len(data) < CPX_HEADER_SIZE:
                continue
            payload = data[CPX_HEADER_SIZE:]

            if len(payload) >= IMG_HEADER_SIZE and payload[0] == IMG_HEADER_MAGIC:
                _, w, h, _, _, size = struct.unpack("<BHHBBI", payload[:IMG_HEADER_SIZE])
                if w == IMG_WIDTH and h == IMG_HEIGHT and 0 < size < 65536:
                    self._stats["headers"] += 1
                    expected_size = size
                    buffer = bytearray()
                    receiving = True
                    continue

            if not receiving:
                continue
            buffer.extend(payload)

            if len(buffer) >= expected_size:
                frame = self._decode(buffer)
                if frame is None:
                    self._stats["decode_failures"] += 1
                else:
                    with self._lock:
                        self._frame = frame
                        self._frame_ts = time.time()
                        self._stats["frames"] += 1
                    self.frame_ready.emit(frame)
                receiving = False

            if now - last_status > 1.0:
                self.status_ready.emit(self._format_status())
                last_status = now

        sock.close()

    def _decode(self, buffer):
        soi = buffer.find(b"\xff\xd8")
        eoi = buffer.rfind(b"\xff\xd9")
        if soi < 0 or eoi <= soi:
            return None
        jpeg_len = eoi + 2 - soi
        if jpeg_len < MIN_JPEG_BYTES:
            return None
        jpeg = np.frombuffer(buffer, np.uint8, count=jpeg_len, offset=soi)
        with muted_stderr():
            img = cv2.imdecode(jpeg, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
            return None
        if img.ndim == 2:
            return img
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return None

    def _format_status(self):
        now = time.time()
        frame_age = now - self._frame_ts if self._frame_ts else float("inf")
        packet_age = now - self._stats["last_packet_ts"] if self._stats["last_packet_ts"] else float("inf")
        return (
            f'pkts={self._stats["packets"]} headers={self._stats["headers"]} '
            f'frames={self._stats["frames"]} fail={self._stats["decode_failures"]} '
            f'timeout={self._stats["timeouts"]} frame_age={frame_age:.1f}s '
            f'packet_age={packet_age:.1f}s'
        )


class LapOneEvanController:
    def __init__(self, cf, cam):
        self.cf = cf
        self.cam = cam
        self.state_buf = StateBuffer()
        self.stop_requested = False
        self.connected = False

        self.log = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "yaw": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
        }
        self.log_lock = threading.Lock()
        self.log_ready = threading.Event()

        self.last_frame_ts = 0.0
        self.mission_state = "TAKEOFF"
        self.gate_count = 0
        self.search_yaw = 0.0
        self.search_rotated = 0.0
        self.hold = [0.0, 0.0, CRUISE_ALT, 0.0]

        self.observations = []
        self.sample_side = 1.0
        self.sample_target = None
        self.travel_targets = []
        self.travel_index = 0
        self.advance_until = 0.0
        self.advance_heading_deg = 0.0
        self.lost_frames = 0

        self._setup_logs()

    def _setup_logs(self):
        pos = LogConfig(name="LapOnePosVel", period_in_ms=50)
        pos.add_variable("stateEstimate.x", "float")
        pos.add_variable("stateEstimate.y", "float")
        pos.add_variable("stateEstimate.z", "float")
        pos.add_variable("stateEstimate.vx", "float")
        pos.add_variable("stateEstimate.vy", "float")
        pos.add_variable("stateEstimate.vz", "float")

        att = LogConfig(name="LapOneAtt", period_in_ms=50)
        att.add_variable("stabilizer.yaw", "float")
        att.add_variable("stateEstimate.qx", "float")
        att.add_variable("stateEstimate.qy", "float")
        att.add_variable("stateEstimate.qz", "float")
        att.add_variable("stateEstimate.qw", "float")

        try:
            self.cf.log.add_config(pos)
            pos.data_received_cb.add_callback(self._log_cb)
            pos.start()
            self.cf.log.add_config(att)
            att.data_received_cb.add_callback(self._log_cb)
            att.start()
            self.connected = True
        except Exception as exc:
            print(f"Log setup failed: {exc}")

    def _log_cb(self, timestamp, data, logconf):
        with self.log_lock:
            mapping = {
                "stateEstimate.x": "x",
                "stateEstimate.y": "y",
                "stateEstimate.z": "z",
                "stateEstimate.vx": "vx",
                "stateEstimate.vy": "vy",
                "stateEstimate.vz": "vz",
                "stabilizer.yaw": "yaw",
                "stateEstimate.qx": "qx",
                "stateEstimate.qy": "qy",
                "stateEstimate.qz": "qz",
                "stateEstimate.qw": "qw",
            }
            for src, dst in mapping.items():
                if src in data:
                    self.log[dst] = data[src]
            snap = dict(self.log)
        self.state_buf.push(snap)
        self.log_ready.set()

    def _state(self):
        with self.log_lock:
            return dict(self.log)

    def _send(self, x, y, z, yaw_deg):
        self.cf.commander.send_position_setpoint(float(x), float(y), float(z), float(yaw_deg))

    def _hold(self):
        self._send(*self.hold)

    def _stop_motors(self):
        self.cf.commander.send_stop_setpoint()

    def _dist(self, s, target):
        return math.sqrt((s["x"] - target[0]) ** 2 + (s["y"] - target[1]) ** 2 + (s["z"] - target[2]) ** 2)

    def _detect(self):
        frame, ts = self.cam.latest_frame_with_ts
        if frame is None or ts == self.last_frame_ts:
            return None
        self.last_frame_ts = ts

        filtered = cv2.medianBlur(frame, DETECTION_MEDIAN_KSIZE) if DETECTION_MEDIAN_KSIZE > 1 else frame
        result = detect_gate(
            filtered,
            threshold=DETECTION_THRESHOLD,
            min_area=DETECTION_MIN_AREA,
            border_pad=DETECTION_BORDER_PAD,
            close_coverage=DETECTION_CLOSE_COVERAGE,
            close_short=DETECTION_CLOSE_SHORT,
            close_long=DETECTION_CLOSE_LONG,
            dilate_extra=DETECTION_DILATE_EXTRA,
            open_size=1,
        )
        result = self._choose_rightmost_detection(result)
        self.cam.set_detection(filtered, result)
        if result["status"] == "commit_to_pass":
            return {"status": "commit_to_pass", "result": result}
        if result["status"] != "ok":
            return None

        quad_pix = result["quad_pix"]
        quad_norm = result["quad_norm"]
        if quad_pix is None or quad_norm is None:
            return None
        h, w = frame.shape[:2]
        cx = float(np.mean(quad_pix[:, 0]))
        cy = float(np.mean(quad_pix[:, 1]))
        area = float(cv2.contourArea(result["selected"])) if result["selected"] is not None else 0.0
        area_rel = area / float(w * h)
        if area_rel < MIN_DETECTION_AREA_REL:
            return None

        return {
            "status": "ok",
            "result": result,
            "center_norm": np.mean(quad_norm, axis=0),
            "dx": (cx - 0.5 * w) / (0.5 * w),
            "dy": (cy - 0.5 * h) / (0.5 * h),
            "area_rel": area_rel,
        }

    def _choose_rightmost_detection(self, result):
        candidates = list(result.get("accepted") or [])
        candidates.extend(result.get("rejected_border") or [])
        if result.get("commit_to_pass") is not None:
            candidates.append(result["commit_to_pass"])
        if not candidates:
            return result

        def contour_center_x(contour):
            moments = cv2.moments(contour)
            if abs(moments["m00"]) > 1e-6:
                return moments["m10"] / moments["m00"]
            x, _, w, _ = cv2.boundingRect(contour)
            return x + 0.5 * w

        rightmost = max(candidates, key=contour_center_x)
        if rightmost is result.get("selected"):
            return result

        corners, approx = find_corners_polydp(rightmost)
        result = dict(result)
        result["selected"] = rightmost
        result["smoothed"] = approx
        result["curvature"] = None
        result["quad_pix"] = None
        result["quad_norm"] = None

        if corners is None:
            result["status"] = "no_corners"
            return result

        corners = order_corners(corners)
        result["quad_pix"] = corners
        result["quad_norm"] = cv2.undistortPoints(
            corners.reshape(-1, 1, 2).astype(np.float32), K, DIST
        ).reshape(-1, 2)
        result["status"] = "ok"
        return result

    def _camera_ray(self, s, center_norm):
        rotation = quat_to_matrix(s["qx"], s["qy"], s["qz"], s["qw"])
        ray_cam = np.array([center_norm[0], center_norm[1], 1.0], dtype=float)
        ray_world = rotation @ (R_CAM_TO_BODY @ ray_cam)
        norm = np.linalg.norm(ray_world)
        if norm < 1e-6:
            return None
        origin = np.array([s["x"], s["y"], s["z"]], dtype=float) + rotation @ CAMERA_OFFSET_BODY
        return origin, ray_world / norm

    def _add_observation(self, s, det):
        ray = self._camera_ray(s, det["center_norm"])
        if ray is None:
            return False
        origin, direction = ray

        if self.observations:
            spacing = max(np.linalg.norm(origin - obs["origin"]) for obs in self.observations)
            if spacing < MIN_OBS_BASELINE and len(self.observations) >= 1:
                return False

        self.observations.append(
            {
                "origin": origin,
                "direction": direction,
                "yaw_deg": s["yaw"],
                "area_rel": det["area_rel"],
            }
        )
        print(f'[OBS] {len(self.observations)}/{TRIANGULATION_FRAMES} area={det["area_rel"]:.3f}')
        return True

    def _triangulate_center(self):
        if len(self.observations) < TRIANGULATION_FRAMES:
            raise ValueError("not enough observations")

        a = np.zeros((3, 3), dtype=float)
        b = np.zeros(3, dtype=float)
        for obs in self.observations:
            d = obs["direction"]
            o = obs["origin"]
            projector = np.eye(3) - np.outer(d, d)
            weight = 1.0 + 25.0 * obs["area_rel"]
            a += weight * projector
            b += weight * projector @ o
        center = np.linalg.solve(a, b)

        dists = [np.linalg.norm(np.cross(center - obs["origin"], obs["direction"])) for obs in self.observations]
        if max(dists) > 0.45:
            raise ValueError(f"ray residual too large: {max(dists):.2f} m")

        first = self.observations[0]["origin"]
        range_est = float(np.linalg.norm(center - first))
        if range_est > MAX_OBS_RANGE:
            raise ValueError(f"gate range too large: {range_est:.2f} m")
        return center

    def _set_gate_targets(self, center, s):
        drone_xy = np.array([s["x"], s["y"]], dtype=float)
        gate_xy = np.array([center[0], center[1]], dtype=float)
        to_gate = gate_xy - drone_xy
        norm = np.linalg.norm(to_gate)
        if norm < 1e-6:
            yaw_rad = math.radians(s["yaw"])
            forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=float)
        else:
            forward = to_gate / norm

        yaw_deg = math.degrees(math.atan2(forward[1], forward[0]))
        z = float(np.clip(center[2], CRUISE_ALT - 0.25, CRUISE_ALT + 0.25))

        app = gate_xy - APPROACH_DIST * forward
        mid = gate_xy
        exit_xy = gate_xy + EXIT_DIST * forward
        self.travel_targets = [
            [app[0], app[1], z, yaw_deg],
            [mid[0], mid[1], z, yaw_deg],
            [exit_xy[0], exit_xy[1], z, yaw_deg],
        ]
        self.travel_index = 0
        self.advance_heading_deg = yaw_deg
        print(f"[GATE] center=({center[0]:.2f},{center[1]:.2f},{z:.2f}) yaw={yaw_deg:.1f}")

    def _start_sampling_move(self, s):
        yaw_rad = math.radians(s["yaw"])
        side = self.sample_side
        self.sample_side *= -1.0
        self.sample_target = [
            s["x"] + side * LATERAL_SAMPLE_DIST * math.sin(yaw_rad),
            s["y"] - side * LATERAL_SAMPLE_DIST * math.cos(yaw_rad),
            CRUISE_ALT,
            s["yaw"],
        ]
        self.mission_state = "SAMPLE_MOVE"
        print(f"[SAMPLE_MOVE] to ({self.sample_target[0]:.2f},{self.sample_target[1]:.2f})")

    def _next_map_orbit_hold(self, s):
        current_xy = np.array([s["x"], s["y"]], dtype=float)
        relative = current_xy - MAP_CENTER_XY
        if np.linalg.norm(relative) < 1e-6:
            relative = np.array([-1.0, 0.0], dtype=float)

        angle = math.atan2(relative[1], relative[0]) + math.radians(18.0)
        target_xy = MAP_CENTER_XY + MAP_RADIUS * np.array([math.cos(angle), math.sin(angle)])
        yaw_to_center = math.degrees(
            math.atan2(MAP_CENTER_XY[1] - target_xy[1], MAP_CENTER_XY[0] - target_xy[0])
        )
        return [float(target_xy[0]), float(target_xy[1]), CRUISE_ALT, yaw_to_center]

    def _force_pass(self, s):
        yaw_rad = math.radians(s["yaw"])
        forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=float)
        start = np.array([s["x"], s["y"]], dtype=float)
        self.travel_targets = [
            [start[0] + 0.35 * forward[0], start[1] + 0.35 * forward[1], CRUISE_ALT, s["yaw"]],
            [start[0] + EXIT_DIST * forward[0], start[1] + EXIT_DIST * forward[1], CRUISE_ALT, s["yaw"]],
        ]
        self.travel_index = 0
        self.advance_heading_deg = s["yaw"]
        self.mission_state = "TRAVEL_GATE"
        print("[COMMIT] gate is close; passing straight through")

    def takeoff(self, x, y, yaw_deg):
        start_z = self._state()["z"]
        steps = max(1, int(TAKEOFF_DURATION / SETPOINT_PERIOD))
        print(f"[TAKEOFF] to {CRUISE_ALT:.2f} m")
        for i in range(1, steps + 1):
            if self.stop_requested:
                return
            z = start_z + (CRUISE_ALT - start_z) * i / steps
            self._send(x, y, z, yaw_deg)
            time.sleep(SETPOINT_PERIOD)
        self.hold = [x, y, CRUISE_ALT, yaw_deg]

    def land(self):
        print("[LAND]")
        s = self._state()
        start_z = max(s["z"], 0.05)
        steps = max(1, int(LAND_DURATION / SETPOINT_PERIOD))
        for i in range(1, steps + 1):
            if self.stop_requested and i > 3:
                break
            z = start_z * (1.0 - i / steps)
            self._send(s["x"], s["y"], max(0.0, z), s["yaw"])
            time.sleep(SETPOINT_PERIOD)
        self._stop_motors()

    def run_mission(self):
        self.cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        self.cf.param.set_value("kalman.resetEstimation", "0")
        time.sleep(2.0)

        if not self.log_ready.wait(timeout=5.0):
            print("No state estimate received; aborting")
            return

        s = self._state()
        self.search_yaw = s["yaw"]
        self.hold = [s["x"], s["y"], CRUISE_ALT, s["yaw"]]
        self.takeoff(s["x"], s["y"], s["yaw"])
        self.mission_state = "SEARCH"
        print("[SEARCH]")

        try:
            while not self.stop_requested:
                s = self._state()
                det = self._detect()

                if det is not None and det["status"] == "commit_to_pass":
                    self._force_pass(s)

                if self.mission_state == "SEARCH":
                    if det is not None and det["status"] == "ok":
                        self.mission_state = "TRACK"
                        self.observations = []
                        self.lost_frames = 0
                        print("[TRACK]")
                    else:
                        step = SEARCH_YAW_RATE_DEG * SETPOINT_PERIOD
                        self.search_yaw += step
                        self.search_rotated += abs(step)
                        if self.search_rotated >= SEARCH_FULL_TURN_DEG:
                            if MAP_CENTER_XY is not None and MAP_RADIUS is not None:
                                self.hold = self._next_map_orbit_hold(s)
                                self.search_yaw = self.hold[3]
                            else:
                                yaw_rad = math.radians(s["yaw"])
                                self.hold[0] = s["x"] + SEARCH_STEP_AFTER_FULL_TURN * math.cos(yaw_rad)
                                self.hold[1] = s["y"] + SEARCH_STEP_AFTER_FULL_TURN * math.sin(yaw_rad)
                            self.search_rotated = 0.0
                        self.hold = [self.hold[0], self.hold[1], CRUISE_ALT, self.search_yaw]
                        self._hold()

                elif self.mission_state == "TRACK":
                    if det is None:
                        self.lost_frames += 1
                        self._hold()
                        if self.lost_frames > 9:
                            print("[SEARCH] gate lost")
                            self.observations = []
                            self.search_yaw = s["yaw"]
                            self.search_rotated = 0.0
                            self.mission_state = "SEARCH"
                    elif det["status"] == "ok":
                        self.lost_frames = 0
                        yaw_target = s["yaw"] - ALIGN_GAIN_DEG_PER_NORM * det["dx"]
                        forward = APPROACH_NUDGE if abs(det["dx"]) < ALIGN_DEADBAND else 0.0
                        yaw_rad = math.radians(s["yaw"])
                        self.hold = [
                            s["x"] + forward * math.cos(yaw_rad),
                            s["y"] + forward * math.sin(yaw_rad),
                            CRUISE_ALT,
                            yaw_target,
                        ]
                        self._hold()

                        if abs(det["dx"]) < ALIGN_DEADBAND:
                            self._add_observation(s, det)
                            if len(self.observations) >= TRIANGULATION_FRAMES:
                                try:
                                    center = self._triangulate_center()
                                    self._set_gate_targets(center, s)
                                    self.mission_state = "TRAVEL_GATE"
                                    print("[TRAVEL_GATE]")
                                except Exception as exc:
                                    print(f"[TRIANGULATE] failed: {exc}; resampling")
                                    self.observations = []
                                    self._start_sampling_move(s)
                            else:
                                self._start_sampling_move(s)

                elif self.mission_state == "SAMPLE_MOVE":
                    if self.sample_target is None:
                        self.mission_state = "TRACK"
                    else:
                        self._send(*self.sample_target)
                        if self._dist(s, self.sample_target) < PASS_TOL:
                            self.hold = list(self.sample_target)
                            self.sample_target = None
                            self.mission_state = "TRACK"
                            print("[TRACK] sample position reached")

                elif self.mission_state == "TRAVEL_GATE":
                    if self.travel_index >= len(self.travel_targets):
                        self.gate_count += 1
                        print(f"[PASSED] gate {self.gate_count}/{NUM_GATES}")
                        self.observations = []
                        self.travel_targets = []
                        self.advance_until = time.time() + ADVANCE_AFTER_GATE_TIME
                        self.mission_state = "ADVANCE"
                    else:
                        target = self.travel_targets[self.travel_index]
                        self._send(*target)
                        if self._dist(s, target) < PASS_TOL:
                            self.travel_index += 1

                elif self.mission_state == "ADVANCE":
                    yaw_rad = math.radians(self.advance_heading_deg)
                    self._send(
                        s["x"] + ADVANCE_AFTER_GATE_DIST * math.cos(yaw_rad),
                        s["y"] + ADVANCE_AFTER_GATE_DIST * math.sin(yaw_rad),
                        CRUISE_ALT,
                        self.advance_heading_deg,
                    )
                    if time.time() >= self.advance_until:
                        if self.gate_count >= NUM_GATES:
                            self.mission_state = "DONE"
                            self.hold = [s["x"], s["y"], CRUISE_ALT, s["yaw"]]
                            print("[DONE] hovering")
                        else:
                            self.search_yaw = s["yaw"]
                            self.search_rotated = 0.0
                            self.hold = [s["x"], s["y"], CRUISE_ALT, s["yaw"]]
                            self.mission_state = "SEARCH"
                            print("[SEARCH]")

                elif self.mission_state == "DONE":
                    self._hold()

                time.sleep(SETPOINT_PERIOD)
        except Exception as exc:
            print(f"Unhandled exception: {exc}")
        finally:
            try:
                self.land()
            except Exception:
                self._stop_motors()


class LapOneWindow(QtWidgets.QWidget):
    connected_signal = QtCore.pyqtSignal(str)
    connection_failed_signal = QtCore.pyqtSignal(str)
    disconnected_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lap One Evan")
        self.image_label = QtWidgets.QLabel("Waiting for video...")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_label = QtWidgets.QLabel(f"Connecting to {URI}...")
        self.camera_label = QtWidgets.QLabel("Camera: starting...")
        self.mission_label = QtWidgets.QLabel("Mission: idle")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.camera_label)
        layout.addWidget(self.mission_label)

        self.video = UdpVideoThread(self)
        self.video.frame_ready.connect(self._show_gray)
        self.video.overlay_ready.connect(self._show_rgb)
        self.video.status_ready.connect(lambda text: self.camera_label.setText(f"Camera: {text}"))
        self.video.start()

        self.cf = Crazyflie(rw_cache="cache")
        self.ctrl = None
        self.mission_thread = None

        self.connected_signal.connect(self._on_connected)
        self.connection_failed_signal.connect(lambda msg: self.status_label.setText(f"Connection failed: {msg}"))
        self.disconnected_signal.connect(lambda uri: self.status_label.setText(f"Disconnected from {uri}"))
        self.cf.connected.add_callback(self.connected_signal.emit)
        self.cf.connection_failed.add_callback(lambda uri, msg: self.connection_failed_signal.emit(msg))
        self.cf.disconnected.add_callback(self.disconnected_signal.emit)
        self.cf.open_link(URI)

        self.monitor = QtCore.QTimer(self)
        self.monitor.timeout.connect(self._monitor)
        self.monitor.setInterval(250)
        self.monitor.start()

    def _on_connected(self, uri):
        self.status_label.setText(f"Connected to {uri}; starting lap one")
        try:
            self.cf.supervisor.send_arming_request(True)
        except Exception:
            pass
        self.ctrl = LapOneEvanController(self.cf, self.video)
        if not self.ctrl.connected:
            self.status_label.setText("Log setup failed; mission not started")
            return
        self.mission_thread = threading.Thread(target=self.ctrl.run_mission, daemon=True)
        self.mission_thread.start()

    def _monitor(self):
        if self.ctrl is not None:
            self.mission_label.setText(
                f"Mission: {self.ctrl.mission_state}  gate {self.ctrl.gate_count}/{NUM_GATES}"
            )

    def _show_gray(self, gray):
        h, w = gray.shape
        qimg = QtGui.QImage(gray.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8).copy()
        self.image_label.setPixmap(self._scaled_pixmap(qimg))

    def _show_rgb(self, rgb):
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
        self.image_label.setPixmap(self._scaled_pixmap(qimg))

    def _scaled_pixmap(self, qimg):
        return QtGui.QPixmap.fromImage(qimg).scaled(
            qimg.width() * 2,
            qimg.height() * 2,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

    def _controlled_stop(self):
        if self.ctrl is not None:
            self.ctrl.stop_requested = True
            self.status_label.setText("Stop requested; landing...")
        else:
            self.close()

    def _immediate_stop(self):
        if self.ctrl is not None:
            self.ctrl.stop_requested = True
        try:
            self.cf.commander.send_stop_setpoint()
        except Exception:
            pass
        self.close()

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        if event.key() == QtCore.Qt.Key.Key_Q:
            self._immediate_stop()
            return
        if event.key() == QtCore.Qt.Key.Key_Space:
            self._immediate_stop()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.monitor.stop()
        if self.ctrl is not None:
            self.ctrl.stop_requested = True
        self.video.stop()
        self.video.wait(1000)
        try:
            self.cf.commander.send_stop_setpoint()
        except Exception:
            pass
        try:
            self.cf.close_link()
        except Exception:
            pass
        event.accept()


if __name__ == "__main__":
    cflib.crtp.init_drivers()
    app = QtWidgets.QApplication(sys.argv)
    win = LapOneWindow()
    win.resize(760, 660)
    win.show()
    sys.exit(app.exec())
