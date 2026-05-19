# -*- coding: utf-8 -*-
"""
Gate traversal controller — Crazyflie + AI deck + Lighthouse
=============================================================

Split links (like FPV example):
    • Control/logs via Crazyradio (CRTP)
    • Video via AI-deck UDP stream (JPEG)

This file is the deployment controller for the hardware project. It has
TWO missions, selected by the MISSION constant below:

  MISSION = 'vision'    → Part 1: detect gates with the camera and fly
                          through them by visual servoing  (run_mission)
  MISSION = 'position'  → Part 2: the exact gate positions are known, so
                          fly waypoints with send_position_setpoint as
                          fast as possible for N_LAPS laps
                          (run_position_mission)

Architecture:
    UdpVideoThread — listens to UDP packets from AI-deck, decodes JPEG
                     frames, stores latest in a shared slot.
    GateController — sends setpoints over the radio, reads Lighthouse
                     state from the log callback, reads camera detections
                     from UdpVideoThread.

Part 1 — visual servoing (no world-frame gate position needed):
  lateral error  cx - frame_cx          → vy   (strafe)
  vertical error cy_mid - cy            → Δz   (altitude nudge)
  size gap       GATE_SIZE_CLOSE - size → vx   (forward speed)

Part 2 — minimum-jerk trajectory (Lighthouse world frame):
  Mirrors the simulation's handle_fast_lap: build one waypoint list
  (start → pre-gate, gate centre, post-gate ×N → return), fit a single
  minimum-jerk polynomial trajectory through it, then follow it open-loop
  by streaming the time-indexed send_position_setpoints. Boundary safety
  clamps any waypoint whose radius exceeds 2 m.

Boundary safety:
  stateEstimate.x/y from Lighthouse → enforce 2 m radius hard limit.

State machine per gate (Part 1):  SEARCH → APPROACH → TRANSIT → (next gate)

Press 'q' at any time for emergency stop.
"""

import contextlib
import logging
import math
import os
import socket
import struct
import time
import threading
import warnings

import cv2
import numpy as np
from pynput import keyboard

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

from gate_detection import GATE_SIZE_MIN, GATE_SIZE_CLOSE, get_gate_detection

warnings.filterwarnings("ignore", message=".*supervisor subsystem requires CRTP.*")
logging.basicConfig(level=logging.ERROR)

# ── mission selection ──────────────────────────────────────────────────────────

MISSION = 'vision'   # 'vision' = Part 1 (camera) | 'position' = Part 2 (waypoints)

# ── connection ─────────────────────────────────────────────────────────────────

CONTROL_URI = uri_helper.uri_from_env(default='radio://0/20/2M/E7E7E7E708')

UDP_AIDECK_IP   = '192.168.4.1'
UDP_AIDECK_PORT = 5000
UDP_LOCAL_PORT  = 5001
UDP_START_MAGIC = b'FER'

# ── camera ─────────────────────────────────────────────────────────────────────

CAM_WIDTH  = 324
CAM_HEIGHT = 244

CPX_HEADER_SIZE  = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE  = 11
MIN_JPEG_BYTES   = 5000

# ── arena bounds (Lighthouse world frame) ──────────────────────────────────────
ARENA_X_MIN = -1.0   # m — back wall
ARENA_X_MAX = +3.0   # m — front wall
ARENA_Y_MIN = -0.9   # m — right wall
ARENA_Y_MAX = +0.9   # m — left wall

SAFETY_MARGIN_HARD = 0.15  # m — never fly closer than this to a wall

# ── flight ─────────────────────────────────────────────────────────────────────

CRUISE_ALT       = 0.6   # m ## The height position of the drone
TAKEOFF_DURATION = 3.0   # s
LAND_DURATION    = 3.0   # s

# IBVS gains — tune these on the real drone
KP_VX        = 0.010  # size error  (GATE_SIZE_CLOSE - size) → forward speed
KP_VY        = 0.005  # lateral pixel error (cx - cx_mid)    → strafe speed
KP_VZ        = 0.005  # vertical pixel error (cy_mid - cy)   → altitude delta
MAX_VX       = 1.0    # m/s — forward cap (never backward)
MAX_VY       = 0.5    # m/s — strafe cap
MAX_VZ_DELTA = 0.5    # m   — altitude adjustment cap

TRANSIT_VX   = 0.5    # m/s
TRANSIT_TIME = 2.0    # s

SEARCH_YAW_RATE = 5.0  # deg/s CCW
SEARCH_TIMEOUT  = 10.0  # s — double yaw rate after this
LOST_TOLERANCE  = 15    # consecutive no-detection frames before re-search

N_GATES = 5  # Part 1 vision gates to search for 

# ── Part 1: patrol search path ─────────────────────────────────────────────────
#
# In the vision lap the drone flies this circular patrol while watching the
# camera. The yaw-sweep search_for_gate() is only a FALLBACK, used when a
# patrol leg ends with no gate in view.
PATROL_RADIUS      = 1.2   # m — radius of the circular search path (Lighthouse frame)
WAYPOINT_TOL       = 0.20  # m — patrol-waypoint reached tolerance
WAYPOINT_KP        = 0.8   # position error (m) → velocity (m/s)
WAYPOINT_SPEED_MIN = 0.05  # m/s
WAYPOINT_SPEED_MAX = 0.30  # m/s
WAYPOINT_YAW_KP    = 1.5   # heading error (deg) → yaw rate (deg/s)

# One patrol waypoint per gate, evenly spaced counter-clockwise around the
# circle. Replace with explicit (x, y) coordinates once you know your arena.
NO_GATE_WAYPOINTS = [
    (0, -0.2)
]


# ── Part 2: position-based mission ─────────────────────────────────────────────
#
# Fill GATE_POSITIONS once the instructor gives you the exact gate positions.
# Each gate entry will be (x, y, z, theta, height, width)
GATE_POSITIONS = []

N_LAPS           = 2     # number of timed laps
PRE_GATE_OFFSET  = 0.2   # m — waypoint placed before the gate along its approach axis
POST_GATE_OFFSET = 0.2   # m — waypoint placed after the gate (clears the frame)
LAP_DURATION_S   = 30.0  # s — total minimum-jerk trajectory time per lap.

TRAJ_VEL_LIM     = 5.0  # m/s — planner velocity ceiling (high: real speed set by LAP_DURATION_S)
TRAJ_ACC_LIM     = 5.0  # m/s² — planner acceleration ceiling
TRAJ_DISC_STEPS  = 20    # trajectory setpoints generated per segment
POSITION_RATE_HZ = 20.0  # trajectory setpoint streaming rate
GATE_INWARD_BIAS = 0.0   # m — shift pre/post waypoints toward arena origin
                         #     to tighten the racing line. Start at 0, try
                         #     0.1–0.3 once basic laps work. Capped at
                         #     half the gate width minus a safety margin.


# ── camera thread ──────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _muted_stderr():
    saved = os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(null)
        os.close(saved)


class UdpVideoThread(threading.Thread):
    """
    Reads AI-deck UDP packets, decodes JPEG, exposes latest frame via .latest_frame.
    """

    def __init__(self):
        super().__init__(daemon=True, name='UdpVideoThread')
        self._lock = threading.Lock()
        self._frame = None
        self._running = False

    def start(self):
        self._running = True
        super().start()

    def stop(self):
        self._running = False

    @property
    def latest_frame(self):
        with self._lock:
            return self._frame

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(('0.0.0.0', UDP_LOCAL_PORT))
        sock.sendto(UDP_START_MAGIC, (UDP_AIDECK_IP, UDP_AIDECK_PORT))

        buffer = bytearray()
        expected_size = 0
        receiving = False

        while self._running:
            try:
                data, _ = sock.recvfrom(2048)
            except Exception:
                time.sleep(0.01)
                continue
            if len(data) < CPX_HEADER_SIZE:
                continue
            payload = data[CPX_HEADER_SIZE:]

            if len(payload) >= IMG_HEADER_SIZE and payload[0] == IMG_HEADER_MAGIC:
                _, w, h, _, _, size = struct.unpack(
                    '<BHHBBI', payload[:IMG_HEADER_SIZE])
                if w == CAM_WIDTH and h == CAM_HEIGHT and 0 < size < 65536:
                    expected_size = size
                    buffer = bytearray()
                    receiving = True
                    continue

            if not receiving:
                continue

            buffer.extend(payload)

            if len(buffer) >= expected_size:
                frame = self._decode_frame(buffer)
                if frame is not None:
                    with self._lock:
                        self._frame = frame
                receiving = False

    def _decode_frame(self, buffer):
        soi = buffer.find(b'\xff\xd8')
        eoi = buffer.rfind(b'\xff\xd9')
        if soi < 0 or eoi <= soi:
            return None
        jpeg_len = eoi + 2 - soi
        if jpeg_len < MIN_JPEG_BYTES:
            return None
        jpeg = np.frombuffer(buffer, np.uint8, count=jpeg_len, offset=soi)
        with _muted_stderr():
            img = cv2.imdecode(jpeg, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[:2] != (CAM_HEIGHT, CAM_WIDTH):
            return None
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


# ── helpers ────────────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _fill_holes(mask):
    """Fill dark regions fully enclosed by white — turns a closed gate
    outline into a solid rectangle so the contour shape tests are reliable."""
    flood = mask.copy()
    h, w = mask.shape
    scratch = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, scratch, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


# Gate-detection HSV mask. The gates are bright glowing LED frames, so we
# threshold on high brightness (V) with low-to-moderate saturation (S), any
# hue. Verified against myassignement/frames in noteboks/gate_detection.ipynb —
# adaptive grayscale thresholding looks for DARK objects and missed the gates.
GATE_HSV_LOWER = np.array([0,   0,   200], dtype=np.uint8)
GATE_HSV_UPPER = np.array([255, 100, 255], dtype=np.uint8)


def get_gate_detection(frame):
    """
    Detect the bright LED gate frame.
    Returns: cx, cy, size  (or None, None, None if no gate-shaped contour).

    The gate is a bright rectangle on a dark background, so it is segmented
    with an HSV brightness mask, then the most rectangular contour is picked.
    """
    if frame is None:
        return None, None, None

    # Bright-LED mask in HSV space
    if len(frame.shape) == 3:
        bgr = frame
    else:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GATE_HSV_LOWER, GATE_HSV_UPPER)

    # Close gaps along the thin gate edges, then fill the interior so the gate
    # becomes a solid rectangle (a broken outline fails the shape tests below).
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 20), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mask = _fill_holes(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = 0.0
    best_gate = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 250:
            continue

        rect = cv2.minAreaRect(cnt)
        (rcx, rcy), (rw, rh), angle = rect
        if rw < 10 or rh < 10:
            continue

        # Aspect ratio check
        short_side, long_side = sorted([rw, rh])
        aspect = short_side / long_side
        if aspect < 0.45:
            continue

        # Rectangularity check
        rotated_area = rw * rh
        rectangularity = area / rotated_area
        if rectangularity < 0.80:
            continue

        # Solidity check
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        if solidity < 0.85:
            continue

        # Score based on shape perfection and size
        score = rectangularity * solidity * np.log1p(area)
        if score > best_score:
            best_score = score
            # Return center x, center y, and the largest dimension as "size"
            best_gate = (rcx, rcy, max(rw, rh))

    if best_gate is not None:
        return best_gate[0], best_gate[1], best_gate[2]

    return None, None, None


# ── minimum-jerk trajectory planner ─────────────────────────────────────────────

class MotionPlanner3D:
    """Minimum-jerk polynomial trajectory generator (5th-order per segment)."""

    def run_planner(self, path_waypoints):
        poly_coeffs = self.compute_poly_coefficients(path_waypoints)
        self.trajectory_setpoints, self.time_setpoints = self.poly_setpoint_extraction(
            poly_coeffs, path_waypoints)

    def compute_poly_matrix(self, t):
        # Rows: [position, velocity, acceleration, jerk, snap]
        return np.array([
            [t**5,     t**4,    t**3,    t**2,  t,  1],
            [5*t**4,   4*t**3,  3*t**2,  2*t,   1,  0],
            [20*t**3,  12*t**2, 6*t,     2,     0,  0],
            [60*t**2,  24*t,    6,       0,     0,  0],
            [120*t,    24,      0,       0,     0,  0],
        ])

    def compute_poly_coefficients(self, path_waypoints):
        seg_times = np.diff(self.times)
        m = len(path_waypoints)
        poly_coeffs = np.zeros((6 * (m - 1), 3))

        for dim in range(3):
            A = np.zeros((6 * (m - 1), 6 * (m - 1)))
            b = np.zeros(6 * (m - 1))
            pos = np.array([p[dim] for p in path_waypoints])
            A_0 = self.compute_poly_matrix(0)

            row = 0
            for i in range(m - 1):
                A_f = self.compute_poly_matrix(seg_times[i])
                if i == 0:
                    A[row, i*6:(i+1)*6] = A_0[0]; b[row] = pos[i];   row += 1
                    A[row, i*6:(i+1)*6] = A_0[1]; b[row] = 0;        row += 1
                    A[row, i*6:(i+1)*6] = A_0[2]; b[row] = 0;        row += 1
                    A[row, i*6:(i+1)*6] = A_f[0]; b[row] = pos[i+1]; row += 1
                    A[row:row+4, i*6:(i+1)*6]     =  A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    b[row:row+4] = 0; row += 4
                elif i < m - 2:
                    A[row, i*6:(i+1)*6] = A_0[0]; b[row] = pos[i];   row += 1
                    A[row, i*6:(i+1)*6] = A_f[0]; b[row] = pos[i+1]; row += 1
                    A[row:row+4, i*6:(i+1)*6]     =  A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    b[row:row+4] = 0; row += 4
                elif i == m - 2:
                    A[row, i*6:(i+1)*6] = A_0[0]; b[row] = pos[i];   row += 1
                    A[row, i*6:(i+1)*6] = A_f[0]; b[row] = pos[i+1]; row += 1
                    A[row, i*6:(i+1)*6] = A_f[1]; b[row] = 0;        row += 1
                    A[row, i*6:(i+1)*6] = A_f[2]; b[row] = 0;        row += 1

            poly_coeffs[:, dim] = np.linalg.solve(A, b)
        return poly_coeffs

    def poly_setpoint_extraction(self, poly_coeffs, path_waypoints):
        n = self.disc_steps * len(self.times)
        x_vals, y_vals, z_vals = np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1))
        v_x, v_y, v_z = np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1))
        a_x, a_y, a_z = np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1))

        time_setpoints = np.linspace(self.times[0], self.times[-1], n)
        cx = poly_coeffs[:, 0]
        cy = poly_coeffs[:, 1]
        cz = poly_coeffs[:, 2]

        for i, t in enumerate(time_setpoints):
            seg = min(max(np.searchsorted(self.times, t) - 1, 0), len(cx) - 1)
            M = self.compute_poly_matrix(t - self.times[seg])
            kx = cx[seg*6:(seg+1)*6]
            ky = cy[seg*6:(seg+1)*6]
            kz = cz[seg*6:(seg+1)*6]
            x_vals[i] = M[0] @ kx; y_vals[i] = M[0] @ ky; z_vals[i] = M[0] @ kz
            v_x[i] = M[1] @ kx;    v_y[i] = M[1] @ ky;    v_z[i] = M[1] @ kz
            a_x[i] = M[2] @ kx;    a_y[i] = M[2] @ ky;    a_z[i] = M[2] @ kz

        yaw_vals = np.zeros((n, 1))
        trajectory_setpoints = np.hstack((x_vals, y_vals, z_vals, yaw_vals))

        vel_max = np.max(np.sqrt(v_x**2 + v_y**2 + v_z**2))
        acc_max = np.max(np.sqrt(a_x**2 + a_y**2 + a_z**2))
        print(f'  [PLAN] max speed {vel_max:.2f} m/s, max accel {acc_max:.2f} m/s²')
        assert vel_max <= self.vel_lim, f'planned velocity {vel_max:.2f} m/s exceeds limit'
        assert acc_max <= self.acc_lim, f'planned acceleration {acc_max:.2f} m/s² exceeds limit'
        return trajectory_setpoints, time_setpoints


class WaypointPlanner3D(MotionPlanner3D):
    """Minimum-jerk trajectory through given waypoints (no obstacles, no A*).
    Yaw is set to follow the velocity direction, like the simulation."""

    def __init__(self, waypoints, t_f, disc_steps=20, vel_lim=20.0, acc_lim=50.0):
        self._t_f = t_f
        self.disc_steps = disc_steps
        self.vel_lim = vel_lim
        self.acc_lim = acc_lim
        self.path = waypoints

        # Distribute total time t_f proportionally to segment distances
        pts = np.array(waypoints, dtype=float)[:, :3]
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = np.sum(dists)
        if total < 1e-6:
            self.times = np.linspace(0, t_f, len(waypoints))
        else:
            cum = np.concatenate(([0.0], np.cumsum(dists)))
            self.times = (cum / total) * t_f

        self.run_planner(waypoints)
        self._apply_yaw()

    def _apply_yaw(self):
        sp = self.trajectory_setpoints
        vx = np.gradient(sp[:, 0], self.time_setpoints)
        vy = np.gradient(sp[:, 1], self.time_setpoints)
        speed = np.sqrt(vx**2 + vy**2)
        yaw = np.arctan2(vy, vx)
        # Hold the last meaningful heading while nearly stationary
        thresh = 0.2
        last_good = yaw[np.argmax(speed > thresh)] if (speed > thresh).any() else 0.0
        for i in range(len(yaw)):
            if speed[i] > thresh:
                last_good = yaw[i]
            yaw[i] = last_good
        sp[:, 3] = np.unwrap(yaw)
        self.trajectory_setpoints = sp


# ── flight controller ──────────────────────────────────────────────────────────

class GateController:

    def __init__(self, cf: Crazyflie, cam: UdpVideoThread = None):
        self._cf  = cf
        self._cam = cam

        self.is_connected = False
        self._stop        = False
        self._state       = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}

        # Log config is set up after connection (called from main)
        lg = LogConfig(name='State', period_in_ms=100)
        lg.add_variable('stateEstimate.x', 'float')
        lg.add_variable('stateEstimate.y', 'float')
        lg.add_variable('stateEstimate.z', 'float')
        lg.add_variable('stabilizer.yaw',  'float')
        try:
            self._cf.log.add_config(lg)
            lg.data_received_cb.add_callback(self._log_cb)
            lg.start()
            self.is_connected = True
        except Exception as e:
            print(f'Log setup failed: {e}')

    def _log_cb(self, timestamp, data, logconf):
        self._state['x']   = data['stateEstimate.x']
        self._state['y']   = data['stateEstimate.y']
        self._state['z']   = data['stateEstimate.z']
        self._state['yaw'] = data['stabilizer.yaw']
        # Uncomment to debug:
        # r = math.sqrt(self._state['x']**2 + self._state['y']**2)
        # print(f"x={self._state['x']:.2f}  y={self._state['y']:.2f}  "
        #       f"z={self._state['z']:.2f}  yaw={self._state['yaw']:.1f}  r={r:.2f}")

    # ── boundary-safe hover ──────────────────────────────────────────────────

    def _safe_hover(self, vx=0.0, vy=0.0, yaw_rate=0.0, z=CRUISE_ALT):
        """
        Send hover setpoint, blocking outward velocity at the arena boundary.

        For each axis independently: if the drone is past the safety line and
        trying to move further into the wall, zero that velocity component.
        Tangential and inward motion are unaffected.
        """
        x   = self._state['x']
        y   = self._state['y']
        yaw = math.radians(self._state['yaw'])

        # Body → world
        wx = vx * math.cos(yaw) - vy * math.sin(yaw)
        wy = vx * math.sin(yaw) + vy * math.cos(yaw)

        # Hard cutoff per axis
        if x <= ARENA_X_MIN + SAFETY_MARGIN_HARD and wx < 0:
            wx = 0.0
        if x >= ARENA_X_MAX - SAFETY_MARGIN_HARD and wx > 0:
            wx = 0.0
        if y <= ARENA_Y_MIN + SAFETY_MARGIN_HARD and wy < 0:
            wy = 0.0
        if y >= ARENA_Y_MAX - SAFETY_MARGIN_HARD and wy > 0:
            wy = 0.0

        # World → body
        vx_s =  wx * math.cos(yaw) + wy * math.sin(yaw)
        vy_s = -wx * math.sin(yaw) + wy * math.cos(yaw)

        self._cf.commander.send_hover_setpoint(vx_s, vy_s, yaw_rate, z)

    def _stop_motors(self):
        self._cf.commander.send_stop_setpoint()

    # ── take-off / landing ───────────────────────────────────────────────────

    def takeoff(self, target_z=CRUISE_ALT):
        print(f'Taking off to {target_z:.2f} m')
        steps = int(TAKEOFF_DURATION / 0.1)
        for i in range(steps):
            self._safe_hover(z=target_z * (i / steps))
            time.sleep(0.1)
        for _ in range(20):
            self._safe_hover(z=target_z)
            time.sleep(0.1)

    def land(self):
        print('Landing')
        current_z = max(self._state['z'], CRUISE_ALT)
        steps = int(LAND_DURATION / 0.1)
        for i in range(steps):
            self._safe_hover(z=current_z * (1.0 - i / steps))
            time.sleep(0.1)
        self._stop_motors()

    # ── state: SEARCH ────────────────────────────────────────────────────────

    def search_for_gate(self):
        """
        Search logic: Continuous Sweep
        Oscillates the yaw back and forth to scan the immediate area.
        """
        print('  [SEARCH] Sweeping...')
        t_start = time.time()

        while not self._stop:
            # 1. Check the camera
            cx, cy, size = get_gate_detection(self._cam.latest_frame)
            if cx is not None:
                print(f'  [SEARCH] gate found  cx={cx:.0f}  cy={cy:.0f}  size={size:.0f}px')
                return

            # 2. Execute flight pattern
            elapsed = time.time() - t_start

            # Oscillate yaw rate like a pendulum to sweep the camera
            sweep_yaw_rate = math.sin(elapsed * math.pi / 3.0) * SEARCH_YAW_RATE
            self._safe_hover(yaw_rate=sweep_yaw_rate, z=CRUISE_ALT)

            time.sleep(0.05)

    # ── state: APPROACH ──────────────────────────────────────────────────────

    def approach_gate(self):
        """
        Image-Based Visual Servoing (IBVS).

        Three pixel errors drive three independent velocity commands:
          e_x = cx - cx_mid          → vy  (strafe left/right)
          e_y = cy_mid - cy          → Δz  (climb/descend)
          e_z = GATE_SIZE_CLOSE - sz → vx  (fly forward, never backward)

        Each error is multiplied by its Kp gain and clamped to a safe speed.
        The drone flies forward until the gate fills the frame (size ≥ GATE_SIZE_CLOSE).

        Returns True  when size ≥ GATE_SIZE_CLOSE  (→ TRANSIT).
        Returns False when gate lost > LOST_TOLERANCE (→ SEARCH).
        """
        print('  [APPROACH] IBVS toward gate')
        cx_mid   = CAM_WIDTH  / 2.0   # 162 px
        cy_mid   = CAM_HEIGHT / 2.0   # 122 px
        target_z = CRUISE_ALT
        lost     = 0

        while not self._stop:
            cx, cy, size = get_gate_detection(self._cam.latest_frame)

            if cx is None:
                lost += 1
                if lost > LOST_TOLERANCE:
                    print('  [APPROACH] gate lost — back to SEARCH')
                    return False
                # Hold last commanded altitude while waiting for gate to reappear
                self._safe_hover(z=target_z)
                time.sleep(0.05)
                continue

            lost = 0

            # ── Step 1: check termination ───────────────────────────────────
            if size >= GATE_SIZE_CLOSE:
                print(f'  [APPROACH] gate fills frame (size={size:.0f}px) → TRANSIT')
                return True

            # ── Step 2: pixel errors ────────────────────────────────────────
            e_x = cx - cx_mid          # +ve → gate right of centre
            e_y = cy_mid - cy          # +ve → gate above centre
            e_z = GATE_SIZE_CLOSE - size  # +ve → gate too small → move forward

            # ── Step 3: proportional control ────────────────────────────────
            v_strafe  = KP_VY * e_x
            v_climb   = KP_VZ * e_y
            v_forward = KP_VX * e_z

            # ── Step 4: clamp to safe speed limits ──────────────────────────
            v_strafe  = clamp(v_strafe,  -MAX_VY,  MAX_VY)
            v_climb   = clamp(v_climb,   -MAX_VZ_DELTA, MAX_VZ_DELTA)
            v_forward = clamp(v_forward,  0.0,     MAX_VX)   # never fly backward

            target_z  = clamp(CRUISE_ALT + v_climb,
                              CRUISE_ALT - MAX_VZ_DELTA,
                              CRUISE_ALT + MAX_VZ_DELTA)

            print(f'  [IBVS] ex={e_x:+.0f}px ey={e_y:+.0f}px size={size:.0f}px'
                  f'  → vx={v_forward:.2f} vy={v_strafe:+.2f} z={target_z:.2f}')

            # ── Step 5: send command ─────────────────────────────────────────
            self._safe_hover(vx=v_forward, vy=v_strafe, z=target_z)
            time.sleep(0.05)

        return False

    # ── state: TRANSIT ───────────────────────────────────────────────────────

    def transit_gate(self):
        """Push straight forward for TRANSIT_TIME to clear the gate."""
        print('  [TRANSIT] flying through gate')
        t_end = time.time() + TRANSIT_TIME
        while time.time() < t_end and not self._stop:
            self._safe_hover(vx=TRANSIT_VX, z=CRUISE_ALT)
            time.sleep(0.05)
        print('  [TRANSIT] done')

    # ── mission: Part 1 — vision ─────────────────────────────────────────────

    def fly_to_waypoint(self, wx, wy, wz=CRUISE_ALT, scan=True):
        """
        Fly toward the patrol waypoint (wx, wy) in the Lighthouse world frame
        using boundary-safe hover setpoints, yawing to face the direction of
        travel so the camera looks ahead. While `scan` is True the camera is
        checked every cycle.

        Returns:
          'found'   — a gate was detected (→ approach it)
          'reached' — waypoint reached without seeing a gate (→ fallback search)
          'stopped' — emergency stop requested
        """
        print(f'  [PATROL] flying to waypoint ({wx:.2f}, {wy:.2f})')
        while not self._stop:
            if scan:
                cx, cy, size = get_gate_detection(self._cam.latest_frame)
                if cx is not None:
                    print(f'  [PATROL] gate spotted  cx={cx:.0f}  size={size:.0f}px')
                    return 'found'

            ex = wx - self._state['x']
            ey = wy - self._state['y']
            dist = math.hypot(ex, ey)
            if dist < WAYPOINT_TOL:
                return 'reached'

            # World-frame velocity toward the waypoint, speed proportional to distance
            speed = clamp(WAYPOINT_KP * dist, WAYPOINT_SPEED_MIN, WAYPOINT_SPEED_MAX)
            wvx = speed * ex / dist
            wvy = speed * ey / dist

            # Rotate world → body: _safe_hover expects body-frame vx, vy
            yaw = math.radians(self._state['yaw'])
            bvx =  wvx * math.cos(yaw) + wvy * math.sin(yaw)
            bvy = -wvx * math.sin(yaw) + wvy * math.cos(yaw)

            # Yaw toward the direction of travel so the camera scans ahead
            desired_yaw = math.degrees(math.atan2(ey, ex))
            yaw_err = (desired_yaw - self._state['yaw'] + 180) % 360 - 180
            yaw_rate = clamp(WAYPOINT_YAW_KP * yaw_err,
                             -SEARCH_YAW_RATE, SEARCH_YAW_RATE)

            self._safe_hover(vx=bvx, vy=bvy, yaw_rate=yaw_rate, z=wz)
            time.sleep(0.05)

        return 'stopped'

    def run_vision_lap(self):
        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2)

        try:
            self.takeoff(target_z=1.0)

            # ── SIMPLE MISSION: search → approach → transit × N_GATES ──────────
            for gate_idx in range(N_GATES):
                if self._stop:
                    break
                
                print(f'\n=== Gate {gate_idx + 1} / {N_GATES} ===')

                # 1. Sweep yaw until the gate appears in frame
                self.search_for_gate()
                if self._stop:
                    break

                # 2. Servo toward gate; if lost mid-approach, search again
                while not self._stop:
                    if self.approach_gate():
                        self.transit_gate()
                        break
                    print('  gate lost — searching again')
                    self.search_for_gate()

            msg = 'All gates complete' if not self._stop else 'Emergency stop'
            print(f'\n{msg} — landing')
            # ──────────────────────────────────────────────────────────────────

            # ── TEST CODE (commented out) ──────────────────────────────────────
            # print('\n[TEST] Hovering at 1.0 m — calling get_gate_detection every tick')
            # os.makedirs('gate_frames', exist_ok=True)
            # frame_idx = 0
            # target_z  = 1.0
            # cx_mid    = CAM_WIDTH  / 2.0
            # cy_mid    = CAM_HEIGHT / 2.0
            # while not self._stop:
            #     frame = self._cam.latest_frame
            #     cx, cy, size = get_gate_detection(frame)
            #     if cx is not None:
            #         lat_err  = cx - cx_mid
            #         vy       = clamp(KP_VY * lat_err, -MAX_VY, MAX_VY)
            #         vert_err = cy_mid - cy
            #         target_z = clamp(1.0 + KP_VZ * vert_err,
            #                          1.0 - MAX_VZ_DELTA, 1.0 + MAX_VZ_DELTA)
            #         print(f'  [GATE DETECTED] cx={cx:.0f} cy={cy:.0f} size={size:.0f}px'
            #               f'  → vy={vy:+.3f} m/s  z={target_z:.2f} m')
            #         if frame is not None:
            #             path = os.path.join('gate_frames', f'gate_{frame_idx:04d}.png')
            #             cv2.imwrite(path, frame)
            #             print(f'  [SAVED] {path}')
            #             frame_idx += 1
            #         self._safe_hover(vy=vy, z=target_z)
            #     else:
            #         print('  [NO GATE] — holding position')
            #         self._safe_hover(z=target_z)
            #     time.sleep(0.1)
            # ──────────────────────────────────────────────────────────────────

        except Exception as e:
            print(f'\nUnhandled exception during mission: {e} — landing now')

        finally:
            # Always attempt a controlled landing, whatever happened above.
            # _stop_motors() is NOT called here — land() does a gradual descent
            # and only cuts motors at the end, so the drone doesn't just drop.
            try:
                self.land()
            except Exception as e:
                print(f'Land failed ({e}) — cutting motors')
                self._stop_motors()

    # =========================================================================
    # PART 2 — position-based mission (known gate coordinates)
    # =========================================================================

    def _clamp_to_boundary(self, x, y):
        """Clamp a target (x, y) to the rectangular arena minus the hard safety margin."""
        x_lo = ARENA_X_MIN + SAFETY_MARGIN_HARD
        x_hi = ARENA_X_MAX - SAFETY_MARGIN_HARD
        y_lo = ARENA_Y_MIN + SAFETY_MARGIN_HARD
        y_hi = ARENA_Y_MAX - SAFETY_MARGIN_HARD
        xc = clamp(x, x_lo, x_hi)
        yc = clamp(y, y_lo, y_hi)
        if (xc, yc) != (x, y):
            print(f'  [BOUND] target ({x:.2f}, {y:.2f}) clamped to ({xc:.2f}, {yc:.2f})')
        return xc, yc

    def _plan_lap(self, gates, start_xyz):
        """
        Build the one-lap waypoint list and fit a single minimum-jerk
        trajectory through it.

        Each gate is (x, y, z, theta, height, width):
        x, y, z  — gate centre in the Lighthouse world frame (metres)
        theta    — orientation of the gate's PLANE in the arena frame (radians).
                    theta=0 means the plane lies along world X, so the drone
                    flies through along world Y.
        height   — gate opening height (unused; z is already the centre)
        width    — gate opening width; used to cap the inward bias so we
                    never aim through the gate frame itself.
        """
        waypoints = [tuple(start_xyz)]
        prev_xy = (start_xyz[0], start_xyz[1])

        for g in gates:
            gx, gy, gz, theta, _gh, gw = g

            # Gate's plane direction, then flythrough direction (perpendicular).
            # Two candidate flythrough axes: ±(-sin θ, cos θ). Pick whichever
            # points AWAY from the previous waypoint so the drone keeps moving
            # forward through the course instead of doubling back.
            nx, ny = -math.sin(theta), math.cos(theta)
            if (gx - prev_xy[0]) * nx + (gy - prev_xy[1]) * ny < 0:
                nx, ny = -nx, -ny
            dirx, diry = nx, ny

            # Inward bias: shift pre/post waypoints toward the arena origin.
            # The bias direction is along the gate's PLANE (perpendicular to
            # flythrough), pointing toward origin. Cap by gate half-width so we
            # never aim outside the gate opening.
            r = math.hypot(gx, gy)
            if r > 1e-6 and GATE_INWARD_BIAS > 0:
                # Unit vector from gate toward origin
                ix_full, iy_full = -gx / r, -gy / r
                # Project onto the gate-plane axis (perpendicular to flythrough)
                plane_x, plane_y = -diry, dirx
                proj = ix_full * plane_x + iy_full * plane_y
                bias = clamp(GATE_INWARD_BIAS, 0.0, 0.45 * gw) * proj
                bx, by = bias * plane_x, bias * plane_y
            else:
                bx, by = 0.0, 0.0

            # Three colinear waypoints along the flythrough axis: before, centre, after.
            # Pre and post are nudged inward by (bx, by); the centre is not.
            px, py = self._clamp_to_boundary(gx - PRE_GATE_OFFSET * dirx + bx,
                                            gy - PRE_GATE_OFFSET * diry + by)
            cx, cy = self._clamp_to_boundary(gx, gy)
            qx, qy = self._clamp_to_boundary(gx + POST_GATE_OFFSET * dirx + bx,
                                            gy + POST_GATE_OFFSET * diry + by)
            waypoints.append((px, py, gz))
            waypoints.append((cx, cy, gz))
            waypoints.append((qx, qy, gz))

            prev_xy = (qx, qy)   # next gate's flythrough sign is chosen relative
                                # to where we just exited, not where we started.

        waypoints.append((start_xyz[0], start_xyz[1], CRUISE_ALT))

        return WaypointPlanner3D(waypoints, t_f=LAP_DURATION_S,
                                disc_steps=TRAJ_DISC_STEPS,
                                vel_lim=TRAJ_VEL_LIM, acc_lim=TRAJ_ACC_LIM)

    def run_fast_lap(self, gates, n_laps=N_LAPS):
        """
        Part 2: gate positions are known. For each lap, fit ONE minimum-jerk
        polynomial trajectory through all gates and follow it open-loop by streaming
        the time-indexed setpoints. The lap is re-planned from the current position
        each lap so drift from the previous lap does not accumulate. Best lap counts.
        """
        if not gates:
            print('GATE_POSITIONS is empty — fill it in before running Part 2.')
            return

        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2)

        dt = 1.0 / POSITION_RATE_HZ
        lap_times = []
        try:
            self.takeoff(CRUISE_ALT)

            for lap in range(n_laps):
                if self._stop:
                    break
                print(f'\n=== Lap {lap + 1} / {n_laps} ===')

                start_xyz = (self._state['x'], self._state['y'], self._state['z'])
                try:
                    planner = self._plan_lap(gates, start_xyz)
                except AssertionError as e:
                    print(f'  Trajectory infeasible ({e}). Increase LAP_DURATION_S.')
                    break

                setpoints = planner.trajectory_setpoints
                times = planner.time_setpoints

                # Follow the trajectory open-loop: index by elapsed time.
                t_lap = time.time()
                while not self._stop:
                    elapsed = time.time() - t_lap
                    if elapsed >= times[-1]:
                        break
                    n = min(int(np.searchsorted(times, elapsed)), len(setpoints) - 1)
                    sx, sy, sz, syaw = setpoints[n]
                    self._cf.commander.send_position_setpoint(
                        sx, sy, sz, math.degrees(syaw))
                    time.sleep(dt)

                lap_times.append(time.time() - t_lap)
                print(f'  Lap {lap + 1} time: {lap_times[-1]:.2f} s')

            print(f'\nLap times: {[f"{t:.2f}s" for t in lap_times]}')
            if lap_times:
                print(f'Best lap: {min(lap_times):.2f} s')

        except Exception as e:
            print(f'\nUnhandled exception during fast lap: {e} — landing now')

        finally:
            try:
                self.land()
            except Exception as e:
                print(f'Land failed ({e}) — cutting motors')
                self._stop_motors()


# ── emergency stop ─────────────────────────────────────────────────────────────

def emergency_stop_callback(cf):
    def on_press(key):
        try:
            if key.char == 'q':  # Check if the "space" key is pressed
                print("Emergency stop triggered!")
                cf.commander.send_stop_setpoint()  # Stop the Crazyflie
                cf.close_link()  # Close the link to the Crazyflie
                return False     # Stop the listener
        except AttributeError:
            pass

    # Start listening for key presses
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def emergency_stop_listener(ctrl: GateController, cam: UdpVideoThread, cf: Crazyflie):
    """Press 'q' for emergency stop.
    Attempts a controlled landing first; if that fails, cuts motors immediately.
    Press 'q' a second time to skip the landing and cut motors instantly.
    """
    stop_count = [0]

    def on_press(key):
        try:
            if key.char != 'q':
                return
        except AttributeError:
            return

        stop_count[0] += 1
        ctrl._stop = True

        if stop_count[0] == 1:
            print('\n[EMERGENCY STOP] landing — press Q again to cut motors immediately')
            try:
                ctrl.land()
            except Exception as e:
                print(f'Land failed ({e}) — cutting motors')
                ctrl._stop_motors()
            finally:
                if cam is not None:
                    cam.stop()
                cf.close_link()
            return False

        elif stop_count[0] >= 2:
            print('\n[EMERGENCY STOP] cutting motors immediately')
            ctrl._stop_motors()
            if cam is not None:
                cam.stop()
            cf.close_link()
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    assert MISSION in ('vision', 'position'), "MISSION must be 'vision' or 'position'"
    cflib.crtp.init_drivers()

    # Single Crazyflie instance over Crazyradio (control + logs)
    cf = Crazyflie(rw_cache='cache')
    connected_event = threading.Event()

    def on_connected(uri):
        print(f'Connected: {uri}')
        cf.supervisor.send_arming_request(True)
        connected_event.set()

    def on_connection_failed(uri, msg):
        print(f'Connection failed: {msg}')

    def on_disconnected(uri):
        print(f'Disconnected: {uri}')

    cf.connected.add_callback(on_connected)
    cf.connection_failed.add_callback(on_connection_failed)
    cf.disconnected.add_callback(on_disconnected)

    print(f'Connecting to {CONTROL_URI}')
    cf.open_link(CONTROL_URI)

    if not connected_event.wait(timeout=10):
        print('Connection timed out — exiting')
        exit(1)

    # The camera is only needed for the vision mission.
    cam = None
    if MISSION == 'vision':
        cam = UdpVideoThread()
        cam.start()
        print('Waiting for first camera frame...')
        while cam.latest_frame is None:
            time.sleep(0.05)
        print('Camera ready')

    # Build controller (sets up log variables)
    ctrl = GateController(cf, cam)
    if not ctrl.is_connected:
        print('Log setup failed — exiting')
        if cam is not None:
            cam.stop()
        cf.close_link()
        exit(1)

    # Emergency stop listener
    # threading.Thread(target=emergency_stop_listener,args=(ctrl, cam, cf), daemon=True).start()
    emergency_stop_thread = threading.Thread(target=emergency_stop_callback, args=(cf,))
    emergency_stop_thread.start()

    try:
        if MISSION == 'vision':
            ctrl.run_vision_lap()
        else:
            ctrl.run_fast_lap(GATE_POSITIONS, N_LAPS)
    finally:
        if cam is not None:
            cam.stop()
        cf.close_link()
