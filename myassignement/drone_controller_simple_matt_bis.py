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

Part 1 — triangulation-based gate finding:
    two camera views + drone pose estimate → triangulate gate centre and yaw,
    then fly approach / middle / exit waypoints through the gate.

Part 2 — raw waypoint streaming (Lighthouse world frame):
  Build one waypoint list per lap (start → pre-gate, gate centre, post-gate
  ×N → return). Stream each waypoint via send_position_setpoint and advance
  to the next once the drone is within WAYPOINT_REACH_TOL. The Crazyflie's
  onboard position controller smooths the motion between waypoints; no
  polynomial trajectory fit needed.

Boundary safety:
  stateEstimate.x/y from Lighthouse → enforce 2 m radius hard limit.

State machine per gate (Part 1):  SEARCH → APPROACH → TRANSIT → (next gate)

Press 'q' at any time for emergency stop.
"""

import contextlib
import logging
import math
import multiprocessing as mp
import os
import queue as queue_mod
import socket
import struct
import time
import threading
import warnings

import cv2
import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

from cv_detection import detect_gate
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

# ── FPV viewer ────────────────────────────────────────────────────────────────
# The viewer now runs in a SEPARATE PROCESS (mp.Process). It receives frames
# over a maxsize=1 queue — when the queue is full we drop, so the control
# loop never waits on the GUI. Because the viewer has its own GIL/interpreter,
# its render rate is no longer coupled to the recv thread; the rate is driven
# by the camera's natural fps (~3 fps).
FPV_ENABLED  = True   # show live camera window with detection overlay
FPV_SCALE    = 2      # upscale factor for the display window

CPX_HEADER_SIZE  = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE  = 11
MIN_JPEG_BYTES   = 1000

# ── arena bounds (Lighthouse world frame) ──────────────────────────────────────
ARENA_X_MIN = -1.0   # m — back wall
ARENA_X_MAX = +3.0   # m — front wall
ARENA_Y_MIN = -0.9   # m — right wall
ARENA_Y_MAX = +0.9   # m — left wall

SAFETY_MARGIN_HARD = 0.00000  # m — never fly closer than this to a wall

# ── flight ─────────────────────────────────────────────────────────────────────

CRUISE_ALT       = 1.25  # m ## The height position of the drone
TAKEOFF_DURATION = 3.0   # s
LAND_DURATION    = 3.0   # s

# ── triangulation mission (hardware_lap1 port) ────────────────────────────────

GATE_HEIGHT_REAL = 0.40   # m
GATE_HEIGHT_TOL  = 0.15   # m
GATE_DIST_MIN    = 0.30   # m
GATE_DIST_MAX    = 3.00   # m

LATERAL_DIST     = 0.50   # m — baseline between the two triangulation views
REQ_FRAMES       = 5      # fresh good detections needed at each view
SPEED_THRESHOLD  = 0.10   # m/s — used when stateEstimate velocity is available
APPROACH_DIST    = 0.60   # m
EXIT_DIST        = 0.60   # m
PASS_TOLERANCE   = 0.10   # m
LOST_THRESHOLD   = 20     # fresh no-gate detections before retrying

CIRCUIT_OFFSET       = 1.0    # m
EXPECTED_SECTORS     = [4, 2, 0, 10, 8]
SECTOR_TOLERANCE_DEG = 45.0

R_CAM_TO_BODY = np.array([[0, 0, 1],
                          [-1, 0, 0],
                          [0, -1, 0]], dtype=float)

FIRS_TRIANGULATION   = 'FIRS_TRIANGULATION'
FIRST_TRIANGULATION  = FIRS_TRIANGULATION
LATERAL_DRIFT        = 'LATERAL_DRIFT'
SECOND_TRIANGULATION = 'SECOND_TRIANGULATION'
COMPUTE_GATE_POS     = 'COMPUTE_GATE_POS'
TRAVEL_TO_GATE       = 'TRAVEL_TO_GATE'
DONE                 = 'DONE'

# IBVS gains — tune these on the real drone
# (Conservative: user said "slow is fine"; prefer reliable centering over speed.)
KP_VX        = 0.005  # size error  (GATE_SIZE_CLOSE - size) → forward speed
KP_VY        = 0.004  # lateral pixel error (cx - cx_mid)    → strafe speed
KP_VZ        = 0.004  # vertical pixel error (cy_mid - cy)   → altitude delta
MAX_VX       = 0.10   # m/s — forward cap (never backward)
MAX_VY       = 0.15   # m/s — strafe cap
MAX_VZ_DELTA = 0.4    # m   — altitude adjustment cap

# Alignment gating for APPROACH → TRANSIT handoff
ALIGN_TOL_X      = 25   # px — |cx - cx_mid| must be under this to allow TRANSIT
ALIGN_TOL_Y      = 25   # px — |cy_mid - cy| must be under this to allow TRANSIT
# Forward speed is scaled by how well-aligned the gate is in the frame. When
# pixel error fills this fraction of the frame, vx is throttled to zero —
# the drone strafes/climbs in place until the gate is roughly centred.
ALIGN_SCALE_DENOM = 0.6  # fraction of half-frame at which vx → 0

TRANSIT_VX   = 0.20   # m/s — slower than before; still open-loop forward push
TRANSIT_TIME = 1.8    # s

SEARCH_YAW_RATE = 12.0  # deg/s CCW
SEARCH_TIMEOUT  = 15.0  # s — double yaw rate after this

# After SEARCH spots a gate, hold position for this long while continuing to
# detect, then commit to APPROACH. Lock requires LOCK_MIN_HITS consecutive
# detections on UNIQUE frames to filter out single-frame flukes. Tuned for
# the AI-deck's real ~2–3 fps: 2 confirmed fresh detections in ~2.5 s.
LOCK_DURATION   = 1.0   # s — hover-and-confirm window
LOCK_MIN_HITS   = 1     # unique-frame detections required (non-consecutive)

# Lost-detection timeout is wall-clock based (not tick-based) so a slow
# camera doesn't make us bail prematurely. With ~3 fps, 1.5 s = ~4–5 frames.
LOST_TIMEOUT_S  = 1.5   # s — APPROACH gives up after this long without a fresh hit

# Exponential moving average on detections — smooths out single-frame noise.
# alpha = weight given to the newest sample. 1.0 = no smoothing.
DETECT_EMA_ALPHA = 0.5

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
GATE_HEIGHT = 0.4
NO_GATE_WAYPOINTS = []


# ── Part 2: position-based mission ─────────────────────────────────────────────
#
# Fill GATE_POSITIONS once the instructor gives you the exact gate positions.
# Each gate entry will be (x, y, z, theta, height, width)
GATE_POSITIONS = [    (0.65, -0.74, 1.28, np.deg2rad(58), 0.5, GATE_HEIGHT), (1.78, -0.92, 1.13, np.deg2rad(100), 0.29, GATE_HEIGHT),
    (2.22, 0.05, 1.42, np.deg2rad(188), 0.4, GATE_HEIGHT), (1.52, 0.83, 1.17, np.deg2rad(233), 0.4, GATE_HEIGHT),
    (0.51, 0.9, 1.28, np.deg2rad(280), 0.29, GATE_HEIGHT)]

N_LAPS           = 2     # number of timed laps
PRE_GATE_OFFSET  = 0.2   # m — waypoint placed before the gate along its approach axis
POST_GATE_OFFSET = 0.2   # m — waypoint placed after the gate (clears the frame)

POSITION_RATE_HZ   = 20.0  # setpoint streaming rate
WAYPOINT_REACH_TOL = 0.15  # m — final-waypoint reached tolerance
PURSUIT_LOOKAHEAD  = 0.35  # m — carrot distance ahead of drone along the path
                            #     larger = smoother + faster, smaller = tighter tracking


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
        self._frame_ts = 0.0

    def stop(self):
        # Kept for API compatibility; the daemon thread exits with the process.
        pass

    @property
    def latest_frame(self):
        with self._lock:
            return self._frame

    @property
    def latest_frame_with_ts(self):
        with self._lock:
            return self._frame, self._frame_ts

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            sock.bind(('0.0.0.0', UDP_LOCAL_PORT))
            sock.sendto(UDP_START_MAGIC, (UDP_AIDECK_IP, UDP_AIDECK_PORT))
            print(f'[UDP] bound :{UDP_LOCAL_PORT}, sent START to {UDP_AIDECK_IP}:{UDP_AIDECK_PORT}')
        except Exception as e:
            print(f'[UDP] socket setup failed: {e!r}')
            return

        buffer = bytearray()
        expected_size = 0
        receiving = False
        pkt_count = 0
        last_log = time.time()

        while True:
            data, _ = sock.recvfrom(2048)
            pkt_count += 1
            if pkt_count == 1:
                print(f'[UDP] first packet received ({len(data)} bytes)')
            now = time.time()
            if now - last_log > 2.0:
                print(f'[UDP] {pkt_count} packets received in last {now - last_log:.1f}s '
                      f'(receiving={receiving}, buf={len(buffer)}/{expected_size})')
                last_log = now
                pkt_count = 0
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
                self._decode_and_store(buffer)
                receiving = False

    def _decode_and_store(self, buffer):
        soi = buffer.find(b'\xff\xd8')
        eoi = buffer.rfind(b'\xff\xd9')
        if soi < 0 or eoi <= soi:
            return
        jpeg_len = eoi + 2 - soi
        if jpeg_len < MIN_JPEG_BYTES:
            return
        jpeg = np.frombuffer(buffer, np.uint8, count=jpeg_len, offset=soi)
        with _muted_stderr():
            img = cv2.imdecode(jpeg, cv2.IMREAD_UNCHANGED)
        if img is None:# or img.shape[:2] != (CAM_HEIGHT, CAM_WIDTH):
            return
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        with self._lock:
            self._frame = img
            self._frame_ts = time.time()


# ── FPV viewer (separate process) ────────────────────────────────────────────
#
# Runs in its OWN OS process via multiprocessing.Process so its cv2.imshow,
# Qt event loop, drawing, etc. all happen with a separate GIL and a separate
# interpreter. Nothing it does can stall the UDP recv thread or the control
# loop in the parent process. We communicate over a maxsize=1 queue and drop
# frames if the viewer is behind — the viewer is purely cosmetic.

def _fpv_viewer_process(q, scale, cam_w, cam_h, gate_size_close):
    """
    Child-process entry point. Receives (frame, detection, state) tuples and
    renders them with cv2.imshow. `detection` is (cx, cy, size) or None.
    `state` is a (x, y, z, yaw) tuple. A None message is the shutdown sentinel.
    """
    import cv2 as _cv2  # re-import inside child so spawn-based start works
    import numpy as _np

    win = 'Crazyflie FPV'
    _cv2.namedWindow(win, _cv2.WINDOW_NORMAL)
    _cv2.resizeWindow(win, cam_w * scale, cam_h * scale)
    while True:
        try:
            msg = q.get(timeout=0.5)
        except Exception:
            # No frame in 500 ms — still pump the GUI so the window is responsive
            if (_cv2.waitKey(1) & 0xFF) == ord('Q'):
                break
            continue
        if msg is None:
            break
        frame, det, state = msg
        disp = frame.copy() if frame.ndim == 3 else _cv2.cvtColor(frame, _cv2.COLOR_GRAY2BGR)

        # Crosshair
        _cv2.line(disp, (cam_w // 2, 0), (cam_w // 2, cam_h), (60, 60, 60), 1)
        _cv2.line(disp, (0, cam_h // 2), (cam_w, cam_h // 2), (60, 60, 60), 1)

        if det is not None:
            rcx, rcy, size = det
            color = (0, 255, 0) if size >= gate_size_close else (0, 200, 255)
            half = int(size / 2)
            _cv2.rectangle(disp,
                           (int(rcx) - half, int(rcy) - half),
                           (int(rcx) + half, int(rcy) + half),
                           color, 2)
            _cv2.drawMarker(disp, (int(rcx), int(rcy)), color,
                            _cv2.MARKER_CROSS, 14, 2)
            _cv2.putText(disp, f'cx={rcx:.0f} cy={rcy:.0f} size={size:.0f}',
                         (5, 18), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            _cv2.putText(disp, 'no gate', (5, 18),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if state is not None:
            sx, sy, sz, syaw = state
            _cv2.putText(disp,
                         f"x={sx:+.2f} y={sy:+.2f} z={sz:.2f} yaw={syaw:+.0f}",
                         (5, cam_h - 6), _cv2.FONT_HERSHEY_SIMPLEX,
                         0.42, (255, 255, 255), 1)

        if scale != 1:
            disp = _cv2.resize(disp, (cam_w * scale, cam_h * scale),
                               interpolation=_cv2.INTER_NEAREST)
        _cv2.imshow(win, disp)
        if (_cv2.waitKey(1) & 0xFF) == ord('Q'):
            break
    try:
        _cv2.destroyAllWindows()
    except Exception:
        pass


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


# ── flight controller ──────────────────────────────────────────────────────────

class GateController:

    def __init__(self, cf: Crazyflie, cam: UdpVideoThread = None):
        self._cf  = cf
        self._cam = cam

        self.is_connected = False
        self._stop        = False
        self._state       = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
        self._state['qx'] = 0.0
        self._state['qy'] = 0.0
        self._state['qz'] = 0.0
        self._state['qw'] = 1.0

        # Triangulation mission state.
        self._last_frame_ts = 0.0
        self._frames_detected = 0
        self._lost_count = 0
        self._travel_phase = 0
        self._gate_count = 0
        self._circuit_center = np.zeros(2)

        self._search_yaw_deg = 0.0
        self._search_yaw_total = 0.0

        self._P = None
        self._Q = None
        self._r_corners = np.zeros((4, 3))
        self._s_corners = np.zeros((4, 3))
        self._gate_yaw = 0.0
        self._gate_z = CRUISE_ALT
        self._app_x = self._app_y = 0.0
        self._mid_x = self._mid_y = 0.0
        self._exit_x = self._exit_y = 0.0
        self._lat_x = self._lat_y = self._lat_z = 0.0
        self._hold_x = self._hold_y = 0.0
        self._hold_z = CRUISE_ALT
        self._hold_yaw = 0.0

        # Fresh-vs-stale detection bookkeeping. The AI-deck only delivers
        # ~2–3 fps with notable latency, so the 20 Hz control loop must
        # distinguish a NEW frame from a repeat of the last one.
        self._last_frame_obj    = None  # identity of last processed frame
        self._ema_cx            = None  # smoothed detection
        self._ema_cy            = None
        self._ema_size          = None

        # Shared with FPV viewer (separate process). last_detection is
        # (cx, cy, size) of the most recent successful detection, or None.
        self.last_detection = None
        self.last_detection_miss = False

        # Optional multiprocessing.Queue (maxsize=1) the FPV process reads
        # from. Wire it in from main; we push (frame, detection, state) here
        # whenever we process a fresh frame, dropping silently if full.
        self._fpv_q = None

        self._log_ready = threading.Event()

        # Log config is set up after connection (called from main)
        lg1 = LogConfig(name='StatePosYaw', period_in_ms=100)
        lg1.add_variable('stateEstimate.x', 'float')
        lg1.add_variable('stateEstimate.y', 'float')
        lg1.add_variable('stateEstimate.z', 'float')
        lg1.add_variable('stabilizer.yaw',  'float')
        lg2 = LogConfig(name='StateQuat', period_in_ms=100)
        lg2.add_variable('stateEstimate.qx', 'float')
        lg2.add_variable('stateEstimate.qy', 'float')
        lg2.add_variable('stateEstimate.qz', 'float')
        lg2.add_variable('stateEstimate.qw', 'float')
        try:
            self._cf.log.add_config(lg1)
            lg1.data_received_cb.add_callback(self._log_cb)
            lg1.start()
            self._cf.log.add_config(lg2)
            lg2.data_received_cb.add_callback(self._log_cb)
            lg2.start()
            self.is_connected = True
        except Exception as e:
            print(f'Log setup failed: {e}')

    # ── detection polling (fresh vs stale frame) ──────────────────────────────
    #
    # The camera UDP stream tops out at ~2–3 fps. At a 20 Hz control loop, the
    # SAME frame would be processed 7–10× in a row — every "stale" hit looking
    # like a new detection. _poll_detection returns one of three statuses so
    # callers can react ONLY on fresh information.

    def _poll_detection(self):
        """
        Returns (cx, cy, size, status) where status ∈
          'new_hit'  — fresh frame, gate detected (cx,cy,size are EMA-smoothed)
          'new_miss' — fresh frame, no gate detected
          'stale'    — same frame object as last poll (no fresh info available)
        """
        frame = self._cam.latest_frame if self._cam is not None else None
        if frame is None or frame is self._last_frame_obj:
            return None, None, None, 'stale'
        self._last_frame_obj = frame

        cx, cy, size = get_gate_detection(frame)
        if cx is None:
            self.last_detection = None
            self.last_detection_miss = True
            self._push_fpv(frame, None)
            return None, None, None, 'new_miss'

        # EMA smoothing — damp single-frame outliers (one wild detection out
        # of every 3 frames is a 33% noise pulse without smoothing).
        a = DETECT_EMA_ALPHA
        if self._ema_cx is None:
            self._ema_cx, self._ema_cy, self._ema_size = cx, cy, size
        else:
            self._ema_cx   = a * cx   + (1 - a) * self._ema_cx
            self._ema_cy   = a * cy   + (1 - a) * self._ema_cy
            self._ema_size = a * size + (1 - a) * self._ema_size
        self.last_detection = (self._ema_cx, self._ema_cy, self._ema_size)
        self.last_detection_miss = False
        self._push_fpv(frame, self.last_detection)
        return self._ema_cx, self._ema_cy, self._ema_size, 'new_hit'

    def _push_fpv(self, frame, det):
        """Non-blocking send to the FPV viewer process. Drops if the queue is
        full so the control loop never waits on the GUI."""
        if self._fpv_q is None:
            return
        state = (self._state['x'], self._state['y'],
                 self._state['z'], self._state['yaw'])
        try:
            self._fpv_q.put_nowait((frame, det, state))
        except queue_mod.Full:
            pass

    def _current_state(self):
        return dict(self._state)

    def _send(self, x, y, z, yaw_deg):
        self._cf.commander.send_position_setpoint(x, y, z, yaw_deg)

    def _hold(self):
        self._send(self._hold_x, self._hold_y, self._hold_z, self._hold_yaw)

    def _detect(self):
        frame, ts = self._cam.latest_frame_with_ts if self._cam is not None else (None, 0.0)
        if frame is None or ts == self._last_frame_ts:
            return None
        self._last_frame_ts = ts

        result = detect_gate(frame)
        if result['status'] == 'ok':
            quad_norm = result['quad_norm']
            cx_norm = float(np.mean(quad_norm[:, 0]))
            return ('ok', quad_norm, cx_norm)
        if result['status'] == 'commit_to_pass':
            return ('commit_to_pass', None, 0.0)
        return ('no_gate', None, 0.0)

    def _corners_to_rays(self, quad_norm, s):
        """
        Convert undistorted normalized image corners to world-frame rays.

        Input:
        - quad_norm: (4,2) array of undistorted normalized image coordinates (xn, yn)
          where the camera pinhole model is used and z=1 in camera frame.
        - s: state dict with either quaternion (`qx,qy,qz,qw`) or yaw fallback.

        Geometry / frames:
        - v_cam = [xn, yn, 1] is the ray in the camera frame (pinhole model).
        - R_CAM_TO_BODY rotates camera-frame vectors into the vehicle body frame
          (accounts for camera mounting orientation).
        - R_b2w rotates body-frame vectors into the world frame using the
          vehicle attitude (quaternion or yaw fallback).

        Output: array (4,3) of world-frame direction vectors (not explicitly
        normalized here, but used downstream as directions for line intersections).
        """
        quat = np.array([s.get('qx', 0.0), s.get('qy', 0.0), s.get('qz', 0.0), s.get('qw', 1.0)], dtype=float)
        if np.linalg.norm(quat) < 1e-9:
            yaw = math.radians(s['yaw'])
            R_b2w = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                              [math.sin(yaw),  math.cos(yaw), 0.0],
                              [0.0, 0.0, 1.0]])
        else:
            R_b2w = R.from_quat(quat).as_matrix()

        # Build world rays for each detected corner
        rays = np.zeros((4, 3))
        for i, (xn, yn) in enumerate(quad_norm):
            # camera-frame direction (pinhole assumption): z=1
            v_cam = np.array([xn, yn, 1.0])
            # rotate: camera -> body -> world
            rays[i] = R_b2w @ (R_CAM_TO_BODY @ v_cam)
        return rays

    def _validate_detection(self, quad_norm, s):
        y_span = float(np.max(quad_norm[:, 1]) - np.min(quad_norm[:, 1]))
        if y_span < 1e-4:
            return None

        est_dist = GATE_HEIGHT_REAL / y_span
        if est_dist < GATE_DIST_MIN or est_dist > GATE_DIST_MAX:
            return None

        cx_norm = float(np.mean(quad_norm[:, 0]))
        bearing = math.radians(s['yaw']) - math.atan2(cx_norm, 1.0)
        est_x = s['x'] + est_dist * math.cos(bearing)
        est_y = s['y'] + est_dist * math.sin(bearing)

        angle_rad = math.atan2(est_y - self._circuit_center[1], est_x - self._circuit_center[0])
        angle_deg = math.degrees(angle_rad)
        clock_angle = (360 - angle_deg + 15) % 360
        sector = EXPECTED_SECTORS[self._gate_count % len(EXPECTED_SECTORS)]
        sector_centre = sector * 30 + 15
        angle_diff = abs(clock_angle - sector_centre)
        angle_diff = min(angle_diff, 360 - angle_diff)
        if angle_diff > SECTOR_TOLERANCE_DEG:
            return None

        return est_dist, est_x, est_y

    def _triangulate(self):
        """
        Triangulate 3D corner positions from two views.

        Algorithm summary:
        - Each corner defines two skew lines:
            L1(λ) = P + λ * r_i   (origin P, direction r_i)
            L2(μ) = Q + μ * s_i   (origin Q, direction s_i)
        - Solve for λ and μ that minimize the distance between points on the
          two lines (least-squares solve of a 3×2 linear system A [λ; μ] = Q-P
          where A = [r_i, -s_i]). This yields the closest points F and G.
        - Use the midpoint (F+G)/2 as the estimated 3D corner position.

        Notes on conditioning and checks:
        - If the two rays are nearly parallel the system is ill-conditioned and
          the least-squares result will be unreliable; calling code should
          detect inconsistent geometry (we check gate height below).
        - After computing 4 corners we estimate the gate height by averaging
          the vertical distances of left/right corner pairs and compare to
          the expected `GATE_HEIGHT_REAL` with tolerance `GATE_HEIGHT_TOL`.

        Returns: (corners_3d (4x3), H (3,), gate_yaw)
        """
        if self._P is None or self._Q is None:
            raise ValueError('missing triangulation views')

        corners_3d = np.zeros((4, 3))
        for i in range(4):
            # Build 3x2 matrix A = [r_i, -s_i] and solve A [lambda; mu] = Q-P
            A = np.column_stack([self._r_corners[i], -self._s_corners[i]])
            b = self._Q - self._P
            # least-squares handles non-exact intersections for skew lines
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            lmbda, mu = float(sol[0]), float(sol[1])
            F = self._P + lmbda * self._r_corners[i]
            G = self._Q + mu * self._s_corners[i]
            # corner is midpoint of closest-approach points
            corners_3d[i] = (F + G) / 2.0

        # Estimate gate vertical size from corner pairs and sanity-check height
        h_left = np.linalg.norm(corners_3d[3] - corners_3d[0])
        h_right = np.linalg.norm(corners_3d[2] - corners_3d[1])
        gate_height = (h_left + h_right) / 2.0
        if abs(gate_height - GATE_HEIGHT_REAL) > GATE_HEIGHT_TOL:
            raise ValueError(
                f'gate height {gate_height:.2f} m outside expected '
                f'{GATE_HEIGHT_REAL - GATE_HEIGHT_TOL:.2f}–{GATE_HEIGHT_REAL + GATE_HEIGHT_TOL:.2f} m'
            )

        # Gate center and orientation (yaw) in world frame
        H = np.mean(corners_3d, axis=0)
        v_width = corners_3d[1] - corners_3d[0]
        gate_yaw = math.atan2(-v_width[1], v_width[0])
        return corners_3d, H, gate_yaw

    def _set_gate_waypoints(self, H, gate_yaw, s):
        """
        Compute approach, mid (pass-through), and exit waypoints from gate pose.

        - Ensures gate yaw points from approach->exit (flip by pi if drone is
          currently 'in front' of the gate), then places approach point
          `APPROACH_DIST` meters before the gate center along gate normal and
          `EXIT_DIST` meters after the gate.
        - Sets `_gate_z` from the triangulated gate center height.
        """
        drone_2d = np.array([s['x'], s['y']])
        gate_2d = np.array([H[0], H[1]])
        to_drone = drone_2d - gate_2d
        forward = np.array([math.cos(gate_yaw), math.sin(gate_yaw)])

        # If drone lies on the 'forward' side of the gate, flip yaw so
        # forward points from approach toward exit (consistent waypoint order).
        if np.dot(to_drone, forward) >= 0:
            gate_yaw += math.pi

        self._gate_yaw = gate_yaw
        self._gate_z = H[2]

        fw = np.array([math.cos(gate_yaw), math.sin(gate_yaw)])
        self._app_x = H[0] - APPROACH_DIST * fw[0]
        self._app_y = H[1] - APPROACH_DIST * fw[1]
        self._mid_x = H[0]
        self._mid_y = H[1]
        self._exit_x = H[0] + EXIT_DIST * fw[0]
        self._exit_y = H[1] + EXIT_DIST * fw[1]
        self._travel_phase = 0

    def _force_pass(self, s):
        print('[COMPUTE_GATE_POS] gate fills frame — forcing straight pass')
        yaw_rad = math.radians(s['yaw'])
        fw = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
        self._gate_yaw = yaw_rad
        self._gate_z = s['z']
        self._app_x = s['x']
        self._app_y = s['y']
        self._mid_x = s['x'] + 0.4 * fw[0]
        self._mid_y = s['y'] + 0.4 * fw[1]
        self._exit_x = s['x'] + EXIT_DIST * fw[0]
        self._exit_y = s['y'] + EXIT_DIST * fw[1]
        self._travel_phase = 1

    def _reset_detection_filter(self):
        """Clear EMA state — call when entering a new visual-servo phase
        so we don't carry pixel state across gates / search → approach."""
        self._ema_cx = None
        self._ema_cy = None
        self._ema_size = None

    def _log_cb(self, timestamp, data, logconf):
        self._state['x']   = data['stateEstimate.x']
        self._state['y']   = data['stateEstimate.y']
        self._state['z']   = data['stateEstimate.z']
        if 'stabilizer.yaw' in data:
            self._state['yaw'] = data['stabilizer.yaw']
        if 'stateEstimate.qx' in data:
            self._state['qx'] = data['stateEstimate.qx']
        if 'stateEstimate.qy' in data:
            self._state['qy'] = data['stateEstimate.qy']
        if 'stateEstimate.qz' in data:
            self._state['qz'] = data['stateEstimate.qz']
        if 'stateEstimate.qw' in data:
            self._state['qw'] = data['stateEstimate.qw']
        self._log_ready.set()
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
        self._reset_detection_filter()
        t_start = time.time()
        tick = 0
        frames_seen = 0

        while not self._stop:
            # 1. Check the camera — react only to FRESH frames
            cx, cy, size, status = self._poll_detection()
            if status != 'stale':
                frames_seen += 1
            if status == 'new_hit':
                print(f'  [SEARCH] gate found  cx={cx:.0f}  cy={cy:.0f}  size={size:.0f}px '
                      f'after {time.time() - t_start:.1f}s, yaw={self._state["yaw"]:+.0f}deg '
                      f'(frames_seen={frames_seen})')
                return

            # 2. Execute flight pattern
            elapsed = time.time() - t_start

            # Oscillate yaw rate like a pendulum to sweep the camera
            sweep_yaw_rate = math.sin(elapsed * math.pi / 5) * SEARCH_YAW_RATE
            self._safe_hover(yaw_rate=sweep_yaw_rate, z=CRUISE_ALT)

            # Periodic heartbeat so we know search is alive (also reports
            # measured camera fps so we can confirm the AI-deck is healthy)
            tick += 1
            if tick % 10 == 0:  # every ~1 s at 0.1 s loop
                fps = frames_seen / max(elapsed, 0.001)
                print(f'  [SEARCH] tick={tick} t={elapsed:.1f}s fps={fps:.1f} '
                      f'yaw={self._state["yaw"]:+.0f}deg yaw_rate={sweep_yaw_rate:+.1f}deg/s '
                      f'pos=({self._state["x"]:+.2f},{self._state["y"]:+.2f},{self._state["z"]:.2f})')

            time.sleep(0.1)

    # ── state: LOCK ──────────────────────────────────────────────────────────

    def lock_on_gate(self):
        """
        After SEARCH first spots a gate, stop yawing and hover in place for
        LOCK_DURATION seconds, polling the detector each tick. Confirmation
        requires LOCK_MIN_HITS consecutive detections (filters single-frame
        flukes). Returns True if confirmed, else False (caller falls back to
        SEARCH).
        """
        print(f'  [LOCK] holding {LOCK_DURATION:.1f}s to confirm detection '
              f'(need {LOCK_MIN_HITS} total UNIQUE-FRAME hits, non-consecutive)')
        # Don't reset EMA here — keep refining the smoothed pose from search.
        t_end = time.time() + LOCK_DURATION
        hits = 0
        misses = 0
        stale = 0
        last = None
        tick = 0
        while time.time() < t_end and not self._stop:
            cx, cy, size, status = self._poll_detection()
            tick += 1
            if status == 'new_hit':
                hits += 1
                last = (cx, cy, size)
                # Early-exit as soon as we have enough hits (no need to wait out
                # the full window — the lock is "less strict" now).
                if hits >= LOCK_MIN_HITS:
                    break
            elif status == 'new_miss':
                misses += 1
            else:
                stale += 1
            self._safe_hover(z=CRUISE_ALT)
            time.sleep(0.1)

        unique = hits + misses
        if hits >= LOCK_MIN_HITS and last is not None:
            print(f'  [LOCK] CONFIRMED  hits={hits} unique_frames={unique} '
                  f'stale_ticks={stale} '
                  f'last(cx={last[0]:.0f},size={last[2]:.0f}px) → APPROACH')
            return True
        print(f'  [LOCK] FAILED  hits={hits} (<{LOCK_MIN_HITS}) '
              f'unique_frames={unique} stale_ticks={stale} → SEARCH')
        return False

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
        # Keep EMA from lock (already converged to the gate); just record start.
        cx_mid   = CAM_WIDTH  / 2.0   # 162 px
        cy_mid   = CAM_HEIGHT / 2.0   # 122 px
        target_z = CRUISE_ALT
        tick     = 0
        t_start  = time.time()
        t_last_hit = time.time()  # wall-clock timestamp of last fresh hit
        frames_seen = 0

        # Last commanded velocities — re-sent on stale ticks so the drone keeps
        # progressing between sparse camera frames instead of stuttering. They
        # are ONLY refreshed on a 'new_hit' tick (true closed-loop update).
        last_vx = 0.0
        last_vy = 0.0

        while not self._stop:
            cx, cy, size, status = self._poll_detection()
            tick += 1

            # ── Stale tick: re-send last command, do NOT recompute from old pixels.
            if status == 'stale':
                age = time.time() - t_last_hit
                if age > LOST_TIMEOUT_S:
                    print(f'  [APPROACH] LOST {age:.2f}s with no fresh frame → SEARCH '
                          f'(approach lasted {time.time() - t_start:.1f}s, '
                          f'fresh_frames={frames_seen})')
                    return False
                self._safe_hover(vx=last_vx, vy=last_vy, z=target_z)
                time.sleep(0.1)
                continue

            # ── Fresh frame received (hit or miss).
            frames_seen += 1

            if status == 'new_miss':
                age = time.time() - t_last_hit
                if age > LOST_TIMEOUT_S:
                    print(f'  [APPROACH] gate lost {age:.2f}s → SEARCH '
                          f'(approach lasted {time.time() - t_start:.1f}s, '
                          f'fresh_frames={frames_seen})')
                    return False
                # Coast on the last good command but never accelerate from stale data
                self._safe_hover(vx=last_vx, vy=last_vy, z=target_z)
                time.sleep(0.1)
                continue

            # status == 'new_hit'
            t_last_hit = time.time()

            # ── Step 1: pixel errors ────────────────────────────────────────
            e_x = -cx + cx_mid          # +ve → gate left of centre (strafe left)
            e_y = cy_mid - cy           # +ve → gate above centre
            e_z = GATE_SIZE_CLOSE - size  # +ve → gate too small → move forward

            # ── Step 2: check termination — only TRANSIT if also aligned ────
            size_ok  = size >= GATE_SIZE_CLOSE
            align_ok = abs(e_x) < ALIGN_TOL_X and abs(e_y) < ALIGN_TOL_Y
            if size_ok and align_ok:
                print(f'  [APPROACH] DONE  size={size:.0f}px>={GATE_SIZE_CLOSE} '
                      f'ex={e_x:+.0f}<{ALIGN_TOL_X} ey={e_y:+.0f}<{ALIGN_TOL_Y} '
                      f'after {time.time() - t_start:.1f}s → TRANSIT')
                return True

            # ── Step 3: proportional control ────────────────────────────────
            v_strafe  = KP_VY * e_x
            v_climb   = KP_VZ * e_y
            v_forward = KP_VX * e_z

            # ── Step 4: clamp to safe speed limits ──────────────────────────
            v_strafe  = clamp(v_strafe,  -MAX_VY,  MAX_VY)
            v_climb   = clamp(v_climb,   -MAX_VZ_DELTA, MAX_VZ_DELTA)
            v_forward = clamp(v_forward,  0.0,     MAX_VX)   # never fly backward

            # ── Step 4b: throttle forward speed by alignment ────────────────
            # When the gate is off-centre, slow forward motion so the drone
            # has time to strafe/climb into alignment instead of barrelling
            # diagonally toward the gate plane.
            mis_x = abs(e_x) / cx_mid   # 0 = centred, 1 = at frame edge
            mis_y = abs(e_y) / cy_mid
            align_factor = clamp(1.0 - (mis_x + mis_y) / ALIGN_SCALE_DENOM, 0.0, 1.0)
            v_forward *= align_factor

            target_z  = clamp(CRUISE_ALT + v_climb,
                              CRUISE_ALT - MAX_VZ_DELTA,
                              CRUISE_ALT + MAX_VZ_DELTA)

            # Throttle to ~1 Hz so console I/O doesn't starve UDP recv thread
            if tick % 10 == 0:
                sx = 'OK' if size_ok else '--'
                ax = 'OK' if align_ok else '--'
                fps = frames_seen / max(time.time() - t_start, 0.001)
                print(f'  [IBVS] t={tick:4d} f={frames_seen} fps={fps:.1f}  '
                      f'cx={cx:6.1f} cy={cy:6.1f} sz={size:5.1f}px '
                      f'[{sx}|{ax}]  ex={e_x:+5.0f} ey={e_y:+5.0f} ez={e_z:+5.0f}  '
                      f'align={align_factor:.2f}  vx={v_forward:.3f} vy={v_strafe:+.3f} '
                      f'z={target_z:.2f}  pos=({self._state["x"]:+.2f},{self._state["y"]:+.2f})')

            # ── Step 5: cache + send command ─────────────────────────────────
            # Cache for stale-tick re-send (no fresh frame → coast on last cmd)
            last_vx, last_vy = v_forward, v_strafe
            self._safe_hover(vx=v_forward, vy=v_strafe, z=target_z)
            time.sleep(0.1)

        return False

    # ── state: TRANSIT ───────────────────────────────────────────────────────

    def transit_gate(self):
        """
        Push forward to clear the gate. For as long as the detector still sees
        the gate (i.e. we haven't passed the plane yet) keep correcting lateral
        and vertical pixel error so a late drift doesn't clip the frame. Once
        the gate disappears from view, finish the push open-loop.
        """
        print(f'  [TRANSIT] flying through gate (vx={TRANSIT_VX} m/s for {TRANSIT_TIME}s)')
        cx_mid = CAM_WIDTH  / 2.0
        cy_mid = CAM_HEIGHT / 2.0
        target_z = CRUISE_ALT
        t_end = time.time() + TRANSIT_TIME
        tick = 0
        seen = 0
        miss = 0
        stale = 0
        last_vy = 0.0
        while time.time() < t_end and not self._stop:
            cx, cy, _size, status = self._poll_detection()
            tick += 1
            if status == 'new_hit':
                seen += 1
                e_x = -cx + cx_mid
                e_y = cy_mid - cy
                v_strafe = clamp(KP_VY * e_x, -MAX_VY, MAX_VY)
                v_climb  = clamp(KP_VZ * e_y, -MAX_VZ_DELTA, MAX_VZ_DELTA)
                target_z = clamp(CRUISE_ALT + v_climb,
                                 CRUISE_ALT - MAX_VZ_DELTA,
                                 CRUISE_ALT + MAX_VZ_DELTA)
                last_vy = v_strafe
                self._safe_hover(vx=TRANSIT_VX, vy=v_strafe, z=target_z)
            elif status == 'new_miss':
                miss += 1
                # Gate no longer visible → past the plane, drive straight forward
                self._safe_hover(vx=TRANSIT_VX, z=target_z)
            else:  # stale — keep coasting on last command
                stale += 1
                self._safe_hover(vx=TRANSIT_VX, vy=last_vy, z=target_z)
            time.sleep(0.1)
        print(f'  [TRANSIT] done (seen={seen} miss={miss} stale={stale} ticks, '
              f'pos=({self._state["x"]:+.2f},{self._state["y"]:+.2f},{self._state["z"]:.2f}))')

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
            time.sleep(0.1)

        return 'stopped'

    def run_vision_lap(self):
        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2)

        if not self._log_ready.wait(timeout=5.0):
            print('No state estimate received — aborting vision mission')
            return

        # ── background frame saver ─────────────────────────────────────────────
        os.makedirs('gate_frames', exist_ok=True)
        frame_idx    = [0]
        last_frame   = [None]
        save_running = [False]  # DISABLED for frame-throughput test

        def _frame_saver():
            while save_running[0]:
                frame = self._cam.latest_frame
                if frame is not None and frame is not last_frame[0]:
                    path = os.path.join('gate_frames', f'frame_{frame_idx[0]:05d}.png')
                    cv2.imwrite(path, frame)
                    last_frame[0]  = frame
                    frame_idx[0]  += 1
                time.sleep(0.05)   # ~20 fps

        # threading.Thread(target=_frame_saver, daemon=True, name='FrameSaver').start()
        print(f'[FRAME SAVER] DISABLED')
        # ──────────────────────────────────────────────────────────────────────

        try:
            s = self._current_state()
            start_x = s['x']
            start_y = s['y']
            start_yaw = s['yaw']
            self._hold_x = start_x
            self._hold_y = start_y
            self._hold_z = CRUISE_ALT
            self._hold_yaw = start_yaw
            self._search_yaw_deg = start_yaw

            yaw_r = math.radians(start_yaw)
            self._circuit_center = np.array([
                start_x + CIRCUIT_OFFSET * math.cos(yaw_r),
                start_y + CIRCUIT_OFFSET * math.sin(yaw_r),
            ])
            print(f'Start: x={start_x:.2f}  y={start_y:.2f}  yaw={start_yaw:.1f}°')
            print(f'Circuit centre: ({self._circuit_center[0]:.2f}, {self._circuit_center[1]:.2f})')

            self.takeoff(target_z=CRUISE_ALT)

            mission_complete = False
            for gate_idx in range(N_GATES):
                if self._stop:
                    break

                print(f'\n=== Gate {gate_idx + 1} / {N_GATES} ===')
                self._frames_detected = 0
                self._lost_count = 0
                mission_state = FIRST_TRIANGULATION
                print(f'[{mission_state}]')

                while not self._stop:
                    s = self._current_state()
                    det = self._detect()

                    if mission_state == FIRST_TRIANGULATION:
                        self._hold_x = s['x']
                        self._hold_y = s['y']
                        self._hold_z = CRUISE_ALT
                        self._hold_yaw = self._search_yaw_deg
                        self._search_yaw_deg += SEARCH_YAW_RATE * 0.1
                        self._search_yaw_total += SEARCH_YAW_RATE * 0.1
                        self._hold()

                        if det is not None:
                            status, quad_norm, cx_norm = det
                            if status == 'commit_to_pass':
                                self._force_pass(s)
                                mission_state = TRAVEL_TO_GATE
                                print(f'[{mission_state}] from FIRST_TRIANGULATION commit')
                            elif status == 'ok' and self._validate_detection(quad_norm, s) is not None:
                                gate_angle = math.atan2(cx_norm, 1.0)
                                self._hold_yaw = math.degrees(math.radians(s['yaw']) - gate_angle)
                                self._frames_detected += 1
                                self._lost_count = 0
                                if self._frames_detected >= REQ_FRAMES:
                                    self._P = np.array([s['x'], s['y'], s['z']])
                                    self._r_corners = self._corners_to_rays(quad_norm, s)
                                    self._frames_detected = 0
                                    yaw_r = math.radians(s['yaw'])
                                    self._lat_x = s['x'] + LATERAL_DIST * math.sin(yaw_r)
                                    self._lat_y = s['y'] + LATERAL_DIST * -math.cos(yaw_r)
                                    self._lat_z = CRUISE_ALT
                                    mission_state = LATERAL_DRIFT
                                    print(f'[{mission_state}] to ({self._lat_x:.2f},{self._lat_y:.2f})')
                            else:
                                self._frames_detected = 0
                                self._lost_count += 1
                                if self._lost_count >= LOST_THRESHOLD:
                                    print(f'[{FIRST_TRIANGULATION}] gate lost, retrying')
                                    self._lost_count = 0

                    elif mission_state == LATERAL_DRIFT:
                        self._send(self._lat_x, self._lat_y, self._lat_z, self._hold_yaw)
                        if self._dist3(s, self._lat_x, self._lat_y, self._lat_z) < PASS_TOLERANCE:
                            self._hold_x = self._lat_x
                            self._hold_y = self._lat_y
                            self._frames_detected = 0
                            self._lost_count = 0
                            mission_state = SECOND_TRIANGULATION
                            print(f'[{mission_state}]')

                    elif mission_state == SECOND_TRIANGULATION:
                        self._hold()
                        if det is not None:
                            status, quad_norm, cx_norm = det
                            if status == 'commit_to_pass':
                                self._force_pass(s)
                                mission_state = TRAVEL_TO_GATE
                                print(f'[{mission_state}] from SECOND_TRIANGULATION commit')
                            elif status == 'ok' and self._validate_detection(quad_norm, s) is not None:
                                gate_angle = math.atan2(cx_norm, 1.0)
                                self._hold_yaw = math.degrees(math.radians(s['yaw']) - gate_angle)
                                self._frames_detected += 1
                                self._lost_count = 0
                                if self._frames_detected >= REQ_FRAMES:
                                    self._Q = np.array([s['x'], s['y'], s['z']])
                                    self._s_corners = self._corners_to_rays(quad_norm, s)
                                    self._frames_detected = 0
                                    mission_state = COMPUTE_GATE_POS
                                    print(f'[{mission_state}]')
                            else:
                                self._frames_detected = 0
                                self._lost_count += 1
                                if self._lost_count >= LOST_THRESHOLD:
                                    print(f'[{FIRST_TRIANGULATION}] gate lost during SECOND_TRIANGULATION')
                                    self._lost_count = 0
                                    mission_state = FIRST_TRIANGULATION

                    elif mission_state == COMPUTE_GATE_POS:
                        self._hold()
                        try:
                            _, H, gate_yaw = self._triangulate()
                            print(f'Gate {self._gate_count + 1}: centre=({H[0]:.2f},{H[1]:.2f},{H[2]:.2f}) '
                                  f'yaw={math.degrees(gate_yaw):.1f}°')
                            self._set_gate_waypoints(H, gate_yaw, s)
                            mission_state = TRAVEL_TO_GATE
                            print(f'[{mission_state}]')
                        except Exception as e:
                            print(f'Triangulation failed ({e}) — retrying')
                            self._frames_detected = 0
                            self._lost_count = 0
                            mission_state = FIRST_TRIANGULATION

                    elif mission_state == TRAVEL_TO_GATE:
                        gyd = math.degrees(self._gate_yaw)
                        if self._travel_phase == 0:
                            self._send(self._app_x, self._app_y, self._gate_z, gyd)
                            if self._dist3(s, self._app_x, self._app_y, self._gate_z) < PASS_TOLERANCE:
                                self._travel_phase = 1
                        elif self._travel_phase == 1:
                            self._send(self._mid_x, self._mid_y, self._gate_z, gyd)
                            if self._dist3(s, self._mid_x, self._mid_y, self._gate_z) < PASS_TOLERANCE:
                                self._travel_phase = 2
                        elif self._travel_phase == 2:
                            self._send(self._exit_x, self._exit_y, self._gate_z, gyd)
                            if self._dist3(s, self._exit_x, self._exit_y, self._gate_z) < PASS_TOLERANCE:
                                self._gate_count += 1
                                print(f'Gate {self._gate_count} passed!')
                                self._hold_x = s['x']
                                self._hold_y = s['y']
                                self._hold_yaw = s['yaw']
                                self._search_yaw_deg = s['yaw']
                                self._search_yaw_total = 0.0
                                if self._gate_count >= N_GATES:
                                    print(f'[{DONE}] all {N_GATES} gates passed — hovering')
                                    mission_complete = True
                                    break
                                else:
                                    mission_state = FIRST_TRIANGULATION
                                    print(f'[{mission_state}] ({self._gate_count}/{N_GATES} done)')

                    elif mission_state == DONE:
                        self._hold()

                    time.sleep(0.1)

                if mission_complete:
                    break

            msg = 'All gates complete' if mission_complete and not self._stop else 'Emergency stop'
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
            save_running[0] = False
            print(f'[FRAME SAVER] stopped — {frame_idx[0]} frames saved to gate_frames/')
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
        Build the one-lap waypoint list: start → (pre, centre, post) per gate
        → return to start. Yaw is set to face the flythrough direction at each
        gate. Returns a list of (x, y, z, yaw_rad) tuples.

        Each gate is (x, y, z, theta, height, width). theta is the orientation
        of the gate's PLANE; the flythrough axis is perpendicular to it.
        """
        waypoints = [(start_xyz[0], start_xyz[1], start_xyz[2], 0.0)]
        prev_xy = (start_xyz[0], start_xyz[1])

        for g in gates:
            gx, gy, gz, theta, _gw, _gh = g

            # Flythrough direction; flip if it points back toward where we came from.
            dirx, diry = -math.sin(theta), math.cos(theta)
            if (gx - prev_xy[0]) * dirx + (gy - prev_xy[1]) * diry < 0:
                dirx, diry = -dirx, -diry
            yaw = math.atan2(diry, dirx)

            px, py = self._clamp_to_boundary(gx - PRE_GATE_OFFSET * dirx,
                                            gy - PRE_GATE_OFFSET * diry)
            cx, cy = self._clamp_to_boundary(gx, gy)
            qx, qy = self._clamp_to_boundary(gx + POST_GATE_OFFSET * dirx,
                                            gy + POST_GATE_OFFSET * diry)
            waypoints.append((px, py, gz, yaw))
            waypoints.append((cx, cy, gz, yaw))
            waypoints.append((qx, qy, gz, yaw))

            prev_xy = (qx, qy)

        waypoints.append((start_xyz[0], start_xyz[1], CRUISE_ALT, 0.0))
        return waypoints

    def _follow_path_pure_pursuit(self, waypoints, dt):
        """
        Stream a moving carrot point along the polyline `waypoints`
        (list of (x, y, z, yaw_rad)). At each tick we find the drone's closest
        point on the path, then walk forward along segments by PURSUIT_LOOKAHEAD
        metres to get the carrot, and send that as the position setpoint.

        Yaw at the carrot is interpolated from the surrounding segment's yaws.
        The path is considered finished when the projection reaches the final
        segment and the drone is within WAYPOINT_REACH_TOL of the last point.
        """
        if len(waypoints) < 2:
            return

        pts = [np.array([w[0], w[1], w[2]]) for w in waypoints]
        yaws = [w[3] for w in waypoints]
        seg_idx = 0  # current segment is pts[seg_idx] → pts[seg_idx+1]

        while not self._stop:
            drone = np.array([self._state['x'], self._state['y'], self._state['z']])

            # Project drone onto current segment; if past its end, advance.
            while seg_idx < len(pts) - 2:
                a, b = pts[seg_idx], pts[seg_idx + 1]
                ab = b - a
                ab_len2 = float(ab @ ab)
                if ab_len2 < 1e-9:
                    seg_idx += 1
                    continue
                t = float((drone - a) @ ab) / ab_len2
                if t >= 1.0:
                    seg_idx += 1
                else:
                    break

            # Termination: on the last segment and near the final waypoint.
            if seg_idx >= len(pts) - 1:
                break
            if seg_idx == len(pts) - 2:
                if np.linalg.norm(drone - pts[-1]) < WAYPOINT_REACH_TOL:
                    break

            # Carrot: start from projection on current segment, then walk
            # forward along the polyline by PURSUIT_LOOKAHEAD metres.
            a, b = pts[seg_idx], pts[seg_idx + 1]
            ab = b - a
            ab_len = float(np.linalg.norm(ab))
            if ab_len < 1e-9:
                t = 0.0
            else:
                t = clamp(float((drone - a) @ ab) / (ab_len * ab_len), 0.0, 1.0)

            remaining = PURSUIT_LOOKAHEAD
            cur_seg = seg_idx
            cur_t = t
            carrot = a + t * ab
            carrot_yaw = yaws[cur_seg + 1]
            while remaining > 0 and cur_seg < len(pts) - 1:
                sa, sb = pts[cur_seg], pts[cur_seg + 1]
                seg_vec = sb - sa
                seg_len = float(np.linalg.norm(seg_vec))
                left = seg_len * (1.0 - cur_t)
                if left >= remaining:
                    cur_t += remaining / seg_len if seg_len > 1e-9 else 0.0
                    carrot = sa + cur_t * seg_vec
                    carrot_yaw = yaws[cur_seg + 1]
                    remaining = 0
                else:
                    remaining -= left
                    cur_seg += 1
                    cur_t = 0.0
                    if cur_seg >= len(pts) - 1:
                        carrot = pts[-1]
                        carrot_yaw = yaws[-1]
                        break

            self._cf.commander.send_position_setpoint(
                float(carrot[0]), float(carrot[1]), float(carrot[2]),
                math.degrees(carrot_yaw))
            time.sleep(dt)

    def run_fast_lap(self, gates, n_laps=N_LAPS):
        """
        Part 2: gate positions are known. Stream raw (pre, centre, post)
        waypoints per gate to the Crazyflie's onboard position controller,
        advancing once the drone is within WAYPOINT_REACH_TOL of each.
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
                waypoints = self._plan_lap(gates, start_xyz)

                t_lap = time.time()
                self._follow_path_pure_pursuit(waypoints, dt)
                if self._stop:
                    break

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
    #stop_count = [0]

    def on_press(key):
        # try:
        #     if key.char != 'q':
        #         return
        # except AttributeError:
        #     return

        # stop_count[0] += 1
        # ctrl._stop = True

        # if stop_count[0] == 1:
        #     print('\n[EMERGENCY STOP] landing — press Q again to cut motors immediately')
        #     try:
        #         ctrl.land()
        #     except Exception as e:
        #         print(f'Land failed ({e}) — cutting motors')
        #         ctrl._stop_motors()
        #     finally:
        #         if cam is not None:
        #             cam.stop()
        #         cf.close_link()
        #     return False

        # elif stop_count[0] >= 2:
        #     print('\n[EMERGENCY STOP] cutting motors immediately')
        #     ctrl._stop_motors()
        #     if cam is not None:
        #         cam.stop()
        #     cf.close_link()
        #     return False
        # ESC → controlled landing
        if key == keyboard.Key.esc:
            print('\n[EMERGENCY STOP] landing')

            ctrl._stop = True

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

        # Q → immediate kill
        try:
            if key.char == 'q':
                print('\n[EMERGENCY STOP] cutting motors immediately')

                ctrl._stop = True
                ctrl._stop_motors()

                if cam is not None:
                    cam.stop()

                cf.close_link()

                return False
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    assert MISSION in ('vision', 'position'), "MISSION must be 'vision' or 'position'"
    # Start the camera thread FIRST (matches FPV example ordering) so the
    # AI-deck receives START_MAGIC and begins streaming while the Crazyradio
    # link is still being established.
    cam = None
    if MISSION == 'vision':
        cam = UdpVideoThread()
        cam.start()

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

    if cam is not None:
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

    # Live FPV window — runs in a SEPARATE PROCESS so its GUI work cannot
    # GIL-stall the UDP recv thread or the control loop.
    fpv_proc = None
    fpv_q = None
    if FPV_ENABLED and cam is not None:
        fpv_q = mp.Queue(maxsize=1)   # latest-wins; controller drops if full
        ctrl._fpv_q = fpv_q
        fpv_proc = mp.Process(
            target=_fpv_viewer_process,
            args=(fpv_q, FPV_SCALE, CAM_WIDTH, CAM_HEIGHT, GATE_SIZE_CLOSE),
            daemon=True, name='FpvViewerProc')
        fpv_proc.start()
        print(f'FPV viewer process started (pid={fpv_proc.pid})')

    # Emergency stop listener
    emergency_stop_thread = threading.Thread(target=emergency_stop_listener, args=(ctrl, cam, cf), daemon=True)
    emergency_stop_thread.start()

    try:
        if MISSION == 'vision':
            ctrl.run_vision_lap()
        else:
            ctrl.run_fast_lap(GATE_POSITIONS, N_LAPS)
    finally:
        if fpv_q is not None:
            try:
                fpv_q.put_nowait(None)  # graceful shutdown sentinel
            except Exception:
                pass
        if fpv_proc is not None:
            fpv_proc.join(timeout=1.0)
        if cam is not None:
            cam.stop()
        cf.close_link()
