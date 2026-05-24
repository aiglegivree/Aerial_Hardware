# -*- coding: utf-8 -*-
"""
drone_controller_simple_matt_triangulation.py
=============================================

Same scaffolding as drone_controller_simple_matt.py (UDP video thread,
in-thread FPV viewer, ESC/Q emergency stop, entry point), but the per-gate
flow is replaced with the 2-view triangulation pipeline from
hardware_lap1_simple.py:

    TAKEOFF → SEARCH → DETECT_1 → LATERAL_MOVE → DETECT_2
            → TRIANGULATE → TRAVEL_GATE  → (loop / DONE) → LAND

Detection comes from cv_detection.detect_gate (4 corners + normalised quad),
which lets us recover the gate's full pose (centre + yaw of its plane) via a
4-corner least-squares ray intersection.
"""

import contextlib
import logging
import math
import os
import socket
import struct
import threading
import time
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

warnings.filterwarnings("ignore", message=".*supervisor subsystem requires CRTP.*")
logging.basicConfig(level=logging.ERROR)

# ── connection ─────────────────────────────────────────────────────────────────

CONTROL_URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E708')

UDP_AIDECK_IP   = '192.168.4.1'
UDP_AIDECK_PORT = 5000
UDP_LOCAL_PORT  = 5001
UDP_START_MAGIC = b'FER'

# ── camera ─────────────────────────────────────────────────────────────────────

CAM_WIDTH  = 324
CAM_HEIGHT = 244

# ── FPV viewer ────────────────────────────────────────────────────────────────

FPV_ENABLED  = True
FPV_SCALE    = 2
FPV_RATE_HZ  = 10

CPX_HEADER_SIZE  = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE  = 11
MIN_JPEG_BYTES   = 1000

# ── arena bounds (Lighthouse world frame) ──────────────────────────────────────

ARENA_X_MIN = -1.0
ARENA_X_MAX = +3.0
ARENA_Y_MIN = -0.9
ARENA_Y_MAX = +0.9
SAFETY_MARGIN_HARD = 0.0

# ── flight ─────────────────────────────────────────────────────────────────────

CRUISE_ALT       = 1.2
TAKEOFF_DURATION = 4.0
LAND_DURATION    = 3.0
SETPOINT_PERIOD  = 0.05      # 20 Hz

SEARCH_YAW_RATE  = 10.0      # deg/s, CCW target-yaw increment
LATERAL_DIST     = 0.5       # m, sideways baseline between views (direction auto-picked)
LATERAL_STEP     = 0.07      # m, max carrot distance per setpoint during lateral move (smaller = slower)
LATERAL_SETTLE_S = 1.0       # s, hold at lateral target before starting DETECT_2
REQ_FRAMES       = 5         # consecutive stationary hits to commit a view
SPEED_THRESHOLD  = 0.10      # m/s
APPROACH_DIST    = 0.4       # m, pre-gate waypoint offset
EXIT_DIST        = 0.4       # m, post-gate waypoint offset
PASS_TOLERANCE   = 0.15      # m, waypoint-reached tolerance
TRAVEL_STEP      = 0.07      # m, max carrot distance per setpoint during TRAVEL_GATE
TRAVEL_SETTLE_S  = 0.8       # s, hold at pre-gate point before going through
LOST_THRESHOLD   = 20        # fresh misses before bailing back to SEARCH
N_GATES          = 5

# ── triangulation guards ───────────────────────────────────────────────────────

GATE_HEIGHT_REAL = 0.40
GATE_HEIGHT_TOL  = 0.15

SIZE_RATIO_MIN          = 0.5
SIZE_RATIO_MAX          = 2.0

# Rectangularity filter applied to every detection before it is used
RECT_MIN_SIDE_PX        = 10
RECT_OPPOSITE_RATIO     = 2.0   # opposite sides within this factor of each other
RECT_MIN_ASPECT         = 0.4   # min(W,H)/max(W,H)

R_CAM_TO_BODY = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)

# ── state-machine labels ───────────────────────────────────────────────────────

TAKEOFF      = 'TAKEOFF'
SEARCH       = 'SEARCH'
DETECT_1     = 'DETECT_1'
LATERAL_MOVE = 'LATERAL_MOVE'
DETECT_2     = 'DETECT_2'
TRIANGULATE  = 'TRIANGULATE'
TRAVEL_GATE  = 'TRAVEL_GATE'
DONE         = 'DONE'


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
    """Reads AI-deck UDP packets, decodes JPEG, exposes latest frame."""

    def __init__(self):
        super().__init__(daemon=True, name='UdpVideoThread')
        self._lock = threading.Lock()
        self._frame = None

    def stop(self):
        pass

    @property
    def latest_frame(self):
        with self._lock:
            return self._frame

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

        while True:
            data, _ = sock.recvfrom(2048)
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
        if img is None:
            return
        with self._lock:
            self._frame = img


# ── FPV viewer thread ─────────────────────────────────────────────────────────

class FpvViewerThread(threading.Thread):
    """
    Live FPV window. Reads the controller's last quad (does NOT re-run the
    detector) and overlays it. Capital 'Q' inside the window closes it
    (mission keeps running; lowercase 'q' is the emergency stop).
    """

    def __init__(self, cam: 'UdpVideoThread', ctrl=None):
        super().__init__(daemon=True, name='FpvViewerThread')
        self._cam = cam
        self._ctrl = ctrl
        self._running = False

    def start(self):
        self._running = True
        super().start()

    def stop(self):
        self._running = False

    def run(self):
        win = 'Crazyflie FPV'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, CAM_WIDTH * FPV_SCALE, CAM_HEIGHT * FPV_SCALE)
        last = None
        period = 1.0 / max(1.0, FPV_RATE_HZ)
        next_t = time.time()
        while self._running:
            now = time.time()
            if now < next_t:
                time.sleep(min(period, next_t - now))
                cv2.waitKey(1)
                continue
            next_t = now + period

            frame = self._cam.latest_frame
            if frame is None or frame is last:
                cv2.waitKey(1)
                continue
            last = frame
            disp = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            cv2.line(disp, (CAM_WIDTH // 2, 0), (CAM_WIDTH // 2, CAM_HEIGHT),
                     (60, 60, 60), 1)
            cv2.line(disp, (0, CAM_HEIGHT // 2), (CAM_WIDTH, CAM_HEIGHT // 2),
                     (60, 60, 60), 1)

            quad = self._ctrl.last_quad_pix if self._ctrl is not None else None
            if quad is not None:
                pts = np.int32(quad).reshape(-1, 1, 2)
                cv2.polylines(disp, [pts], True, (0, 255, 255), 2)
                for (x, y) in quad:
                    cv2.circle(disp, (int(x), int(y)), 4, (0, 255, 255), -1)
            else:
                cv2.putText(disp, 'no gate', (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if self._ctrl is not None:
                s = self._ctrl._state()
                cv2.putText(disp,
                            f"x={s['x']:+.2f} y={s['y']:+.2f} z={s['z']:.2f} yaw={s['yaw']:+.0f}",
                            (5, CAM_HEIGHT - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (255, 255, 255), 1)

            if FPV_SCALE != 1:
                disp = cv2.resize(disp,
                                  (CAM_WIDTH * FPV_SCALE, CAM_HEIGHT * FPV_SCALE),
                                  interpolation=cv2.INTER_NEAREST)
            cv2.imshow(win, disp)
            if (cv2.waitKey(1) & 0xFF) == ord('Q'):
                break
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass


# ── helpers ────────────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _clamp_to_arena(x, y):
    return (clamp(x, ARENA_X_MIN + SAFETY_MARGIN_HARD, ARENA_X_MAX - SAFETY_MARGIN_HARD),
            clamp(y, ARENA_Y_MIN + SAFETY_MARGIN_HARD, ARENA_Y_MAX - SAFETY_MARGIN_HARD))


def _quad_mean_side(quad):
    s = 0.0
    for i in range(4):
        s += float(np.linalg.norm(quad[(i + 1) % 4] - quad[i]))
    return s / 4.0


def _is_rectangular(quad_pix):
    """True if quad_pix looks roughly rectangular (gate-like). Rejects
    degenerate, sliver, or very skewed quads. Corners assumed ordered
    TL → TR → BR → BL by cv_detection.order_corners."""
    top    = float(np.linalg.norm(quad_pix[1] - quad_pix[0]))
    right  = float(np.linalg.norm(quad_pix[2] - quad_pix[1]))
    bottom = float(np.linalg.norm(quad_pix[3] - quad_pix[2]))
    left   = float(np.linalg.norm(quad_pix[0] - quad_pix[3]))
    if min(top, right, bottom, left) < RECT_MIN_SIDE_PX:
        return False
    if not (1.0 / RECT_OPPOSITE_RATIO) <= (top / bottom) <= RECT_OPPOSITE_RATIO:
        return False
    if not (1.0 / RECT_OPPOSITE_RATIO) <= (left / right) <= RECT_OPPOSITE_RATIO:
        return False
    width  = (top + bottom) / 2.0
    height = (left + right) / 2.0
    aspect = min(width, height) / max(width, height)
    if aspect < RECT_MIN_ASPECT:
        return False
    return True


# ── flight controller ──────────────────────────────────────────────────────────

class GateController:

    def __init__(self, cf: Crazyflie, cam: UdpVideoThread = None):
        self._cf  = cf
        self._cam = cam

        self.is_connected = False
        self._stop        = False

        # Log state
        self._log = {'x': 0.0, 'y': 0.0, 'z': 0.0,
                     'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
                     'yaw': 0.0,
                     'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0}
        self._log_lock = threading.Lock()
        self._log_ready = threading.Event()

        # Frame freshness
        self._last_frame_obj = None

        # For FPV overlay (no re-detection in the viewer)
        self.last_quad_pix = None

        # Two-view data
        self._P = None
        self._r_corners = np.zeros((4, 3))
        self._Q = None
        self._s_corners = np.zeros((4, 3))
        self._qn1 = None
        self._qn2 = None

        # Gate waypoints (world frame)
        self._app_x = self._app_y = 0.0
        self._mid_x = self._mid_y = 0.0
        self._exit_x = self._exit_y = 0.0
        self._gate_z   = CRUISE_ALT
        self._gate_yaw = 0.0   # rad

        # Counters
        self._frames_detected = 0
        self._lost_count      = 0
        self._travel_phase    = 0
        self._gate_count      = 0

        # Hold target (world frame, yaw in deg)
        self._hold_x = self._hold_y = 0.0
        self._hold_z = CRUISE_ALT
        self._hold_yaw = 0.0

        # SEARCH yaw target accumulator (deg)
        self._search_yaw_deg = 0.0

        # LATERAL_MOVE target + settle timer
        self._lat_x = self._lat_y = self._lat_z = 0.0
        self._lat_settle_start = None
        # TRAVEL_GATE pre-gate settle timer
        self._travel_settle_start = None

        self._setup_log()

    # ── log setup ────────────────────────────────────────────────────────────

    def _setup_log(self):
        lg1 = LogConfig(name='PosVel', period_in_ms=50)
        lg1.add_variable('stateEstimate.x',  'float')
        lg1.add_variable('stateEstimate.y',  'float')
        lg1.add_variable('stateEstimate.z',  'float')
        lg1.add_variable('stateEstimate.vx', 'float')
        lg1.add_variable('stateEstimate.vy', 'float')
        lg1.add_variable('stateEstimate.vz', 'float')

        lg2 = LogConfig(name='Att', period_in_ms=50)
        lg2.add_variable('stabilizer.yaw',   'float')
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

    def _log_cb(self, timestamp, data, logconf):
        with self._log_lock:
            for k_src, k_dst in (('stateEstimate.x', 'x'), ('stateEstimate.y', 'y'),
                                 ('stateEstimate.z', 'z'),
                                 ('stateEstimate.vx', 'vx'), ('stateEstimate.vy', 'vy'),
                                 ('stateEstimate.vz', 'vz'),
                                 ('stabilizer.yaw', 'yaw'),
                                 ('stateEstimate.qx', 'qx'), ('stateEstimate.qy', 'qy'),
                                 ('stateEstimate.qz', 'qz'), ('stateEstimate.qw', 'qw')):
                if k_src in data:
                    self._log[k_dst] = data[k_src]
        self._log_ready.set()

    def _state(self):
        with self._log_lock:
            return dict(self._log)

    # ── setpoint helpers ─────────────────────────────────────────────────────

    def _send(self, x, y, z, yaw_deg):
        self._cf.commander.send_position_setpoint(x, y, z, yaw_deg)

    def _hold(self):
        self._send(self._hold_x, self._hold_y, self._hold_z, self._hold_yaw)

    def _stop_motors(self):
        self._cf.commander.send_stop_setpoint()

    @staticmethod
    def _dist3(s, tx, ty, tz):
        return math.sqrt((s['x'] - tx)**2 + (s['y'] - ty)**2 + (s['z'] - tz)**2)

    # ── detection ────────────────────────────────────────────────────────────

    def _poll_detection(self):
        """
        Returns (status, quad_norm, quad_pix) or None for a stale frame.
        status ∈ {'ok', 'no_gate', 'no_corners', 'commit_to_pass'}.
        """
        frame = self._cam.latest_frame if self._cam is not None else None
        if frame is None or frame is self._last_frame_obj:
            return None
        self._last_frame_obj = frame

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        res = detect_gate(gray)
        status    = res['status']
        quad_pix  = res['quad_pix']
        quad_norm = res['quad_norm']

        # Reject non-rectangular quads. cv_detection still drew them, so we
        # keep last_quad_pix for the FPV overlay but downgrade the status so
        # the state machine treats it as 'no usable detection'.
        if status == 'ok' and quad_pix is not None and not _is_rectangular(quad_pix):
            status = 'no_corners'

        self.last_quad_pix = quad_pix
        return (status, quad_norm, quad_pix)

    # ── rays / triangulation ────────────────────────────────────────────────

    def _corners_to_rays(self, quad_norm, s):
        R_b2w = R.from_quat([s['qx'], s['qy'], s['qz'], s['qw']]).as_matrix()
        rays = np.zeros((4, 3))
        for i, (xn, yn) in enumerate(quad_norm):
            v_cam = np.array([xn, yn, 1.0])
            rays[i] = R_b2w @ (R_CAM_TO_BODY @ v_cam)
        return rays

    def _triangulate(self):
        """Returns (corners_3d (4,3), H (3,), gate_yaw_rad). Raises on height fail."""
        corners_3d = np.zeros((4, 3))
        for i in range(4):
            A = np.column_stack([self._r_corners[i], -self._s_corners[i]])
            b = self._Q - self._P
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            lmbda, mu = float(sol[0]), float(sol[1])
            F = self._P + lmbda * self._r_corners[i]
            G = self._Q + mu    * self._s_corners[i]
            corners_3d[i] = (F + G) / 2.0

        h_left  = np.linalg.norm(corners_3d[3] - corners_3d[0])
        h_right = np.linalg.norm(corners_3d[2] - corners_3d[1])
        gate_height = (h_left + h_right) / 2.0
        if abs(gate_height - GATE_HEIGHT_REAL) > GATE_HEIGHT_TOL:
            raise ValueError(f'gate height {gate_height:.2f} m out of range')

        H = np.mean(corners_3d, axis=0)
        v_width = corners_3d[1] - corners_3d[0]   # TL → TR
        gate_yaw = math.atan2(-v_width[1], v_width[0])
        return corners_3d, H, gate_yaw

    def _check_same_gate(self):
        """Cheap consistency test between view-1 and view-2 quads.

        Only checks size ratio — the previous bearing-shift sign/magnitude
        test assumed yaw was held constant between views, which it isn't
        (we yaw-track to keep the gate centred). Triangulation's gate-height
        check still catches gross mismatches.
        """
        q1, q2 = self._qn1, self._qn2

        size1 = _quad_mean_side(q1)
        size2 = _quad_mean_side(q2)
        if size1 < 1e-6 or size2 < 1e-6:
            return False, 'degenerate size'
        ratio = size2 / size1
        if not (SIZE_RATIO_MIN <= ratio <= SIZE_RATIO_MAX):
            return False, f'size ratio {ratio:.2f} outside [{SIZE_RATIO_MIN},{SIZE_RATIO_MAX}]'

        return True, ''

    # ── waypoints ────────────────────────────────────────────────────────────

    def _set_gate_waypoints(self, H, gate_yaw, s):
        drone_2d = np.array([s['x'], s['y']])
        gate_2d  = np.array([H[0], H[1]])
        to_drone = drone_2d - gate_2d
        forward  = np.array([math.cos(gate_yaw), math.sin(gate_yaw)])
        if np.dot(to_drone, forward) >= 0:
            gate_yaw += math.pi

        self._gate_yaw = gate_yaw
        self._gate_z   = H[2]
        fw = np.array([math.cos(gate_yaw), math.sin(gate_yaw)])

        self._app_x,  self._app_y  = _clamp_to_arena(H[0] - APPROACH_DIST * fw[0],
                                                     H[1] - APPROACH_DIST * fw[1])
        self._mid_x,  self._mid_y  = _clamp_to_arena(H[0], H[1])
        self._exit_x, self._exit_y = _clamp_to_arena(H[0] + EXIT_DIST * fw[0],
                                                     H[1] + EXIT_DIST * fw[1])
        self._travel_phase = 0

    def _force_pass(self, s):
        print('  [COMMIT_TO_PASS] gate fills frame — forcing straight pass')
        yaw_r = math.radians(s['yaw'])
        fw = np.array([math.cos(yaw_r), math.sin(yaw_r)])
        self._gate_yaw = yaw_r
        self._gate_z   = s['z']
        self._app_x,  self._app_y  = s['x'], s['y']
        mid_d = EXIT_DIST / 2.0
        self._mid_x,  self._mid_y  = _clamp_to_arena(s['x'] + mid_d * fw[0],
                                                     s['y'] + mid_d * fw[1])
        self._exit_x, self._exit_y = _clamp_to_arena(s['x'] + EXIT_DIST * fw[0],
                                                     s['y'] + EXIT_DIST * fw[1])
        self._travel_phase = 1

    # ── takeoff / land ───────────────────────────────────────────────────────

    def takeoff(self, sx, sy, syaw_deg):
        print(f'[TAKEOFF] to {CRUISE_ALT:.2f} m')
        start_z = self._state()['z']
        steps = max(1, int(TAKEOFF_DURATION / SETPOINT_PERIOD))
        for i in range(1, steps + 1):
            if self._stop:
                return
            z = start_z + (CRUISE_ALT - start_z) * (i / steps)
            self._send(sx, sy, z, syaw_deg)
            time.sleep(SETPOINT_PERIOD)
        for _ in range(20):
            if self._stop:
                return
            self._send(sx, sy, CRUISE_ALT, syaw_deg)
            time.sleep(SETPOINT_PERIOD)

    def land(self):
        print('[LAND]')
        s = self._state()
        x, y, yaw_d = s['x'], s['y'], s['yaw']
        start_z = max(s['z'], 0.1)
        steps = max(1, int(LAND_DURATION / SETPOINT_PERIOD))
        for i in range(1, steps + 1):
            if self._stop:
                break
            z = start_z * (1.0 - i / steps)
            self._send(x, y, max(z, 0.0), yaw_d)
            time.sleep(SETPOINT_PERIOD)
        self._stop_motors()

    # ── main loop ────────────────────────────────────────────────────────────

    def run_mission(self):
        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2.0)

        if not self._log_ready.wait(timeout=5.0):
            print('No state estimate received — aborting')
            return

        s = self._state()
        start_x, start_y, start_yaw = s['x'], s['y'], s['yaw']
        print(f'Start: x={start_x:.2f}  y={start_y:.2f}  yaw={start_yaw:.1f}°')

        self._hold_x, self._hold_y = start_x, start_y
        self._hold_z   = CRUISE_ALT
        self._hold_yaw = start_yaw
        self._search_yaw_deg = start_yaw

        self.takeoff(start_x, start_y, start_yaw)
        if self._stop:
            return

        mission_state = SEARCH
        print(f'[{SEARCH}]')

        try:
            while not self._stop:
                s   = self._state()
                det = self._poll_detection()
                status, quad_norm, _quad_pix = det if det is not None else (None, None, None)

                # ── SEARCH ───────────────────────────────────────────────────
                if mission_state == SEARCH:
                    self._search_yaw_deg += SEARCH_YAW_RATE * SETPOINT_PERIOD
                    self._hold_yaw = self._search_yaw_deg
                    self._hold()

                    if status == 'commit_to_pass':
                        self._force_pass(s)
                        mission_state = TRAVEL_GATE
                        print(f'[{TRAVEL_GATE}] from SEARCH commit')
                    elif status == 'ok':
                        cx_norm = float(np.mean(quad_norm[:, 0]))
                        gate_angle = math.atan2(cx_norm, 1.0)
                        self._hold_yaw = math.degrees(math.radians(s['yaw']) - gate_angle)
                        self._frames_detected = 0
                        self._lost_count      = 0
                        mission_state = DETECT_1
                        print(f'[{DETECT_1}]')

                # ── DETECT_1 ─────────────────────────────────────────────────
                elif mission_state == DETECT_1:
                    self._hold()
                    if status == 'commit_to_pass':
                        self._force_pass(s)
                        mission_state = TRAVEL_GATE
                        print(f'[{TRAVEL_GATE}] from DETECT_1 commit')
                    elif status == 'ok':
                        cx_norm = float(np.mean(quad_norm[:, 0]))
                        gate_angle = math.atan2(cx_norm, 1.0)
                        self._hold_yaw = math.degrees(math.radians(s['yaw']) - gate_angle)
                        self._lost_count = 0

                        speed = math.sqrt(s['vx']**2 + s['vy']**2 + s['vz']**2)
                        if speed < SPEED_THRESHOLD:
                            self._frames_detected += 1
                            if self._frames_detected >= REQ_FRAMES:
                                self._P = np.array([s['x'], s['y'], s['z']])
                                self._r_corners = self._corners_to_rays(quad_norm, s)
                                self._qn1 = quad_norm.copy()
                                self._frames_detected = 0

                                yaw_r = math.radians(s['yaw'])
                                # body-left and body-right candidates in world frame
                                bl = (s['x'] + LATERAL_DIST * -math.sin(yaw_r),
                                      s['y'] + LATERAL_DIST *  math.cos(yaw_r))
                                br = (s['x'] + LATERAL_DIST *  math.sin(yaw_r),
                                      s['y'] + LATERAL_DIST * -math.cos(yaw_r))
                                def _margin(p):
                                    x, y = p
                                    return min(x - ARENA_X_MIN, ARENA_X_MAX - x,
                                               y - ARENA_Y_MIN, ARENA_Y_MAX - y)
                                target = bl if _margin(bl) >= _margin(br) else br
                                self._lat_x, self._lat_y = _clamp_to_arena(*target)
                                self._lat_z = CRUISE_ALT
                                self._lat_settle_start = None
                                mission_state = LATERAL_MOVE
                                print(f'[{LATERAL_MOVE}] to ({self._lat_x:.2f},{self._lat_y:.2f})')
                        else:
                            self._frames_detected = 0
                    elif status in ('no_gate', 'no_corners'):
                        self._frames_detected = 0
                        self._lost_count += 1
                        if self._lost_count >= LOST_THRESHOLD:
                            print(f'[{SEARCH}] gate lost during DETECT_1')
                            mission_state = SEARCH

                # ── LATERAL_MOVE ─────────────────────────────────────────────
                elif mission_state == LATERAL_MOVE:
                    dx = self._lat_x - s['x']
                    dy = self._lat_y - s['y']
                    dist = math.hypot(dx, dy)
                    if dist < PASS_TOLERANCE:
                        # Arrived — hold the final target and let oscillations
                        # die down for LATERAL_SETTLE_S before sampling DETECT_2.
                        self._send(self._lat_x, self._lat_y, self._lat_z, self._hold_yaw)
                        if self._lat_settle_start is None:
                            self._lat_settle_start = time.time()
                        elif time.time() - self._lat_settle_start >= LATERAL_SETTLE_S:
                            self._hold_x, self._hold_y = self._lat_x, self._lat_y
                            self._frames_detected = 0
                            self._lost_count = 0
                            mission_state = DETECT_2
                            print(f'[{DETECT_2}]')
                    else:
                        # Carrot setpoint: at most LATERAL_STEP m ahead of the
                        # drone toward the target. Keeps firmware velocity ramp
                        # short → smooth, slow, no overshoot.
                        self._lat_settle_start = None
                        step = min(LATERAL_STEP, dist)
                        cx_t = s['x'] + step * dx / dist
                        cy_t = s['y'] + step * dy / dist
                        self._send(cx_t, cy_t, self._lat_z, self._hold_yaw)

                # ── DETECT_2 ─────────────────────────────────────────────────
                elif mission_state == DETECT_2:
                    self._hold()
                    if status == 'commit_to_pass':
                        self._force_pass(s)
                        mission_state = TRAVEL_GATE
                        print(f'[{TRAVEL_GATE}] from DETECT_2 commit')
                    elif status == 'ok':
                        cx_norm = float(np.mean(quad_norm[:, 0]))
                        gate_angle = math.atan2(cx_norm, 1.0)
                        self._hold_yaw = math.degrees(math.radians(s['yaw']) - gate_angle)
                        self._lost_count = 0

                        speed = math.sqrt(s['vx']**2 + s['vy']**2 + s['vz']**2)
                        if speed < SPEED_THRESHOLD:
                            self._frames_detected += 1
                            if self._frames_detected >= REQ_FRAMES:
                                self._Q = np.array([s['x'], s['y'], s['z']])
                                self._s_corners = self._corners_to_rays(quad_norm, s)
                                self._qn2 = quad_norm.copy()
                                self._frames_detected = 0

                                ok, reason = self._check_same_gate()
                                if not ok:
                                    print(f'[REJECT] same-gate check: {reason} → SEARCH')
                                    self._lost_count = 0
                                    mission_state = SEARCH
                                else:
                                    mission_state = TRIANGULATE
                                    print(f'[{TRIANGULATE}]')
                        else:
                            self._frames_detected = 0
                    elif status in ('no_gate', 'no_corners'):
                        self._frames_detected = 0
                        self._lost_count += 1
                        if self._lost_count >= LOST_THRESHOLD:
                            print(f'[{SEARCH}] gate lost during DETECT_2')
                            mission_state = SEARCH

                # ── TRIANGULATE ──────────────────────────────────────────────
                elif mission_state == TRIANGULATE:
                    self._hold()
                    try:
                        corners_3d, H, gate_yaw = self._triangulate()
                        print(f'Gate {self._gate_count + 1}: centre=({H[0]:.2f},{H[1]:.2f},'
                              f'{H[2]:.2f})  yaw={math.degrees(gate_yaw):.1f}°')
                        self._set_gate_waypoints(H, gate_yaw, s)
                        mission_state = TRAVEL_GATE
                        print(f'[{TRAVEL_GATE}]')
                    except Exception as e:
                        print(f'[REJECT] triangulation: {e} → SEARCH')
                        mission_state = SEARCH

                # ── TRAVEL_GATE ──────────────────────────────────────────────
                elif mission_state == TRAVEL_GATE:
                    gyd = math.degrees(self._gate_yaw)

                    def _carrot_to(tx, ty, tz):
                        """Send a position setpoint at most TRAVEL_STEP m
                        ahead of the drone toward (tx, ty, tz). Returns the
                        remaining 2-D distance so the caller can decide when
                        to advance to the next phase."""
                        dx = tx - s['x']
                        dy = ty - s['y']
                        dist = math.hypot(dx, dy)
                        if dist < 1e-3:
                            self._send(tx, ty, tz, gyd)
                        else:
                            step = min(TRAVEL_STEP, dist)
                            self._send(s['x'] + step * dx / dist,
                                       s['y'] + step * dy / dist,
                                       tz, gyd)
                        return dist

                    if self._travel_phase == 0:
                        # APPROACH — carrot toward pre-gate point, then SETTLE
                        # there before going through (drone must actually stop).
                        dist = _carrot_to(self._app_x, self._app_y, self._gate_z)
                        if dist < PASS_TOLERANCE:
                            if self._travel_settle_start is None:
                                self._travel_settle_start = time.time()
                                print('  [TRAVEL] at pre-gate, settling…')
                            elif time.time() - self._travel_settle_start >= TRAVEL_SETTLE_S:
                                self._travel_settle_start = None
                                self._travel_phase = 1
                                print('  [TRAVEL] settled → going through')
                        else:
                            self._travel_settle_start = None
                    elif self._travel_phase == 1:
                        # GO THROUGH — carrot toward gate centre, no settle
                        dist = _carrot_to(self._mid_x, self._mid_y, self._gate_z)
                        if dist < PASS_TOLERANCE:
                            self._travel_phase = 2
                    elif self._travel_phase == 2:
                        # EXIT — carrot to post-gate point
                        dist = _carrot_to(self._exit_x, self._exit_y, self._gate_z)
                        if dist < PASS_TOLERANCE:
                            self._gate_count += 1
                            print(f'Gate {self._gate_count} passed!')
                            self._hold_x, self._hold_y = s['x'], s['y']
                            self._hold_yaw = s['yaw']
                            self._search_yaw_deg = s['yaw']
                            if self._gate_count >= N_GATES:
                                mission_state = DONE
                                print(f'[{DONE}] all {N_GATES} gates — hovering')
                            else:
                                mission_state = SEARCH
                                print(f'[{SEARCH}] ({self._gate_count}/{N_GATES} done)')

                # ── DONE ─────────────────────────────────────────────────────
                elif mission_state == DONE:
                    self._hold()

                time.sleep(SETPOINT_PERIOD)

        except Exception as e:
            print(f'Unhandled exception: {e}')
        finally:
            try:
                self.land()
            except Exception:
                self._stop_motors()


# ── emergency stop ─────────────────────────────────────────────────────────────

def emergency_stop_listener(ctrl: GateController):
    """ESC → request controlled land (main thread handles it).
       Q   → request immediate motor kill (also fired here, then main thread cleans up).

    Only writes ctrl._stop and (on Q) one stop_setpoint. Landing and link
    teardown stay on the main thread to avoid racing cflib from two threads.
    """

    def on_press(key):
        if key == keyboard.Key.esc:
            print('\n[EMERGENCY STOP] requesting controlled land')
            ctrl._stop = True
            return False

        try:
            if key.char == 'q':
                print('\n[EMERGENCY STOP] cutting motors')
                ctrl._stop = True
                try:
                    ctrl._stop_motors()
                except Exception:
                    pass
                return False
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cam = UdpVideoThread()
    cam.start()

    cflib.crtp.init_drivers()

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

    print('Waiting for first camera frame...')
    while cam.latest_frame is None:
        time.sleep(0.05)
    print('Camera ready')

    ctrl = GateController(cf, cam)
    if not ctrl.is_connected:
        print('Log setup failed — exiting')
        cam.stop()
        cf.close_link()
        exit(1)

    fpv = None
    if FPV_ENABLED:
        fpv = FpvViewerThread(cam, ctrl)
        fpv.start()
        print('FPV viewer started')

    threading.Thread(
        target=emergency_stop_listener,
        args=(ctrl,),
        daemon=True,
    ).start()

    try:
        ctrl.run_mission()
    finally:
        if fpv is not None:
            fpv.stop()
        if cam is not None:
            cam.stop()
        cf.close_link()
