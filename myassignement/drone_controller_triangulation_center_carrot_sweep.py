# -*- coding: utf-8 -*-
"""
drone_controller_triangulation_center_carrot_sweep.py
======================================================

Centre-only triangulation variant of drone_controller_triangulation_carrot.py
with the lateral-direction sweep fix from drone_controller_triangulation_sweep.py.

Differences vs. the 4-corner reference (drone_controller_triangulation.py):
  • One ray per view through the detected gate centroid (no corner rays).
  • Triangulation = single least-squares ray intersection → gate centre only.
  • Gate plane orientation is NOT recovered. Approach axis is approximated as
    "gate faces the midpoint of the two observation positions."
  • TRAVEL_GATE uses carrot setpoints + pre-gate settle (from _carrot variant).
  • LATERAL_MOVE direction picked from image bearing (from _sweep variant):
    if cx_norm > 0.33 the gate is right-of-centre → go LEFT, else go RIGHT.
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
LATERAL_DIST     = 0.5       # m, sideways baseline between views
LATERAL_STEP     = 0.07      # m, carrot step during lateral move
LATERAL_SETTLE_S = 1.0       # s, hold at lateral target before DETECT_2
REQ_FRAMES       = 5
SPEED_THRESHOLD  = 0.10
APPROACH_DIST    = 0.4
EXIT_DIST        = 0.4
PASS_TOLERANCE   = 0.15
TRAVEL_STEP      = 0.07      # m, carrot step during TRAVEL_GATE
TRAVEL_SETTLE_S  = 0.8       # s, settle at pre-gate point before going through
LOST_THRESHOLD   = 20
N_GATES          = 5

# Image-bearing threshold (normalised x): above this → gate is right of centre.
CX_RIGHT_BIAS    = 0.33

# ── triangulation guards (centre-only) ─────────────────────────────────────────

# Closest-approach distance between the two centre rays. Rejects bad geometry.
TRIANGULATE_SKEW_MAX = 0.30   # m
GATE_RANGE_MIN       = 0.2    # m, min distance from baseline midpoint to gate
GATE_RANGE_MAX       = 6.0    # m, max distance

# Vertical bias applied to the triangulated gate centre: single-ray
# triangulation consistently underestimates Z, so we lift the waypoint.
GATE_Z_OFFSET        = 0.15   # m

SIZE_RATIO_MIN          = 0.5
SIZE_RATIO_MAX          = 2.0

RECT_MIN_SIDE_PX        = 10
RECT_OPPOSITE_RATIO     = 2.0
RECT_MIN_ASPECT         = 0.4

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
                cx = int(np.mean(quad[:, 0]))
                cy = int(np.mean(quad[:, 1]))
                cv2.drawMarker(disp, (cx, cy), (0, 255, 0),
                               cv2.MARKER_CROSS, 12, 2)
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

        self._log = {'x': 0.0, 'y': 0.0, 'z': 0.0,
                     'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
                     'yaw': 0.0,
                     'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0}
        self._log_lock = threading.Lock()
        self._log_ready = threading.Event()

        self._last_frame_obj = None
        self.last_quad_pix = None

        # Two-view data — single ray per view through gate centre.
        self._P = None
        self._r_centre = None
        self._Q = None
        self._s_centre = None
        self._qn1 = None
        self._qn2 = None

        # Gate waypoints (world frame)
        self._app_x = self._app_y = 0.0
        self._mid_x = self._mid_y = 0.0
        self._exit_x = self._exit_y = 0.0
        self._gate_z   = CRUISE_ALT
        self._gate_yaw = 0.0   # rad

        self._frames_detected = 0
        self._lost_count      = 0
        self._travel_phase    = 0
        self._gate_count      = 0

        self._hold_x = self._hold_y = 0.0
        self._hold_z = CRUISE_ALT
        self._hold_yaw = 0.0

        self._search_yaw_deg = 0.0

        self._lat_x = self._lat_y = self._lat_z = 0.0
        self._lat_settle_start = None
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

        if status == 'ok' and quad_pix is not None and not _is_rectangular(quad_pix):
            status = 'no_corners'

        self.last_quad_pix = quad_pix
        return (status, quad_norm, quad_pix)

    # ── single-ray triangulation ────────────────────────────────────────────

    def _centre_ray(self, quad_norm, s):
        cx = float(np.mean(quad_norm[:, 0]))
        cy = float(np.mean(quad_norm[:, 1]))
        v_cam = np.array([cx, cy, 1.0])
        R_b2w = R.from_quat([s['qx'], s['qy'], s['qz'], s['qw']]).as_matrix()
        ray = R_b2w @ (R_CAM_TO_BODY @ v_cam)
        n = np.linalg.norm(ray)
        return ray / n if n > 1e-9 else ray

    def _triangulate(self):
        """Closest-point intersection of the two centre rays.

        Returns (H (3,), skew_m). Raises ValueError on bad geometry.
        """
        r = self._r_centre
        s_dir = self._s_centre
        d = self._Q - self._P

        A = np.column_stack([r, -s_dir])
        sol, _, _, _ = np.linalg.lstsq(A, d, rcond=None)
        lmbda, mu = float(sol[0]), float(sol[1])

        if lmbda <= 0 or mu <= 0:
            raise ValueError(f'rays point behind cameras (λ={lmbda:.2f}, μ={mu:.2f})')

        F = self._P + lmbda * r
        G = self._Q + mu    * s_dir
        H = (F + G) / 2.0
        skew = float(np.linalg.norm(F - G))

        if skew > TRIANGULATE_SKEW_MAX:
            raise ValueError(f'ray skew {skew:.2f} m > {TRIANGULATE_SKEW_MAX:.2f}')

        mid = 0.5 * (self._P + self._Q)
        rng = float(np.linalg.norm(H[:2] - mid[:2]))
        if not (GATE_RANGE_MIN <= rng <= GATE_RANGE_MAX):
            raise ValueError(f'gate range {rng:.2f} m outside [{GATE_RANGE_MIN},{GATE_RANGE_MAX}]')

        return H, skew

    def _check_same_gate(self):
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

    def _set_gate_waypoints(self, H, s):
        """No gate-plane normal available with centre-only triangulation.
        Approximate the approach axis as the bearing from the gate back
        toward the midpoint of the two observation positions.
        """
        mid = 0.5 * (self._P + self._Q)
        dx = mid[0] - H[0]
        dy = mid[1] - H[1]
        if math.hypot(dx, dy) < 1e-6:
            yaw_r = math.radians(s['yaw'])
            dx, dy = math.cos(yaw_r), math.sin(yaw_r)

        approach_dir = np.array([dx, dy]) / math.hypot(dx, dy)
        gate_yaw = math.atan2(-approach_dir[1], -approach_dir[0])

        self._gate_yaw = gate_yaw
        self._gate_z   = H[2] + GATE_Z_OFFSET
        fw = np.array([math.cos(gate_yaw), math.sin(gate_yaw)])

        self._app_x,  self._app_y  = _clamp_to_arena(H[0] - APPROACH_DIST * fw[0],
                                                     H[1] - APPROACH_DIST * fw[1])
        self._mid_x,  self._mid_y  = _clamp_to_arena(H[0], H[1])
        self._exit_x, self._exit_y = _clamp_to_arena(H[0] + EXIT_DIST * fw[0],
                                                     H[1] + EXIT_DIST * fw[1])
        self._travel_phase = 0
        self._travel_settle_start = None

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
        self._travel_settle_start = None

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
                                self._r_centre = self._centre_ray(quad_norm, s)
                                self._qn1 = quad_norm.copy()
                                self._frames_detected = 0

                                # --- sweep direction (from _sweep variant) ---
                                yaw_r = math.radians(s['yaw'])
                                right = (s['x'] + LATERAL_DIST *  math.sin(yaw_r),
                                         s['y'] + LATERAL_DIST * -math.cos(yaw_r))
                                left  = (s['x'] + LATERAL_DIST * -math.sin(yaw_r),
                                         s['y'] + LATERAL_DIST *  math.cos(yaw_r))
                                cx_norm = float(np.mean(quad_norm[:, 0]))
                                if cx_norm > CX_RIGHT_BIAS:
                                    target = left
                                    direction_str = 'LEFT'
                                else:
                                    target = right
                                    direction_str = 'RIGHT'

                                self._lat_x, self._lat_y = _clamp_to_arena(*target)
                                self._lat_z = CRUISE_ALT
                                self._lat_settle_start = None
                                mission_state = LATERAL_MOVE
                                print(f'[{LATERAL_MOVE}] cx={cx_norm:.2f} → {direction_str} '
                                      f'({self._lat_x:.2f},{self._lat_y:.2f})')
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
                                self._s_centre = self._centre_ray(quad_norm, s)
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
                        H, skew = self._triangulate()
                        print(f'Gate {self._gate_count + 1}: centre=({H[0]:.2f},{H[1]:.2f},'
                              f'{H[2]:.2f})  skew={skew:.3f} m')
                        self._set_gate_waypoints(H, s)
                        print(f'  gate_yaw≈{math.degrees(self._gate_yaw):.1f}° (assumed facing drone)')
                        mission_state = TRAVEL_GATE
                        print(f'[{TRAVEL_GATE}]')
                    except Exception as e:
                        print(f'[REJECT] triangulation: {e} → SEARCH')
                        mission_state = SEARCH

                # ── TRAVEL_GATE (carrot + pre-gate settle) ───────────────────
                elif mission_state == TRAVEL_GATE:
                    gyd = math.degrees(self._gate_yaw)

                    def _carrot_to(tx, ty, tz):
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
                        dist = _carrot_to(self._mid_x, self._mid_y, self._gate_z)
                        if dist < PASS_TOLERANCE:
                            self._travel_phase = 2
                    elif self._travel_phase == 2:
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
