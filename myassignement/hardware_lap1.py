"""
hardware_lap1.py  —  Crazyflie hardware lap-1 controller
=========================================================

Mission: take off, discover and fly through all NUM_GATES gates using
two-view triangulation, then hover.

Detection  : cv_detection.detect_gate()  (green LED gate, grayscale)
Triangulation : ported from matt_assignment.py COMPUTE_GATE_POS
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
from collections import deque

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

# ── connection ──────────────────────────────────────────────────────────────────

CONTROL_URI = uri_helper.uri_from_env(default='radio://0/20/2M/E7E7E7E708')

UDP_AIDECK_IP   = '192.168.4.1'
UDP_AIDECK_PORT = 5000
UDP_LOCAL_PORT  = 5001
UDP_START_MAGIC = b'FER'

# ── camera ──────────────────────────────────────────────────────────────────────

CAM_WIDTH        = 324
CAM_HEIGHT       = 244
CPX_HEADER_SIZE  = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE  = 11
MIN_JPEG_BYTES   = 5000

# ── flight constants ────────────────────────────────────────────────────────────

CRUISE_ALT       = 1.2    # m
TAKEOFF_DURATION = 4.0    # s
LAND_DURATION    = 3.0    # s
SETPOINT_PERIOD  = 0.05   # s  (20 Hz)

SEARCH_YAW_RATE  = 5.0   # deg/s CCW rotation during search
LATERAL_DIST     = 0.5    # m — baseline between the two detection views
REQ_FRAMES       = 5      # consecutive good detections needed (drone stationary)
SPEED_THRESHOLD  = 0.10   # m/s — "stationary" guard (same as sim)
APPROACH_DIST    = 0.6    # m — approach waypoint in front of gate
EXIT_DIST        = 0.6    # m — exit waypoint behind gate
PASS_TOLERANCE   = 0.10   # m — waypoint-reached threshold
LOST_THRESHOLD   = 20     # new-frame no-gate detections before giving up
NUM_GATES        = 5

# Gate physical dimensions — used both to sanity-check triangulated corners
# and to estimate distance from apparent height during single-view detection.
# Corners are ordered [TL, TR, BR, BL]; height = mean of left-side and right-side spans.
GATE_HEIGHT_REAL = 0.40   # m (nominal)
GATE_HEIGHT_TOL  = 0.15   # m — accept [0.25, 0.55] to allow triangulation noise

# Single-view distance plausibility (from apparent height in normalised coords)
GATE_DIST_MIN = 0.3   # m — closer than this is physically implausible
GATE_DIST_MAX = 3.0   # m — farther than this → detection is too noisy to trust

# Clock-sector gate filter (same convention as matt_assignment.py).
# The 360° around CIRCUIT_CENTER is divided into 12 sectors of 30° each,
# numbered like a clock face (sector 0 = 12 o'clock = +Y axis).
# Clock angle = (360 - math_angle_deg + 15) % 360
# Sector centre = sector_number * 30 + 15 degrees.
# EXPECTED_SECTORS lists the expected sector for each gate in traversal order.
CIRCUIT_OFFSET       = 1.0    # m — circuit centre is this far in front of start position
EXPECTED_SECTORS     = [4, 2, 0, 10, 8]       # one per gate; tune for hardware arena
SECTOR_TOLERANCE_DEG = 45.0                    # accept ± this many degrees

# Camera delay is informational only — drone is stationary during corner grabs,
# so we use the latest log state directly (delay error < 1 cm at 0 m/s).
CAM_DELAY_MS     = 120

# Rotation from camera frame to body frame (same convention as sim)
R_CAM_TO_BODY = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)

# ── states ──────────────────────────────────────────────────────────────────────

TAKEOFF      = 'TAKEOFF'
SEARCH       = 'SEARCH'
DETECT_1     = 'DETECT_1'
LATERAL_MOVE = 'LATERAL_MOVE'
DETECT_2     = 'DETECT_2'
TRIANGULATE  = 'TRIANGULATE'
TRAVEL_GATE  = 'TRAVEL_GATE'
DONE         = 'DONE'


# ── state buffer ────────────────────────────────────────────────────────────────

class StateBuffer:
    """
    Rolling buffer of (wall_time, state_dict) pairs from the log callback.

    Allows matching camera frames to the drone pose at their capture time.
    At the low approach speeds used here the drone is stationary during
    corner grabs, so time-matching is mostly a safety net.
    """

    def __init__(self, max_len=60):
        self._buf  = deque(maxlen=max_len)   # 60 × 50 ms = 3 s of history
        self._lock = threading.Lock()

    def push(self, wall_time, state):
        with self._lock:
            self._buf.append((wall_time, dict(state)))

    def state_at(self, t):
        """Return the state whose timestamp is closest to t (wall-clock seconds)."""
        with self._lock:
            if not self._buf:
                return None
            return min(self._buf, key=lambda e: abs(e[0] - t))[1]

    @property
    def latest(self):
        with self._lock:
            return dict(self._buf[-1][1]) if self._buf else None


# ── UDP video thread ─────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _muted_stderr():
    saved = os.dup(2)
    null  = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(null)
        os.close(saved)


class UdpVideoThread(threading.Thread):
    """Receives AI-deck UDP packets, decodes JPEG, exposes (frame, wall_time)."""

    def __init__(self):
        super().__init__(daemon=True, name='UdpVideoThread')
        self._lock     = threading.Lock()
        self._frame    = None
        self._frame_ts = 0.0     # wall-clock time of decode (arrival proxy)
        self._running  = False

    def start(self):
        self._running = True
        super().start()

    def stop(self):
        self._running = False

    @property
    def latest_frame_with_ts(self):
        """Return (grayscale_frame_or_None, wall_time_seconds)."""
        with self._lock:
            print('Latest frame is ', (self._frame is not None), 'with timestamp', self._frame_ts)

            return self._frame, self._frame_ts

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(('0.0.0.0', UDP_LOCAL_PORT))
        sock.sendto(UDP_START_MAGIC, (UDP_AIDECK_IP, UDP_AIDECK_PORT))

        buffer        = bytearray()
        expected_size = 0
        receiving     = False

        while self._running:
            data, _ = sock.recvfrom(2048)

            if len(data) < CPX_HEADER_SIZE:
                continue
            payload = data[CPX_HEADER_SIZE:]

            if len(payload) >= IMG_HEADER_SIZE and payload[0] == IMG_HEADER_MAGIC:
                _, w, h, _, _, size = struct.unpack('<BHHBBI', payload[:IMG_HEADER_SIZE])
                if w == CAM_WIDTH and h == CAM_HEIGHT and 0 < size < 65536:
                    expected_size = size
                    buffer        = bytearray()
                    receiving     = True
                    continue

            if not receiving:
                continue
            buffer.extend(payload)

            if len(buffer) >= expected_size:
                frame = self._decode(buffer)
                if frame is not None:
                    with self._lock:
                        self._frame    = frame
                        self._frame_ts = time.time()
                receiving = False

    def _decode(self, buf):
        soi = buf.find(b'\xff\xd8')
        eoi = buf.rfind(b'\xff\xd9')
        if soi < 0 or eoi <= soi:
            return None
        jpeg_len = eoi + 2 - soi
        if jpeg_len < MIN_JPEG_BYTES:
            return None
        jpeg = np.frombuffer(buf, np.uint8, count=jpeg_len, offset=soi)
        with _muted_stderr():
            img = cv2.imdecode(jpeg, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[:2] != (CAM_HEIGHT, CAM_WIDTH):
            return None
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


# ── main controller ─────────────────────────────────────────────────────────────

class Lap1Controller:

    def __init__(self, cf: Crazyflie, cam: UdpVideoThread):
        self._cf        = cf
        self._cam       = cam
        self._state_buf = StateBuffer()
        self._stop      = False
        self.is_connected = False

        # triangulation data (ported from matt_assignment.py)
        self._P         = None               # drone position at view 1 (3,)
        self._r_corners = np.zeros((4, 3))   # world-frame ray dirs from P
        self._Q         = None               # drone position at view 2 (3,)
        self._s_corners = np.zeros((4, 3))   # world-frame ray dirs from Q

        # gate traversal waypoints
        self._app_x  = self._app_y  = 0.0
        self._mid_x  = self._mid_y  = 0.0
        self._exit_x = self._exit_y = 0.0
        self._gate_z   = CRUISE_ALT
        self._gate_yaw = 0.0   # rad

        # mission counters
        self._frames_detected  = 0
        self._lost_count       = 0   # consecutive new-frame no-gate results
        self._travel_phase     = 0
        self._gate_count       = 0

        # circuit geometry (set in run_mission from start pose)
        self._circuit_center = np.zeros(2)

        # search tracking
        self._search_yaw_deg   = 0.0   # accumulated target yaw during search
        self._search_yaw_total = 0.0   # deg rotated since search started

        # lateral move target
        self._lat_x = self._lat_y = self._lat_z = 0.0

        # current hold position / yaw target (all in world frame, yaw in deg)
        self._hold_x   = 0.0
        self._hold_y   = 0.0
        self._hold_z   = CRUISE_ALT
        self._hold_yaw = 0.0   # deg

        # last processed frame timestamp (to avoid re-processing same frame)
        self._last_frame_ts = 0.0

        # log state
        self._log = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'yaw': 0.0,
            'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0,
        }
        self._log_lock  = threading.Lock()
        self._log_ready = threading.Event()

        self._setup_log()

    # ── log setup ────────────────────────────────────────────────────────────────

    def _setup_log(self):
        # cflib LogConfig max payload = 26 bytes = 6 floats per block.
        # Split 11 variables across two blocks to stay within the limit.

        # Block 1: position + velocity  (6 × 4 = 24 bytes)
        lg1 = LogConfig(name='Lap1PosVel', period_in_ms=50)
        lg1.add_variable('stateEstimate.x',  'float')
        lg1.add_variable('stateEstimate.y',  'float')
        lg1.add_variable('stateEstimate.z',  'float')
        lg1.add_variable('stateEstimate.vx', 'float')
        lg1.add_variable('stateEstimate.vy', 'float')
        lg1.add_variable('stateEstimate.vz', 'float')

        # Block 2: yaw + quaternion  (5 × 4 = 20 bytes)
        lg2 = LogConfig(name='Lap1Att', period_in_ms=50)
        lg2.add_variable('stabilizer.yaw',   'float')   # degrees
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
        # Called by both log blocks; each call updates only the keys it carries.
        with self._log_lock:
            if 'stateEstimate.x'  in data: self._log['x']  = data['stateEstimate.x']
            if 'stateEstimate.y'  in data: self._log['y']  = data['stateEstimate.y']
            if 'stateEstimate.z'  in data: self._log['z']  = data['stateEstimate.z']
            if 'stateEstimate.vx' in data: self._log['vx'] = data['stateEstimate.vx']
            if 'stateEstimate.vy' in data: self._log['vy'] = data['stateEstimate.vy']
            if 'stateEstimate.vz' in data: self._log['vz'] = data['stateEstimate.vz']
            if 'stabilizer.yaw'   in data: self._log['yaw'] = data['stabilizer.yaw']
            if 'stateEstimate.qx' in data: self._log['qx'] = data['stateEstimate.qx']
            if 'stateEstimate.qy' in data: self._log['qy'] = data['stateEstimate.qy']
            if 'stateEstimate.qz' in data: self._log['qz'] = data['stateEstimate.qz']
            if 'stateEstimate.qw' in data: self._log['qw'] = data['stateEstimate.qw']
            snap = dict(self._log)
        self._state_buf.push(time.time(), snap)
        self._log_ready.set()

    def _state(self):
        with self._log_lock:
            return dict(self._log)

    # ── setpoint helpers ─────────────────────────────────────────────────────────

    def _send(self, x, y, z, yaw_deg):
        self._cf.commander.send_position_setpoint(x, y, z, yaw_deg)

    def _hold(self):
        self._send(self._hold_x, self._hold_y, self._hold_z, self._hold_yaw)

    def _stop_motors(self):
        self._cf.commander.send_stop_setpoint()

    # ── geometry helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _dist3(s, tx, ty, tz):
        return math.sqrt((s['x'] - tx)**2 + (s['y'] - ty)**2 + (s['z'] - tz)**2)

    # ── single-view detection validation ────────────────────────────────────────

    def _validate_detection(self, quad_norm, s):
        """
        Use the known gate height to estimate distance from the apparent
        vertical span in undistorted normalised coords, then check that the
        estimated gate world position lies on the expected gate circuit.

        In normalised coords (focal length = 1), the thin-lens relation gives:
            distance ≈ GATE_HEIGHT_REAL / y_span

        Returns (est_dist, est_gate_x, est_gate_y) on success, or None if the
        detection is inconsistent with the known circuit geometry.
        """
        y_span = float(np.max(quad_norm[:, 1]) - np.min(quad_norm[:, 1]))
        if y_span < 1e-4:
            return None

        est_dist = GATE_HEIGHT_REAL / y_span
        if est_dist < GATE_DIST_MIN or est_dist > GATE_DIST_MAX:
            return None

        # Bearing to gate centre in world frame
        cx_norm = float(np.mean(quad_norm[:, 0]))
        bearing = math.radians(s['yaw']) - math.atan2(cx_norm, 1.0)
        est_x   = s['x'] + est_dist * math.cos(bearing)
        est_y   = s['y'] + est_dist * math.sin(bearing)

        # Clock-sector filter — identical logic to matt_assignment.py:248-263.
        # Compute the clock angle of the estimated gate position relative to
        # CIRCUIT_CENTER, then check it falls within the expected sector for
        # the gate we are currently trying to find.
        angle_rad    = math.atan2(est_y - self._circuit_center[1], est_x - self._circuit_center[0])
        angle_deg    = math.degrees(angle_rad)
        clock_angle  = (360 - angle_deg + 15) % 360
        sector       = EXPECTED_SECTORS[self._gate_count % len(EXPECTED_SECTORS)]
        sector_centre = sector * 30 + 15
        angle_diff   = abs(clock_angle - sector_centre)
        angle_diff   = min(angle_diff, 360 - angle_diff)
        if angle_diff > SECTOR_TOLERANCE_DEG:
            return None

        return est_dist, est_x, est_y

    # ── detection helper ─────────────────────────────────────────────────────────

    def _detect(self):
        """
        Process the latest camera frame through detect_gate().
        Returns one of:
          ('ok',            quad_norm (4,2), cx_norm float)
          ('commit_to_pass', None,           0.0)
          ('no_gate',        None,           0.0)
          None   — no new frame since last call
        """
        frame, ts = self._cam.latest_frame_with_ts
        if frame is None or ts == self._last_frame_ts:
            print('NO FRAMES')
            return None
        self._last_frame_ts = ts

        result = detect_gate(frame)

        if result['status'] == 'ok':
            qn      = result['quad_norm']            # (4,2) undistorted normalised
            cx_norm = float(np.mean(qn[:, 0]))
            return ('ok', qn, cx_norm)
        elif result['status'] == 'commit_to_pass':
            return ('commit_to_pass', None, 0.0)
        else:
            return ('no_gate', None, 0.0)

    # ── ray directions from undistorted normalised corners ───────────────────────

    def _corners_to_rays(self, quad_norm, s):
        """
        Convert undistorted normalised corners to world-frame unit rays.

        quad_norm : (4,2)  from detect_gate()['quad_norm']
        s         : state dict with qx,qy,qz,qw

        In the sim, direction vectors use raw pixels:
            v_cam = [px - W/2, py - H/2, focal_length]
        Here quad_norm is already the result of cv2.undistortPoints() with the
        real calibration matrix K, so v_cam = [x_n, y_n, 1.0] directly.
        """
        R_b2w = R.from_quat([s['qx'], s['qy'], s['qz'], s['qw']]).as_matrix()
        rays   = np.zeros((4, 3))
        for i, (xn, yn) in enumerate(quad_norm):
            v_cam   = np.array([xn, yn, 1.0])
            rays[i] = R_b2w @ (R_CAM_TO_BODY @ v_cam)
        return rays

    # ── triangulation (ported from matt_assignment.py:592-614) ──────────────────

    def _triangulate(self):
        """
        Two-view triangulation for each gate corner.
        Returns (corners_3d (4,3), gate_center (3,), gate_yaw_rad float).
        Raises ValueError if the reconstructed gate height is inconsistent
        with the known physical height (sanity-check against bad baselines
        or mismatched corner associations).
        Corners are ordered [TL, TR, BR, BL].
        """
        corners_3d = np.zeros((4, 3))
        for i in range(4):
            # Solve P + λ·r = Q + μ·s  in least-squares sense
            A         = np.column_stack([self._r_corners[i], -self._s_corners[i]])
            b         = self._Q - self._P
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            lmbda, mu = float(sol[0]), float(sol[1])
            F          = self._P + lmbda * self._r_corners[i]
            G          = self._Q + mu    * self._s_corners[i]
            corners_3d[i] = (F + G) / 2.0

        # Sanity-check: reconstructed gate height (mean of left and right spans)
        # should match the known physical gate height within tolerance.
        h_left  = np.linalg.norm(corners_3d[3] - corners_3d[0])   # BL - TL
        h_right = np.linalg.norm(corners_3d[2] - corners_3d[1])   # BR - TR
        gate_height = (h_left + h_right) / 2.0
        if abs(gate_height - GATE_HEIGHT_REAL) > GATE_HEIGHT_TOL:
            raise ValueError(
                f'gate height {gate_height:.2f} m outside expected '
                f'{GATE_HEIGHT_REAL - GATE_HEIGHT_TOL:.2f}–'
                f'{GATE_HEIGHT_REAL + GATE_HEIGHT_TOL:.2f} m'
            )

        H = np.mean(corners_3d, axis=0)

        v_width   = corners_3d[1] - corners_3d[0]   # TL → TR
        gate_yaw  = math.atan2(-v_width[1], v_width[0])   # normal to width vector

        return corners_3d, H, gate_yaw

    # ── gate waypoints with side disambiguation ──────────────────────────────────

    def _set_gate_waypoints(self, H, gate_yaw, s):
        """
        Compute approach / centre / exit waypoints.
        Flips gate_yaw if the drone is already on the exit side.
        """
        drone_2d = np.array([s['x'], s['y']])
        gate_2d  = np.array([H[0], H[1]])
        to_drone = drone_2d - gate_2d
        forward  = np.array([math.cos(gate_yaw), math.sin(gate_yaw)])

        if np.dot(to_drone, forward) >= 0:
            # drone is on the exit (+forward) side → flip so we approach from here
            gate_yaw += math.pi

        self._gate_yaw = gate_yaw
        self._gate_z   = H[2]

        fw = np.array([math.cos(gate_yaw), math.sin(gate_yaw)])

        self._app_x  = H[0] - APPROACH_DIST * fw[0]
        self._app_y  = H[1] - APPROACH_DIST * fw[1]
        self._mid_x  = H[0]
        self._mid_y  = H[1]
        self._exit_x = H[0] + EXIT_DIST * fw[0]
        self._exit_y = H[1] + EXIT_DIST * fw[1]
        self._travel_phase = 0

    # ── forced pass (gate too close for full measurement) ────────────────────────

    def _force_pass(self, s):
        print('[COMMIT_TO_PASS] gate fills frame — forcing straight pass')
        yaw_rad      = math.radians(s['yaw'])
        fw           = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
        self._gate_yaw = yaw_rad
        self._gate_z   = s['z']
        self._app_x    = s['x']
        self._app_y    = s['y']
        self._mid_x    = s['x'] + 0.4 * fw[0]
        self._mid_y    = s['y'] + 0.4 * fw[1]
        self._exit_x   = s['x'] + EXIT_DIST * fw[0]
        self._exit_y   = s['y'] + EXIT_DIST * fw[1]
        self._travel_phase = 1   # skip approach; already at gate

    # ── takeoff / land ────────────────────────────────────────────────────────────

    def takeoff(self, sx, sy, syaw_deg):
        print(f'[TAKEOFF] to {CRUISE_ALT:.2f} m')
        start_z = self._state()['z']
        steps   = max(1, int(TAKEOFF_DURATION / SETPOINT_PERIOD))
        for i in range(1, steps + 1):
            if self._stop:
                return
            z = start_z + (CRUISE_ALT - start_z) * (i / steps)
            self._send(sx, sy, z, syaw_deg)
            time.sleep(SETPOINT_PERIOD)
        # settle
        for _ in range(20):
            if self._stop:
                return
            self._send(sx, sy, CRUISE_ALT, syaw_deg)
            time.sleep(SETPOINT_PERIOD)

    def land(self):
        print('[LAND]')
        s      = self._state()
        x, y   = s['x'], s['y']
        yaw_d  = s['yaw']
        start_z = max(s['z'], 0.1)
        steps   = max(1, int(LAND_DURATION / SETPOINT_PERIOD))
        for i in range(1, steps + 1):
            if self._stop:
                break
            z = start_z * (1.0 - i / steps)
            self._send(x, y, max(z, 0.0), yaw_d)
            time.sleep(SETPOINT_PERIOD)
        self._stop_motors()

    # ── main mission loop ─────────────────────────────────────────────────────────

    def run_mission(self):
        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2.0)

        if not self._log_ready.wait(timeout=5.0):
            print('No state estimate received — aborting')
            return

        s = self._state()
        start_x   = s['x']
        start_y   = s['y']
        start_yaw = s['yaw']   # deg
        print(f'Start: x={start_x:.2f}  y={start_y:.2f}  yaw={start_yaw:.1f}°')

        self._hold_x   = start_x
        self._hold_y   = start_y
        self._hold_z   = CRUISE_ALT
        self._hold_yaw = start_yaw
        self._search_yaw_deg = start_yaw

        yaw_r = math.radians(start_yaw)
        self._circuit_center = np.array([
            start_x + CIRCUIT_OFFSET * math.cos(yaw_r),
            start_y + CIRCUIT_OFFSET * math.sin(yaw_r),
        ])
        print(f'Circuit centre: ({self._circuit_center[0]:.2f}, {self._circuit_center[1]:.2f})')

        self.takeoff(start_x, start_y, start_yaw)
        if self._stop:
            return

        mission_state = SEARCH
        print(f'[{SEARCH}]')

        try:
            while not self._stop:
                s   = self._state()
                det = self._detect()
                print('DET : ', det)
                # ── SEARCH ───────────────────────────────────────────────────────
                if mission_state == SEARCH:
                    self._search_yaw_deg   += SEARCH_YAW_RATE * SETPOINT_PERIOD
                    self._search_yaw_total += SEARCH_YAW_RATE * SETPOINT_PERIOD
                    self._hold_yaw          = self._search_yaw_deg

                    ''' # after a full 360° with no gate: reposition 1 m forqqward
                    if self._search_yaw_total >= 360.0:
                        print('[SEARCH] full rotation, repositioning 1 m forward')
                        yaw_r = math.radians(s['yaw'])
                        self._hold_x += 1.0 * math.cos(yaw_r)
                        self._hold_y += 1.0 * math.sin(yaw_r)
                        self._search_yaw_total = 0.0
                    '''
                    self._hold()

                    if det is not None:
                        status, quad_norm, cx_norm = det
                        if status == 'commit_to_pass':
                            self._force_pass(s)
                            mission_state = TRAVEL_GATE
                            print(f'[{TRAVEL_GATE}] from SEARCH commit')
                        elif status == 'ok':
                            if self._validate_detection(quad_norm, s) is not None:
                                gate_angle     = math.atan2(cx_norm, 1.0)
                                self._hold_yaw = math.degrees(math.radians(s['yaw']) - gate_angle)
                                self._frames_detected  = 0
                                self._lost_count       = 0
                                self._search_yaw_total = 0.0
                                mission_state = DETECT_1
                                print(f'[{DETECT_1}]')

                # ── DETECT_1 ─────────────────────────────────────────────────────
                elif mission_state == DETECT_1:
                    self._hold()

                    if det is not None:
                        status, quad_norm, cx_norm = det
                        if status == 'commit_to_pass':
                            self._force_pass(s)
                            mission_state = TRAVEL_GATE
                            print(f'[{TRAVEL_GATE}] from DETECT_1 commit')
                        elif status == 'ok':
                            validated = self._validate_detection(quad_norm, s)
                            if validated is None:
                                self._frames_detected = 0
                                self._lost_count += 1
                                if self._lost_count >= LOST_THRESHOLD:
                                    print(f'[{SEARCH}] gate rejected by filter during DETECT_1')
                                    self._search_yaw_total = 0.0
                                    mission_state = SEARCH
                            else:
                                self._lost_count = 0
                                gate_angle     = math.atan2(cx_norm, 1.0)
                                self._hold_yaw = math.degrees(math.radians(s['yaw']) - gate_angle)

                                speed = math.sqrt(s['vx']**2 + s['vy']**2 + s['vz']**2)
                                if speed < SPEED_THRESHOLD:
                                    self._frames_detected += 1
                                    if self._frames_detected >= REQ_FRAMES:
                                        self._P = np.array([s['x'], s['y'], s['z']])
                                        self._r_corners = self._corners_to_rays(quad_norm, s)
                                        self._frames_detected = 0

                                        yaw_r        = math.radians(s['yaw'])
                                        self._lat_x  = s['x'] + LATERAL_DIST *  math.sin(yaw_r)
                                        self._lat_y  = s['y'] + LATERAL_DIST * -math.cos(yaw_r)
                                        self._lat_z  = CRUISE_ALT
                                        mission_state = LATERAL_MOVE
                                        print(f'[{LATERAL_MOVE}] to ({self._lat_x:.2f},{self._lat_y:.2f})')
                                else:
                                    self._frames_detected = 0
                        else:   # no_gate
                            self._frames_detected = 0
                            self._lost_count += 1
                            if self._lost_count >= LOST_THRESHOLD:
                                print(f'[{SEARCH}] gate lost during DETECT_1')
                                self._search_yaw_total = 0.0
                                mission_state = SEARCH

                # ── LATERAL_MOVE ─────────────────────────────────────────────────
                elif mission_state == LATERAL_MOVE:
                    self._send(self._lat_x, self._lat_y, self._lat_z, self._hold_yaw)
                    if self._dist3(s, self._lat_x, self._lat_y, self._lat_z) < PASS_TOLERANCE:
                        self._hold_x          = self._lat_x
                        self._hold_y          = self._lat_y
                        self._frames_detected = 0
                        self._lost_count      = 0
                        mission_state = DETECT_2
                        print(f'[{DETECT_2}]')

                # ── DETECT_2 ─────────────────────────────────────────────────────
                elif mission_state == DETECT_2:
                    self._hold()

                    if det is not None:
                        status, quad_norm, cx_norm = det
                        if status == 'commit_to_pass':
                            self._force_pass(s)
                            mission_state = TRAVEL_GATE
                            print(f'[{TRAVEL_GATE}] from DETECT_2 commit')
                        elif status == 'ok':
                            validated = self._validate_detection(quad_norm, s)
                            if validated is None:
                                self._frames_detected = 0
                                self._lost_count += 1
                                if self._lost_count >= LOST_THRESHOLD:
                                    print(f'[{SEARCH}] gate rejected by filter during DETECT_2')
                                    self._search_yaw_total = 0.0
                                    mission_state = SEARCH
                            else:
                                self._lost_count = 0
                                gate_angle     = math.atan2(cx_norm, 1.0)
                                self._hold_yaw = math.degrees(math.radians(s['yaw']) - gate_angle)

                                speed = math.sqrt(s['vx']**2 + s['vy']**2 + s['vz']**2)
                                if speed < SPEED_THRESHOLD:
                                    self._frames_detected += 1
                                    if self._frames_detected >= REQ_FRAMES:
                                        self._Q         = np.array([s['x'], s['y'], s['z']])
                                        self._s_corners = self._corners_to_rays(quad_norm, s)
                                        self._frames_detected = 0
                                        mission_state = TRIANGULATE
                                        print(f'[{TRIANGULATE}]')
                                else:
                                    self._frames_detected = 0
                        else:   # no_gate
                            self._frames_detected = 0
                            self._lost_count += 1
                            if self._lost_count >= LOST_THRESHOLD:
                                print(f'[{SEARCH}] gate lost during DETECT_2')
                                self._search_yaw_total = 0.0
                                mission_state = SEARCH

                # ── TRIANGULATE ──────────────────────────────────────────────────
                elif mission_state == TRIANGULATE:
                    self._hold()   # keep drone stationary during computation
                    try:
                        corners_3d, H, gate_yaw = self._triangulate()
                        print(f'Gate {self._gate_count + 1}: centre=({H[0]:.2f},{H[1]:.2f},'
                              f'{H[2]:.2f})  yaw={math.degrees(gate_yaw):.1f}°')
                        self._set_gate_waypoints(H, gate_yaw, s)
                        mission_state = TRAVEL_GATE
                        print(f'[{TRAVEL_GATE}]')
                    except Exception as e:
                        print(f'Triangulation failed ({e}) — retrying search')
                        self._search_yaw_total = 0.0
                        mission_state = SEARCH

                # ── TRAVEL_GATE ──────────────────────────────────────────────────
                elif mission_state == TRAVEL_GATE:
                    gyd = math.degrees(self._gate_yaw)

                    if self._travel_phase == 0:
                        self._send(self._app_x, self._app_y, self._gate_z, gyd)
                        at_app = self._dist3(s, self._app_x, self._app_y, self._gate_z)
                        if at_app < PASS_TOLERANCE:
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
                            # reset search from current position
                            self._hold_x           = s['x']
                            self._hold_y           = s['y']
                            self._hold_yaw         = s['yaw']
                            self._search_yaw_deg   = s['yaw']
                            self._search_yaw_total = 0.0
                            if self._gate_count >= NUM_GATES:
                                mission_state = DONE
                                print(f'[{DONE}] all {NUM_GATES} gates passed — hovering')
                            else:
                                mission_state = SEARCH
                                print(f'[{SEARCH}] ({self._gate_count}/{NUM_GATES} done)')

                # ── DONE ─────────────────────────────────────────────────────────
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


# ── emergency stop ───────────────────────────────────────────────────────────────

def emergency_stop_listener(ctrl: Lap1Controller, cam: UdpVideoThread, cf: Crazyflie):
    """Press Q for controlled landing; press Q again to cut motors immediately."""
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
            print('\n[E-STOP] landing — press Q again to cut motors immediately')
            try:
                ctrl.land()
            except Exception as e:
                print(f'Land failed ({e}) — cutting motors')
                ctrl._stop_motors()
            finally:
                cam.stop()
                cf.close_link()
            return False
        elif stop_count[0] >= 2:
            print('\n[E-STOP] cutting motors immediately')
            ctrl._stop_motors()
            cam.stop()
            cf.close_link()
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# ── entry point ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cflib.crtp.init_drivers()

    cf              = Crazyflie(rw_cache='cache')
    connected_event = threading.Event()

    def on_connected(uri):
        print(f'Connected: {uri}')
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

    cam  = UdpVideoThread()
    cam.start()

    ctrl = Lap1Controller(cf, cam)
    if not ctrl.is_connected:
        print('Log setup failed — exiting')
        cam.stop()
        cf.close_link()
        exit(1)

    threading.Thread(
        target=emergency_stop_listener,
        args=(ctrl, cam, cf),
        daemon=True,
    ).start()

    try:
        ctrl.run_mission()
    finally:
        cam.stop()
        cf.close_link()
