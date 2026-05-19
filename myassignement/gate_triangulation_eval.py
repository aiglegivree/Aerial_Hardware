"""
gate_triangulation_eval.py  —  Manual gate-triangulation evaluator
==================================================================

The drone stays grounded or hand-held; you move it manually between two
poses while the tool displays live detections and accumulates triangulated
gate estimates.

Workflow
--------
  1.  Face a gate, press [1]  → capture View 1
  2.  Move ~0.5 m laterally,  press [2]  → capture View 2
  3.  Press [T]               → triangulate and display result
  4.  Press [R]               → reset views (keep triangulated history)
  5.  Press [C]               → clear all triangulated gates
  6.  Press [S]               → save screenshot
  7.  Press [Q] / ESC         → quit

GUI layout
----------
  [  Camera feed (2×)  |  Top-down map  ]
  [     Status bar                      ]
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
from scipy.spatial.transform import Rotation as R

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

from cv_detection import detect_gate, render_detection

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

# ── triangulation ────────────────────────────────────────────────────────────────

R_CAM_TO_BODY    = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
GATE_HEIGHT_REAL = 0.40   # m
GATE_HEIGHT_TOL  = 0.15   # m

# ── GUI constants ─────────────────────────────────────────────────────────────────

CAM_SCALE  = 2            # 2× upscale
CAM_PW     = CAM_WIDTH  * CAM_SCALE   # 648
CAM_PH     = CAM_HEIGHT * CAM_SCALE   # 488
MAP_W      = 500
MAP_H      = CAM_PH       # same height as camera panel
MAP_RANGE  = 5.0          # metres shown across full map width
MAP_PX_M   = MAP_W / MAP_RANGE
STATUS_H   = 140
WIN_W      = CAM_PW + MAP_W
WIN_NAME   = 'Gate Triangulation Evaluator'

GATE_PALETTE = [
    (255,  80, 255),   # magenta
    ( 80, 220, 255),   # cyan
    (255, 160,  80),   # orange
    ( 80, 255, 160),   # mint
    (255, 255,  80),   # yellow
]


# ── state buffer ─────────────────────────────────────────────────────────────────

class StateBuffer:
    def __init__(self, max_len=60):
        self._buf  = deque(maxlen=max_len)
        self._lock = threading.Lock()

    def push(self, t, state):
        with self._lock:
            self._buf.append((t, dict(state)))

    @property
    def latest(self):
        with self._lock:
            return dict(self._buf[-1][1]) if self._buf else None


# ── UDP video thread ──────────────────────────────────────────────────────────────

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
    def __init__(self):
        super().__init__(daemon=True, name='UdpVideo')
        self._lock     = threading.Lock()
        self._frame    = None
        self._frame_ts = 0.0
        self._running  = False

    def start(self):
        self._running = True
        super().start()

    def stop(self):
        self._running = False

    @property
    def latest(self):
        with self._lock:
            return self._frame, self._frame_ts

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(('0.0.0.0', UDP_LOCAL_PORT))
        sock.settimeout(1.0)
        sock.sendto(UDP_START_MAGIC, (UDP_AIDECK_IP, UDP_AIDECK_PORT))

        buf = bytearray()
        expected = 0
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
                _, w, h, _, _, size = struct.unpack('<BHHBBI', payload[:IMG_HEADER_SIZE])
                if w == CAM_WIDTH and h == CAM_HEIGHT and 0 < size < 65536:
                    expected  = size
                    buf       = bytearray()
                    receiving = True
                    continue
            if not receiving:
                continue
            buf.extend(payload)
            if len(buf) >= expected:
                frame = self._decode(buf)
                if frame is not None:
                    with self._lock:
                        self._frame    = frame
                        self._frame_ts = time.time()
                receiving = False

    def _decode(self, raw):
        soi = raw.find(b'\xff\xd8')
        eoi = raw.rfind(b'\xff\xd9')
        if soi < 0 or eoi <= soi:
            return None
        n = eoi + 2 - soi
        if n < MIN_JPEG_BYTES:
            return None
        jpeg = np.frombuffer(raw, np.uint8, count=n, offset=soi)
        with _muted_stderr():
            img = cv2.imdecode(jpeg, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[:2] != (CAM_HEIGHT, CAM_WIDTH):
            return None
        if img.ndim == 2:
            return img
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return None


# ── triangulation helpers ─────────────────────────────────────────────────────────

def _corners_to_rays(quad_norm, s):
    """Undistorted normalised corners (4,2) → world-frame unit rays (4,3)."""
    Rb2w = R.from_quat([s['qx'], s['qy'], s['qz'], s['qw']]).as_matrix()
    rays = np.zeros((4, 3))
    for i, (xn, yn) in enumerate(quad_norm):
        v = np.array([xn, yn, 1.0])
        r = Rb2w @ (R_CAM_TO_BODY @ v)
        rays[i] = r / np.linalg.norm(r)
    return rays


def _triangulate(P, r_corners, Q, s_corners):
    """
    Two-view triangulation.  Returns (corners_3d (4,3), center (3,), gate_yaw_rad).
    Raises ValueError if the reconstructed gate height is implausible.
    """
    corners_3d = np.zeros((4, 3))
    for i in range(4):
        A = np.column_stack([r_corners[i], -s_corners[i]])
        b = Q - P
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        lam, mu = float(sol[0]), float(sol[1])
        corners_3d[i] = (P + lam * r_corners[i] + Q + mu * s_corners[i]) / 2.0

    h_l = np.linalg.norm(corners_3d[3] - corners_3d[0])
    h_r = np.linalg.norm(corners_3d[2] - corners_3d[1])
    h   = (h_l + h_r) / 2.0
    if abs(h - GATE_HEIGHT_REAL) > GATE_HEIGHT_TOL:
        raise ValueError(
            f'height {h:.3f} m outside '
            f'{GATE_HEIGHT_REAL - GATE_HEIGHT_TOL:.2f}–'
            f'{GATE_HEIGHT_REAL + GATE_HEIGHT_TOL:.2f} m'
        )

    center  = np.mean(corners_3d, axis=0)
    v_width = corners_3d[1] - corners_3d[0]
    yaw     = math.atan2(-v_width[1], v_width[0])
    return corners_3d, center, yaw


# ── map rendering ─────────────────────────────────────────────────────────────────

def _w2m(wx, wy, cx, cy):
    """World (x,y) → map pixel (col, row) with drone at map centre."""
    px = int(MAP_W / 2 + (wx - cx) * MAP_PX_M)
    py = int(MAP_H / 2 - (wy - cy) * MAP_PX_M)
    return px, py


def _draw_arrow(img, px, py, yaw_r, length=18, color=(255, 255, 255)):
    tip   = (int(px + length       * math.cos(yaw_r)),
             int(py - length       * math.sin(yaw_r)))
    left  = (int(px + length * 0.4 * math.cos(yaw_r + 2.5)),
             int(py - length * 0.4 * math.sin(yaw_r + 2.5)))
    right = (int(px + length * 0.4 * math.cos(yaw_r - 2.5)),
             int(py - length * 0.4 * math.sin(yaw_r - 2.5)))
    pts = np.array([tip, left, right], np.int32)
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, (255, 255, 255), 1)


def render_map(state, view1, view2, gates):
    img = np.full((MAP_H, MAP_W, 3), 28, np.uint8)
    cx, cy = (state['x'], state['y']) if state else (0.0, 0.0)

    # grid every 0.5 m
    step = 0.5
    half = MAP_RANGE / 2
    for k in range(int(-half / step) - 1, int(half / step) + 2):
        wx = cx + k * step
        wy = cy + k * step
        cv2.line(img, _w2m(wx, cy - half, cx, cy),
                      _w2m(wx, cy + half, cx, cy), (52, 52, 52), 1)
        cv2.line(img, _w2m(cx - half, wy, cx, cy),
                      _w2m(cx + half, wy, cx, cy), (52, 52, 52), 1)
        if abs(round(k * step, 1)) % 1.0 < 1e-6:   # label every 1 m
            lp = _w2m(wx, cy + half - 0.15, cx, cy)
            cv2.putText(img, f'{wx:.0f}', (lp[0] - 12, lp[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (90, 90, 90), 1)
            lp2 = _w2m(cx - half + 0.05, wy, cx, cy)
            cv2.putText(img, f'{wy:.0f}', (lp2[0], lp2[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (90, 90, 90), 1)

    # world origin marker
    op = _w2m(0.0, 0.0, cx, cy)
    if 0 <= op[0] < MAP_W and 0 <= op[1] < MAP_H:
        cv2.drawMarker(img, op, (90, 90, 90), cv2.MARKER_CROSS, 10, 1)

    # triangulated gates
    for idx, g in enumerate(gates):
        col  = GATE_PALETTE[idx % len(GATE_PALETTE)]
        pts  = [_w2m(c[0], c[1], cx, cy) for c in g['corners']]
        for i in range(4):
            cv2.line(img, pts[i], pts[(i + 1) % 4], col, 2)
        cp = _w2m(g['center'][0], g['center'][1], cx, cy)
        cv2.circle(img, cp, 6, col, -1)
        cv2.circle(img, cp, 6, (255, 255, 255), 1)
        # gate normal arrow
        gyw = g['yaw']
        np_ = _w2m(g['center'][0] + 0.35 * math.cos(gyw),
                   g['center'][1] + 0.35 * math.sin(gyw), cx, cy)
        cv2.arrowedLine(img, cp, np_, col, 1, tipLength=0.35)
        cv2.putText(img, str(idx + 1), (cp[0] + 8, cp[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # view 1
    if view1 is not None:
        p1 = _w2m(view1['pos'][0], view1['pos'][1], cx, cy)
        cv2.circle(img, p1, 9, (0, 210, 80), -1)
        cv2.circle(img, p1, 9, (255, 255, 255), 1)
        yr = math.radians(view1['yaw'])
        tp = _w2m(view1['pos'][0] + 0.35 * math.cos(yr),
                  view1['pos'][1] + 0.35 * math.sin(yr), cx, cy)
        cv2.arrowedLine(img, p1, tp, (0, 210, 80), 2, tipLength=0.35)
        cv2.putText(img, 'V1', (p1[0] + 11, p1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 210, 80), 1, cv2.LINE_AA)

    # view 2
    if view2 is not None:
        p2 = _w2m(view2['pos'][0], view2['pos'][1], cx, cy)
        cv2.circle(img, p2, 9, (0, 140, 255), -1)
        cv2.circle(img, p2, 9, (255, 255, 255), 1)
        yr = math.radians(view2['yaw'])
        tp = _w2m(view2['pos'][0] + 0.35 * math.cos(yr),
                  view2['pos'][1] + 0.35 * math.sin(yr), cx, cy)
        cv2.arrowedLine(img, p2, tp, (0, 140, 255), 2, tipLength=0.35)
        cv2.putText(img, 'V2', (p2[0] + 11, p2[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 140, 255), 1, cv2.LINE_AA)

    # baseline line + distance
    if view1 is not None and view2 is not None:
        p1 = _w2m(view1['pos'][0], view1['pos'][1], cx, cy)
        p2 = _w2m(view2['pos'][0], view2['pos'][1], cx, cy)
        cv2.line(img, p1, p2, (160, 160, 160), 1)
        bl = np.linalg.norm(np.array(view2['pos']) - np.array(view1['pos']))
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(img, f'{bl:.2f} m', (mid[0] + 4, mid[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # drone (always at centre)
    if state is not None:
        yr = math.radians(state['yaw'])
        _draw_arrow(img, MAP_W // 2, MAP_H // 2, yr, color=(0, 220, 255))
        cv2.putText(img,
                    f"({state['x']:.2f},{state['y']:.2f})",
                    (MAP_W // 2 + 22, MAP_H // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1, cv2.LINE_AA)

    # 1-m scale bar
    bx, by = 10, MAP_H - 14
    cv2.line(img, (bx, by), (bx + int(MAP_PX_M), by), (200, 200, 200), 2)
    cv2.putText(img, '1 m', (bx, by - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(img, 'Top-down (drone = cyan arrow)',
                (MAP_W // 2 - 85, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)
    return img


# ── status bar ───────────────────────────────────────────────────────────────────

def render_status(state, view1, view2, det_status, gates, msg, msg_col):
    img = np.full((STATUS_H, WIN_W, 3), 22, np.uint8)
    cv2.line(img, (0, 0), (WIN_W, 0), (65, 65, 65), 1)

    def put(text, x, y, scale=0.44, color=(200, 200, 200), thick=1):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thick, cv2.LINE_AA)

    # row 1 — drone state
    if state is not None:
        put(f"Drone  x={state['x']:+.3f}  y={state['y']:+.3f}  "
            f"z={state['z']:.3f}  yaw={state['yaw']:+.1f}°",
            10, 22, color=(0, 220, 255))
        spd = math.sqrt(state['vx'] ** 2 + state['vy'] ** 2 + state['vz'] ** 2)
        put(f"speed {spd:.3f} m/s", 10, 40, 0.38, (140, 200, 200))
    else:
        put('Drone: waiting for Lighthouse log...', 10, 22, color=(100, 100, 200))

    # detection badge
    det_col = {
        'ok':             (80,  210, 80),
        'no_gate':        (80,   80, 200),
        'no_corners':     (200, 200, 50),
        'commit_to_pass': (255, 140,  0),
    }.get(det_status, (130, 130, 130))
    put(f'Detection: {det_status or "---"}', WIN_W - 260, 22, color=det_col)

    # row 2 — view captures
    v1c = (0, 210,  80) if view1 else (100, 100, 100)
    v2c = (0, 140, 255) if view2 else (100, 100, 100)
    v1t = (f"V1 @ ({view1['pos'][0]:+.3f},{view1['pos'][1]:+.3f},"
           f"{view1['pos'][2]:.2f})")  if view1 else 'V1: not captured'
    v2t = (f"V2 @ ({view2['pos'][0]:+.3f},{view2['pos'][1]:+.3f},"
           f"{view2['pos'][2]:.2f})")  if view2 else 'V2: not captured'
    put(v1t, 10,  62, color=v1c)
    put(v2t, 10,  82, color=v2c)
    if view1 and view2:
        bl = np.linalg.norm(np.array(view2['pos']) - np.array(view1['pos']))
        bl_ok = 0.3 <= bl <= 0.8
        put(f'baseline = {bl:.3f} m{"  OK" if bl_ok else "  (aim for 0.4-0.6 m)"}',
            WIN_W - 300, 62, color=(80, 210, 80) if bl_ok else (200, 180, 50))

    put(f'Gates triangulated: {len(gates)}', 10, 103, color=(200, 170, 255))

    # per-gate detail
    for i, g in enumerate(gates):
        col = GATE_PALETTE[i % len(GATE_PALETTE)]
        hl  = np.linalg.norm(g['corners'][3] - g['corners'][0])
        hr  = np.linalg.norm(g['corners'][2] - g['corners'][1])
        h   = (hl + hr) / 2.0
        txt = (f"G{i+1}: ctr=({g['center'][0]:+.2f},{g['center'][1]:+.2f},"
               f"{g['center'][2]:.2f})  h={h:.3f}m  yaw={math.degrees(g['yaw']):.1f}°")
        put(txt, 10 + i * 370, 120, 0.37, col)

    # message row
    put(msg, 10, STATUS_H - 16, 0.42, msg_col)

    # key hints
    put('[1] View1  [2] View2  [T] Triangulate  '
        '[R] Reset views  [C] Clear all  [S] Save  [Q/ESC] Quit',
        10, STATUS_H - 3, 0.35, (110, 110, 110))
    return img


# ── main evaluator ────────────────────────────────────────────────────────────────

class GateEvaluator:
    def __init__(self, cf, cam):
        self._cf        = cf
        self._cam       = cam
        self._state_buf = StateBuffer()

        self._log = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'yaw': 0.0,
            'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0,
        }
        self._log_lock  = threading.Lock()
        self._log_ready = threading.Event()
        self.is_connected = False
        self._setup_log()

        self._view1  = None
        self._view2  = None
        self._gates  = []

        self._msg       = 'Move drone to face a gate, press [1] to capture View 1'
        self._msg_color = (200, 200, 200)
        self._last_ts   = 0.0
        self._last_result = None

    # ── log ──────────────────────────────────────────────────────────────────────

    def _setup_log(self):
        lg = LogConfig(name='EvalState', period_in_ms=50)
        for v in ['stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
                  'stateEstimate.vx', 'stateEstimate.vy', 'stateEstimate.vz',
                  'stabilizer.yaw',
                  'stateEstimate.qx', 'stateEstimate.qy',
                  'stateEstimate.qz', 'stateEstimate.qw']:
            lg.add_variable(v, 'float')
        try:
            self._cf.log.add_config(lg)
            lg.data_received_cb.add_callback(self._log_cb)
            lg.start()
            self.is_connected = True
        except Exception as e:
            print(f'Log setup failed: {e}')

    def _log_cb(self, timestamp, data, logconf):
        with self._log_lock:
            self._log['x']   = data['stateEstimate.x']
            self._log['y']   = data['stateEstimate.y']
            self._log['z']   = data['stateEstimate.z']
            self._log['vx']  = data['stateEstimate.vx']
            self._log['vy']  = data['stateEstimate.vy']
            self._log['vz']  = data['stateEstimate.vz']
            self._log['yaw'] = data['stabilizer.yaw']
            self._log['qx']  = data['stateEstimate.qx']
            self._log['qy']  = data['stateEstimate.qy']
            self._log['qz']  = data['stateEstimate.qz']
            self._log['qw']  = data['stateEstimate.qw']
            snap = dict(self._log)
        self._state_buf.push(time.time(), snap)
        self._log_ready.set()

    def _state(self):
        with self._log_lock:
            return dict(self._log)

    # ── actions ──────────────────────────────────────────────────────────────────

    def _capture(self, view_num):
        if self._last_result is None or self._last_result['status'] != 'ok':
            self._msg       = f'No valid detection — cannot capture View {view_num}'
            self._msg_color = (80, 80, 220)
            return
        s = self._state() if self._log_ready.is_set() else None
        if s is None:
            self._msg       = 'No drone state — cannot capture'
            self._msg_color = (80, 80, 220)
            return

        pos  = np.array([s['x'], s['y'], s['z']])
        rays = _corners_to_rays(self._last_result['quad_norm'], s)
        view = {'pos': pos, 'rays': rays, 'yaw': s['yaw']}

        if view_num == 1:
            self._view1     = view
            self._msg       = (f'View 1 captured ({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.2f}) '
                               '— move ~0.5 m laterally then press [2]')
            self._msg_color = (0, 210, 80)
        else:
            self._view2     = view
            self._msg       = (f'View 2 captured ({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.2f}) '
                               '— press [T] to triangulate')
            self._msg_color = (0, 140, 255)

        print(f'[CAPTURE] View {view_num} at ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})')

    def _triangulate(self):
        if self._view1 is None or self._view2 is None:
            self._msg       = 'Need both View 1 and View 2 before triangulating'
            self._msg_color = (80, 80, 220)
            return
        try:
            corners, center, yaw = _triangulate(
                self._view1['pos'], self._view1['rays'],
                self._view2['pos'], self._view2['rays'],
            )
            hl = np.linalg.norm(corners[3] - corners[0])
            hr = np.linalg.norm(corners[2] - corners[1])
            h  = (hl + hr) / 2.0
            self._gates.append({'corners': corners, 'center': center, 'yaw': yaw})
            self._msg = (
                f'Gate {len(self._gates)} OK — '
                f'centre=({center[0]:+.3f},{center[1]:+.3f},{center[2]:.3f})  '
                f'height={h:.3f} m  yaw={math.degrees(yaw):.1f}°'
            )
            self._msg_color = (80, 210, 80)
            self._view1 = None
            self._view2 = None
            print(f'[TRIANGULATE] Gate {len(self._gates)} at '
                  f'({center[0]:.3f},{center[1]:.3f},{center[2]:.3f})')
        except ValueError as e:
            self._msg       = f'Triangulation failed: {e}'
            self._msg_color = (80, 80, 220)
            print(f'[TRIANGULATE] FAILED — {e}')

    # ── main loop ─────────────────────────────────────────────────────────────────

    def run(self):
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, WIN_W, CAM_PH + STATUS_H)

        print('Resetting Kalman estimator — hold drone still for 2 s...')
        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2.0)

        print('Waiting for Lighthouse state estimate...', end='', flush=True)
        got = self._log_ready.wait(timeout=5.0)
        print(' ready.' if got else ' timed out (no log — state display disabled)')

        while True:
            s = self._state() if self._log_ready.is_set() else None

            # ── detection ─────────────────────────────────────────────────────
            frame, ts = self._cam.latest
            if frame is not None and ts != self._last_ts:
                self._last_ts     = ts
                self._last_result = detect_gate(frame)

            # ── camera panel (render_detection returns RGB) ────────────────────
            if frame is not None and self._last_result is not None:
                cam_rgb = render_detection(frame, self._last_result)
                cam_bgr = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)
            elif frame is not None:
                cam_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                cam_bgr = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint8)
                cv2.putText(cam_bgr, 'No camera feed', (50, CAM_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 200), 1)
            cam_bgr = cv2.resize(cam_bgr, (CAM_PW, CAM_PH), interpolation=cv2.INTER_NEAREST)

            # ── map panel ─────────────────────────────────────────────────────
            map_bgr = render_map(s, self._view1, self._view2, self._gates)

            # ── assemble ──────────────────────────────────────────────────────
            top = np.hstack([cam_bgr, map_bgr])
            det_status = self._last_result['status'] if self._last_result else None
            bar = render_status(s, self._view1, self._view2,
                                det_status, self._gates,
                                self._msg, self._msg_color)
            display = np.vstack([top, bar])

            cv2.imshow(WIN_NAME, display)

            # ── key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(40) & 0xFF   # 25 fps

            if key in (ord('q'), ord('Q'), 27):
                break
            elif key == ord('1'):
                self._capture(1)
            elif key == ord('2'):
                self._capture(2)
            elif key in (ord('t'), ord('T')):
                self._triangulate()
            elif key in (ord('r'), ord('R')):
                self._view1 = None
                self._view2 = None
                self._msg       = 'Views reset — press [1] to capture View 1'
                self._msg_color = (200, 200, 200)
                print('[RESET] views cleared')
            elif key in (ord('c'), ord('C')):
                self._gates.clear()
                self._msg       = 'All gates cleared'
                self._msg_color = (200, 200, 200)
                print('[CLEAR] all gates removed')
            elif key in (ord('s'), ord('S')):
                fname = f'trieval_{int(time.time())}.png'
                cv2.imwrite(fname, display)
                self._msg       = f'Screenshot saved: {fname}'
                self._msg_color = (255, 200, 60)
                print(f'[SAVE] {fname}')

        cv2.destroyAllWindows()


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

    print(f'Connecting to {CONTROL_URI} ...')
    cf.open_link(CONTROL_URI)

    if not connected_event.wait(timeout=10):
        print('Connection timed out — exiting')
        exit(1)

    cam = UdpVideoThread()
    cam.start()

    ev = GateEvaluator(cf, cam)
    if not ev.is_connected:
        print('Warning: log setup failed — drone state unavailable')

    try:
        ev.run()
    finally:
        cam.stop()
        cf.close_link()
