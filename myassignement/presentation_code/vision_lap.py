# -*- coding: utf-8 -*-
"""
Part 1: vision-only gate-racing lap.

Two links:
    Control/logs via Crazyradio (CRTP)
    Video via AI-deck UDP stream (JPEG)

Per-gate state machine: SEARCH -> LOCK -> APPROACH -> TRANSIT.
APPROACH is image-based visual servoing: pixel errors on the detected gate
drive vy (strafe), z (climb), and vx (forward).

Lighthouse pose is only used to keep the drone inside the arena bounds.

Keys: 'q' = motor cut, ESC = controlled landing.
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

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

# Gate-size thresholds in pixels (long side of bounding rect)
GATE_SIZE_MIN   = 30    # below this -> reject as noise
GATE_SIZE_CLOSE = 180   # above this -> close enough, trigger TRANSIT

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
# Runs in a separate process so its GUI never stalls the control loop.
FPV_ENABLED  = True   # show live camera window with detection overlay
FPV_SCALE    = 2      # upscale factor for the display window

CPX_HEADER_SIZE  = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE  = 11
MIN_JPEG_BYTES   = 1000

# ── arena bounds (Lighthouse world frame) ──────────────────────────────────────
ARENA_X_MIN = -1.0   # m, back wall
ARENA_X_MAX = +3.0   # m, front wall
ARENA_Y_MIN = -0.9   # m, right wall
ARENA_Y_MAX = +0.9   # m, left wall

SAFETY_MARGIN_HARD = 0.00000  # m, hard wall margin

# ── flight ─────────────────────────────────────────────────────────────────────

CRUISE_ALT       = 1.25  # m, cruise altitude
TAKEOFF_DURATION = 3.0   # s
LAND_DURATION    = 3.0   # s

# IBVS: three independent P-controllers, one per body axis.
# Each axis takes a pixel error and outputs a velocity (or z offset).
KP_VX        = 0.005  # size error  (GATE_SIZE_CLOSE - size) -> forward speed
KP_VY        = 0.004  # lateral pixel error (cx - cx_mid)    -> strafe speed
KP_VZ        = 0.004  # vertical pixel error (cy_mid - cy)   -> altitude delta
MAX_VX       = 0.10   # m/s, forward cap (never backward)
MAX_VY       = 0.15   # m/s, strafe cap
MAX_VZ_DELTA = 0.4    # m, altitude adjustment cap

# Pixel-alignment tolerance required before APPROACH hands off to TRANSIT.
ALIGN_TOL_X      = 25   # px
ALIGN_TOL_Y      = 25   # px
# vx is throttled down when the gate is off-centre, so the drone strafes/
# climbs into alignment before flying forward.
ALIGN_SCALE_DENOM = 0.6  # fraction of half-frame at which vx -> 0

TRANSIT_VX   = 0.20   # m/s, open-loop forward push through the gate
TRANSIT_TIME = 1.8    # s

SEARCH_YAW_RATE = 12.0  # deg/s
SEARCH_TIMEOUT  = 15.0  # s

# LOCK: hover-and-confirm window after a SEARCH hit, to reject single-frame flukes.
LOCK_DURATION   = 1.0   # s
LOCK_MIN_HITS   = 1     # unique-frame detections required during the window

LOST_TIMEOUT_S  = 1.5   # s without a fresh detection -> APPROACH falls back to SEARCH

DETECT_EMA_ALPHA = 0.5  # EMA weight on the newest detection (1.0 = no smoothing)

N_GATES = 5


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

    def stop(self):
        # daemon thread; exits with the process
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
        if img is None:
            return
        with self._lock:
            self._frame = img


# ── FPV viewer (separate process) ────────────────────────────────────────────

def _fpv_viewer_process(q, scale, cam_w, cam_h, gate_size_close):
    """Render (frame, detection, state) tuples. None message = shutdown."""
    import cv2 as _cv2
    import numpy as _np

    win = 'Crazyflie FPV'
    _cv2.namedWindow(win, _cv2.WINDOW_NORMAL)
    _cv2.resizeWindow(win, cam_w * scale, cam_h * scale)
    while True:
        try:
            msg = q.get(timeout=0.5)
        except Exception:
            if (_cv2.waitKey(1) & 0xFF) == ord('Q'):
                break
            continue
        if msg is None:
            break
        frame, det, state = msg
        disp = frame.copy() if frame.ndim == 3 else _cv2.cvtColor(frame, _cv2.COLOR_GRAY2BGR)

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
    """Fill dark regions enclosed by white, so a closed gate outline becomes
    a solid rectangle for the contour shape tests."""
    flood = mask.copy()
    h, w = mask.shape
    scratch = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, scratch, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


# Gates are bright glowing LED frames: threshold on high V, low-to-mid S, any hue.
GATE_HSV_LOWER = np.array([0,   0,   200], dtype=np.uint8)
GATE_HSV_UPPER = np.array([255, 100, 255], dtype=np.uint8)


def get_gate_detection(frame):
    """HSV-threshold the frame, then pick the most rectangle-like contour.
    Returns (cx, cy, size) or (None, None, None)."""
    if frame is None:
        return None, None, None

    if len(frame.shape) == 3:
        bgr = frame
    else:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GATE_HSV_LOWER, GATE_HSV_UPPER)

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

        short_side, long_side = sorted([rw, rh])
        aspect = short_side / long_side
        if aspect < 0.45:
            continue

        rotated_area = rw * rh
        rectangularity = area / rotated_area
        if rectangularity < 0.80:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        if solidity < 0.85:
            continue

        score = rectangularity * solidity * np.log1p(area)
        if score > best_score:
            best_score = score
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

        # Camera runs at ~3 fps, control loop at 20 Hz: track frame identity
        # so we only act on fresh detections, not repeats.
        self._last_frame_obj    = None
        self._ema_cx            = None
        self._ema_cy            = None
        self._ema_size          = None

        # Latest detection (for the FPV overlay) and the queue to the viewer.
        self.last_detection = None
        self.last_detection_miss = False
        self._fpv_q = None

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

    # ── detection polling ────────────────────────────────────────────────────

    def _poll_detection(self):
        """Returns (cx, cy, size, status). status is:
          'new_hit'  fresh frame, gate detected (EMA-smoothed values)
          'new_miss' fresh frame, no gate
          'stale'    same frame object as last poll
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

        # EMA smoothing to damp single-frame outliers.
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
        """Non-blocking send to the FPV viewer; drops if the queue is full."""
        if self._fpv_q is None:
            return
        state = (self._state['x'], self._state['y'],
                 self._state['z'], self._state['yaw'])
        try:
            self._fpv_q.put_nowait((frame, det, state))
        except queue_mod.Full:
            pass

    def _reset_detection_filter(self):
        """Clear EMA state between gates / phases."""
        self._ema_cx = None
        self._ema_cy = None
        self._ema_size = None

    def _log_cb(self, timestamp, data, logconf):
        self._state['x']   = data['stateEstimate.x']
        self._state['y']   = data['stateEstimate.y']
        self._state['z']   = data['stateEstimate.z']
        self._state['yaw'] = data['stabilizer.yaw']

    # ── boundary-safe hover ──────────────────────────────────────────────────

    def _safe_hover(self, vx=0.0, vy=0.0, yaw_rate=0.0, z=CRUISE_ALT):
        """Hover setpoint, with outward velocity zeroed at the arena bounds."""
        x   = self._state['x']
        y   = self._state['y']
        yaw = math.radians(self._state['yaw'])

        # body -> world, so we can compare against world-frame arena bounds
        wx = vx * math.cos(yaw) - vy * math.sin(yaw)
        wy = vx * math.sin(yaw) + vy * math.cos(yaw)

        if x <= ARENA_X_MIN + SAFETY_MARGIN_HARD and wx < 0:
            wx = 0.0
        if x >= ARENA_X_MAX - SAFETY_MARGIN_HARD and wx > 0:
            wx = 0.0
        if y <= ARENA_Y_MIN + SAFETY_MARGIN_HARD and wy < 0:
            wy = 0.0
        if y >= ARENA_Y_MAX - SAFETY_MARGIN_HARD and wy > 0:
            wy = 0.0

        # world -> body, the Crazyflie's hover setpoint takes body-frame vx/vy
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
        """Yaw back and forth with a sinusoidal rate until a gate is detected."""
        print('  [SEARCH] Sweeping...')
        self._reset_detection_filter()
        t_start = time.time()
        tick = 0
        frames_seen = 0

        while not self._stop:
            cx, cy, size, status = self._poll_detection()
            if status != 'stale':
                frames_seen += 1
            if status == 'new_hit':
                print(f'  [SEARCH] gate found  cx={cx:.0f}  cy={cy:.0f}  size={size:.0f}px '
                      f'after {time.time() - t_start:.1f}s, yaw={self._state["yaw"]:+.0f}deg '
                      f'(frames_seen={frames_seen})')
                return

            elapsed = time.time() - t_start

            # sinusoidal yaw with 10 s period: sweeps a wide arc instead of just spinning
            sweep_yaw_rate = math.sin(elapsed * math.pi / 5) * SEARCH_YAW_RATE
            self._safe_hover(yaw_rate=sweep_yaw_rate, z=CRUISE_ALT)

            tick += 1
            if tick % 10 == 0:  # every ~1 s at 0.1 s loop
                fps = frames_seen / max(elapsed, 0.001)
                print(f'  [SEARCH] tick={tick} t={elapsed:.1f}s fps={fps:.1f} '
                      f'yaw={self._state["yaw"]:+.0f}deg yaw_rate={sweep_yaw_rate:+.1f}deg/s '
                      f'pos=({self._state["x"]:+.2f},{self._state["y"]:+.2f},{self._state["z"]:.2f})')

            time.sleep(0.1)

    # ── state: LOCK ──────────────────────────────────────────────────────────

    def lock_on_gate(self):
        """Hover in place and require LOCK_MIN_HITS unique-frame detections
        within LOCK_DURATION to confirm. Returns True if confirmed."""
        print(f'  [LOCK] holding {LOCK_DURATION:.1f}s to confirm detection '
              f'(need {LOCK_MIN_HITS} total UNIQUE-FRAME hits, non-consecutive)')
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
        """IBVS toward the gate.
        Three pixel errors drive three velocity commands:
            cx - cx_mid          -> vy   (strafe)
            cy_mid - cy          -> dz   (climb)
            GATE_SIZE_CLOSE - sz -> vx   (forward, never backward)
        Returns True on success (-> TRANSIT), False if the gate is lost (-> SEARCH).
        """
        print('  [APPROACH] IBVS toward gate')
        cx_mid   = CAM_WIDTH  / 2.0   # 162 px
        cy_mid   = CAM_HEIGHT / 2.0   # 122 px
        target_z = CRUISE_ALT
        tick     = 0
        t_start  = time.time()
        t_last_hit = time.time()  # wall-clock timestamp of last fresh hit
        frames_seen = 0

        # Last commanded velocities, re-sent on stale ticks so motion stays
        # smooth between the sparse camera frames.
        last_vx = 0.0
        last_vy = 0.0

        while not self._stop:
            cx, cy, size, status = self._poll_detection()
            tick += 1

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

            frames_seen += 1

            if status == 'new_miss':
                age = time.time() - t_last_hit
                if age > LOST_TIMEOUT_S:
                    print(f'  [APPROACH] gate lost {age:.2f}s → SEARCH '
                          f'(approach lasted {time.time() - t_start:.1f}s, '
                          f'fresh_frames={frames_seen})')
                    return False
                self._safe_hover(vx=last_vx, vy=last_vy, z=target_z)
                time.sleep(0.1)
                continue

            # status == 'new_hit'
            t_last_hit = time.time()

            e_x = -cx + cx_mid            # +ve: gate left of centre (strafe left)
            e_y = cy_mid - cy             # +ve: gate above centre (climb)
            e_z = GATE_SIZE_CLOSE - size  # +ve: gate too small (move forward)

            size_ok  = size >= GATE_SIZE_CLOSE
            align_ok = abs(e_x) < ALIGN_TOL_X and abs(e_y) < ALIGN_TOL_Y
            if size_ok and align_ok:
                print(f'  [APPROACH] DONE  size={size:.0f}px>={GATE_SIZE_CLOSE} '
                      f'ex={e_x:+.0f}<{ALIGN_TOL_X} ey={e_y:+.0f}<{ALIGN_TOL_Y} '
                      f'after {time.time() - t_start:.1f}s → TRANSIT')
                return True

            v_strafe  = KP_VY * e_x
            v_climb   = KP_VZ * e_y
            v_forward = KP_VX * e_z

            v_strafe  = clamp(v_strafe,  -MAX_VY,  MAX_VY)
            v_climb   = clamp(v_climb,   -MAX_VZ_DELTA, MAX_VZ_DELTA)
            v_forward = clamp(v_forward,  0.0,     MAX_VX)   # never fly backward

            # Off-centre gates throttle forward speed: strafe/climb in first.
            mis_x = abs(e_x) / cx_mid   # 0 = centred, 1 = at frame edge
            mis_y = abs(e_y) / cy_mid
            align_factor = clamp(1.0 - (mis_x + mis_y) / ALIGN_SCALE_DENOM, 0.0, 1.0)
            v_forward *= align_factor

            target_z  = clamp(CRUISE_ALT + v_climb,
                              CRUISE_ALT - MAX_VZ_DELTA,
                              CRUISE_ALT + MAX_VZ_DELTA)

            if tick % 10 == 0:
                sx = 'OK' if size_ok else '--'
                ax = 'OK' if align_ok else '--'
                fps = frames_seen / max(time.time() - t_start, 0.001)
                print(f'  [IBVS] t={tick:4d} f={frames_seen} fps={fps:.1f}  '
                      f'cx={cx:6.1f} cy={cy:6.1f} sz={size:5.1f}px '
                      f'[{sx}|{ax}]  ex={e_x:+5.0f} ey={e_y:+5.0f} ez={e_z:+5.0f}  '
                      f'align={align_factor:.2f}  vx={v_forward:.3f} vy={v_strafe:+.3f} '
                      f'z={target_z:.2f}  pos=({self._state["x"]:+.2f},{self._state["y"]:+.2f})')

            last_vx, last_vy = v_forward, v_strafe
            self._safe_hover(vx=v_forward, vy=v_strafe, z=target_z)
            time.sleep(0.1)

        return False

    # ── state: TRANSIT ───────────────────────────────────────────────────────

    def transit_gate(self):
        """Push forward through the gate at TRANSIT_VX for TRANSIT_TIME.
        While the gate is still visible, keep correcting vy/z to avoid clipping."""
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
                self._safe_hover(vx=TRANSIT_VX, z=target_z)
            else:
                stale += 1
                self._safe_hover(vx=TRANSIT_VX, vy=last_vy, z=target_z)
            time.sleep(0.1)
        print(f'  [TRANSIT] done (seen={seen} miss={miss} stale={stale} ticks, '
              f'pos=({self._state["x"]:+.2f},{self._state["y"]:+.2f},{self._state["z"]:.2f}))')

    # ── mission: vision ─────────────────────────────────────────────────────

    def run_vision_lap(self):
        # Reset the onboard EKF so the world frame is aligned to the current pose.
        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2)

        try:
            self.takeoff(target_z=CRUISE_ALT)

            for gate_idx in range(N_GATES):
                if self._stop:
                    break

                print(f'\n=== Gate {gate_idx + 1} / {N_GATES} ===')
                self._reset_detection_filter()

                # SEARCH + LOCK until the detection is stable
                while not self._stop:
                    self.search_for_gate()
                    if self._stop or self.lock_on_gate():
                        break

                if self._stop:
                    break

                # APPROACH + TRANSIT, falling back to SEARCH if the gate is lost
                while not self._stop:
                    if self.approach_gate():
                        self.transit_gate()
                        break
                    print('  gate lost — searching again')
                    while not self._stop:
                        self.search_for_gate()
                        if self._stop or self.lock_on_gate():
                            break

            msg = 'All gates complete' if not self._stop else 'Emergency stop'
            print(f'\n{msg} — landing')

        except Exception as e:
            print(f'\nUnhandled exception during mission: {e} — landing now')

        finally:
            try:
                self.land()
            except Exception as e:
                print(f'Land failed ({e}) — cutting motors')
                self._stop_motors()


# ── emergency stop ─────────────────────────────────────────────────────────────

def emergency_stop_listener(ctrl: GateController, cam: UdpVideoThread, cf: Crazyflie):
    """'q' cuts motors immediately. ESC lands first."""

    def on_press(key):
        # ESC: controlled landing
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

        # Q: immediate motor cut
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

def main():
    # Start the camera thread before opening the radio link, so the AI-deck
    # starts streaming while the Crazyradio connection is being set up.
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

    # FPV window runs in a separate process so its GUI cannot stall the control loop.
    fpv_proc = None
    fpv_q = None
    if FPV_ENABLED:
        fpv_q = mp.Queue(maxsize=1)   # latest-wins: controller drops if full
        ctrl._fpv_q = fpv_q
        fpv_proc = mp.Process(
            target=_fpv_viewer_process,
            args=(fpv_q, FPV_SCALE, CAM_WIDTH, CAM_HEIGHT, GATE_SIZE_CLOSE),
            daemon=True, name='FpvViewerProc')
        fpv_proc.start()
        print(f'FPV viewer process started (pid={fpv_proc.pid})')

    emergency_stop_thread = threading.Thread(
        target=emergency_stop_listener, args=(ctrl, cam, cf), daemon=True)
    emergency_stop_thread.start()

    try:
        ctrl.run_vision_lap()
    finally:
        if fpv_q is not None:
            try:
                fpv_q.put_nowait(None)  # graceful shutdown sentinel
            except Exception:
                pass
        if fpv_proc is not None:
            fpv_proc.join(timeout=1.0)
        cam.stop()
        cf.close_link()


if __name__ == '__main__':
    main()
