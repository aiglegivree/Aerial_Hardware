# -*- coding: utf-8 -*-
"""
drone_vision.py — Vision gate traversal (Crazyflie + AI-deck + Lighthouse)
===========================================================================

Architecture
------------
  UdpVideoThread   decodes each JPEG frame AND immediately runs gate
                   detection on it, computing pixel errors.  The result
                   is stored in a thread-safe Detection slot.

  GateController   (main thread) reads the latest Detection from the
                   camera thread — no CV work in the control loop.

Detection namedtuple: (cx, cy, size, e_x, e_y, e_z, box)
  cx, cy   gate centre in pixels
  size     largest bounding-box dimension (px)
  e_x      cx_mid - cx          lateral pixel error  → vy (strafe)
  e_y      cy_mid - cy          vertical pixel error → Δz (climb)
  e_z      GATE_SIZE_CLOSE - sz depth error          → vx (forward)
  box      (4,2) int32 corners of the rotated rect (for FPV overlay)

State machine per gate:  SEARCH → LOCK → APPROACH → TRANSIT
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
from collections import namedtuple

import cv2
import numpy as np
from pynput import keyboard

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

warnings.filterwarnings("ignore", message=".*supervisor subsystem requires CRTP.*")
logging.basicConfig(level=logging.ERROR)

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

# Frame centre — used by the camera thread to compute pixel errors
_CX_MID = CAM_WIDTH  / 2.0   # 162 px
_CY_MID = CAM_HEIGHT / 2.0   # 122 px

# ── FPV viewer ─────────────────────────────────────────────────────────────────

FPV_ENABLED = False
FPV_SCALE   = 2
FPV_RATE_HZ = 10

# ── arena bounds (Lighthouse world frame) ──────────────────────────────────────

ARENA_X_MIN = -1.0
ARENA_X_MAX = +3.0
ARENA_Y_MIN = -0.9
ARENA_Y_MAX = +0.9
SAFETY_MARGIN_HARD = 0.0

# ── flight ─────────────────────────────────────────────────────────────────────

CRUISE_ALT       = 1.25   # m
TAKEOFF_DURATION = 3.0    # s
LAND_DURATION    = 3.0    # s

# IBVS gains
KP_VX        = 0.008
KP_VY        = 0.005
KP_VZ        = 0.005
MAX_VX       = 0.05    # m/s
MAX_VY       = 0.05    # m/s
MAX_VZ_DELTA = 0.5     # m

# Alignment gating: gate must be this centred (px) before TRANSIT is allowed
ALIGN_TOL_X       = 25    # px
ALIGN_TOL_Y       = 25    # px
ALIGN_SCALE_DENOM = 0.6   # fraction of half-frame at which vx -> 0

TRANSIT_VX   = 0.30   # m/s
TRANSIT_TIME = 1.5    # s

SEARCH_YAW_RATE = 15.0   # deg/s
LOST_TOLERANCE  = 15     # consecutive no-detection ticks before re-search
LOCK_DURATION   = 1.5    # s — hover-and-confirm window

N_GATES = 5

# ── gate detector parameters ───────────────────────────────────────────────────

GATE_SIZE_MIN   = 30    # px — smallest accepted detection
GATE_SIZE_CLOSE = 180   # px — gate fills frame -> TRANSIT

GATE_HSV_LOWER = np.array([0,   0,   200], dtype=np.uint8)
GATE_HSV_UPPER = np.array([255, 100, 255], dtype=np.uint8)


# ── detection result ───────────────────────────────────────────────────────────

Detection = namedtuple('Detection', ['cx', 'cy', 'size', 'e_x', 'e_y', 'e_z', 'box'])
# All pixel errors are computed in _detect_gate() so the control loop is
# pure arithmetic with no CV calls.


# ── helpers ────────────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


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


def _fill_holes(mask):
    """Turn a closed gate outline into a solid filled rectangle."""
    flood = mask.copy()
    h, w = mask.shape
    scratch = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, scratch, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


def _detect_gate(frame):
    """
    Run the full gate-detection pipeline on one BGR frame.
    Returns a Detection namedtuple, or None if no gate found.

    This is called by UdpVideoThread immediately after each JPEG is decoded —
    never by the control loop.
    """
    if frame is None:
        return None

    bgr = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, GATE_HSV_LOWER, GATE_HSV_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 20), np.uint8))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mask = _fill_holes(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = 0.0
    best_gate  = None   # (cx, cy, size, rect)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 250:
            continue
        rect = cv2.minAreaRect(cnt)
        (rcx, rcy), (rw, rh), _ = rect
        if rw < 10 or rh < 10:
            continue
        short_side, long_side = sorted([rw, rh])
        if short_side / long_side < 0.45:
            continue
        rectangularity = area / (rw * rh)
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
            best_gate  = (rcx, rcy, max(rw, rh), rect)

    if best_gate is None:
        return None

    cx, cy, size, rect = best_gate

    # Pixel errors computed here — main thread reads them as plain numbers
    e_x = _CX_MID - cx            # +ve -> gate left of centre  -> strafe left
    e_y = _CY_MID - cy            # +ve -> gate above centre     -> climb
    e_z = GATE_SIZE_CLOSE - size  # +ve -> gate too small        -> move forward

    box = cv2.boxPoints(rect).astype(np.int32)
    return Detection(cx=cx, cy=cy, size=size, e_x=e_x, e_y=e_y, e_z=e_z, box=box)


# ── camera thread ──────────────────────────────────────────────────────────────

class UdpVideoThread(threading.Thread):
    """
    Reads AI-deck UDP packets, decodes JPEG frames, and immediately runs
    _detect_gate() on every frame.  Exposes latest_frame and latest_detection
    via thread-safe properties.

    The control loop only reads .latest_detection — no CV in the main thread.
    """

    def __init__(self):
        super().__init__(daemon=True, name='UdpVideoThread')
        self._lock      = threading.Lock()
        self._frame     = None
        self._detection = None   # Detection or None
        self._running   = False

    def start(self):
        self._running = True
        super().start()

    def stop(self):
        self._running = False

    @property
    def latest_frame(self):
        with self._lock:
            return self._frame

    @property
    def latest_detection(self):
        """Pre-computed Detection from the most recent frame, or None."""
        with self._lock:
            return self._detection

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(('0.0.0.0', UDP_LOCAL_PORT))
        sock.sendto(UDP_START_MAGIC, (UDP_AIDECK_IP, UDP_AIDECK_PORT))

        buffer        = bytearray()
        expected_size = 0
        receiving     = False

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
                    expected_size = size
                    buffer        = bytearray()
                    receiving     = True
                    continue

            if not receiving:
                continue

            buffer.extend(payload)

            if len(buffer) >= expected_size:
                frame = self._decode_frame(buffer)
                if frame is not None:
                    det = _detect_gate(frame)   # detection happens here
                    with self._lock:
                        self._frame     = frame
                        self._detection = det
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


# ── FPV viewer thread ──────────────────────────────────────────────────────────

class FpvViewerThread(threading.Thread):
    """
    Live FPV window.  Reads the camera thread's latest_frame and
    latest_detection — no redundant CV here — and draws the overlay.
    """

    def __init__(self, cam: UdpVideoThread, ctrl=None):
        super().__init__(daemon=True, name='FpvViewerThread')
        self._cam     = cam
        self._ctrl    = ctrl
        self._running = False

    def start(self):
        self._running = True
        super().start()

    def stop(self):
        self._running = False

    def run(self):
        win    = 'Crazyflie FPV'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, CAM_WIDTH * FPV_SCALE, CAM_HEIGHT * FPV_SCALE)
        last   = None
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
            det  = self._cam.latest_detection   # pre-computed, no re-detection

            disp = frame.copy()

            cv2.line(disp, (CAM_WIDTH // 2, 0), (CAM_WIDTH // 2, CAM_HEIGHT), (60, 60, 60), 1)
            cv2.line(disp, (0, CAM_HEIGHT // 2), (CAM_WIDTH, CAM_HEIGHT // 2), (60, 60, 60), 1)

            if det is not None:
                color = (0, 255, 0) if det.size >= GATE_SIZE_CLOSE else (0, 200, 255)
                cv2.drawContours(disp, [det.box], 0, color, 2)
                cv2.drawMarker(disp, (int(det.cx), int(det.cy)), color,
                               cv2.MARKER_CROSS, 14, 2)
                cv2.putText(disp,
                            f'cx={det.cx:.0f} cy={det.cy:.0f} sz={det.size:.0f}'
                            f'  ex={det.e_x:+.0f} ey={det.e_y:+.0f} ez={det.e_z:+.0f}',
                            (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            else:
                cv2.putText(disp, 'no gate', (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if self._ctrl is not None:
                s = self._ctrl._state
                cv2.putText(disp,
                            f"x={s['x']:+.2f} y={s['y']:+.2f} z={s['z']:.2f}"
                            f" yaw={s['yaw']:+.0f}",
                            (5, CAM_HEIGHT - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (255, 255, 255), 1)

            if FPV_SCALE != 1:
                disp = cv2.resize(disp, (CAM_WIDTH * FPV_SCALE, CAM_HEIGHT * FPV_SCALE),
                                  interpolation=cv2.INTER_NEAREST)
            cv2.imshow(win, disp)
            if (cv2.waitKey(1) & 0xFF) == ord('Q'):
                break

        try:
            cv2.destroyWindow(win)
        except Exception:
            pass


# ── flight controller ──────────────────────────────────────────────────────────

class GateController:

    def __init__(self, cf: Crazyflie, cam: UdpVideoThread):
        self._cf    = cf
        self._cam   = cam
        self._stop  = False
        self._state = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}

        self.is_connected = False
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

    # ── boundary-safe hover ────────────────────────────────────────────────────

    def _safe_hover(self, vx=0.0, vy=0.0, yaw_rate=0.0, z=CRUISE_ALT):
        x   = self._state['x']
        y   = self._state['y']
        yaw = math.radians(self._state['yaw'])

        wx =  vx * math.cos(yaw) - vy * math.sin(yaw)
        wy =  vx * math.sin(yaw) + vy * math.cos(yaw)

        if x <= ARENA_X_MIN + SAFETY_MARGIN_HARD and wx < 0: wx = 0.0
        if x >= ARENA_X_MAX - SAFETY_MARGIN_HARD and wx > 0: wx = 0.0
        if y <= ARENA_Y_MIN + SAFETY_MARGIN_HARD and wy < 0: wy = 0.0
        if y >= ARENA_Y_MAX - SAFETY_MARGIN_HARD and wy > 0: wy = 0.0

        vx_s =  wx * math.cos(yaw) + wy * math.sin(yaw)
        vy_s = -wx * math.sin(yaw) + wy * math.cos(yaw)
        self._cf.commander.send_hover_setpoint(vx_s, vy_s, yaw_rate, z)

    def _stop_motors(self):
        self._cf.commander.send_stop_setpoint()

    # ── takeoff / land ─────────────────────────────────────────────────────────

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

    # ── SEARCH ────────────────────────────────────────────────────────────────

    def search_for_gate(self):
        """Sweep yaw back and forth until the camera thread reports a gate."""
        print('  [SEARCH] sweeping...')
        t_start = time.time()
        while not self._stop:
            det = self._cam.latest_detection   # pre-computed in camera thread
            if det is not None:
                print(f'  [SEARCH] gate found  cx={det.cx:.0f}  cy={det.cy:.0f}'
                      f'  size={det.size:.0f}px')
                return

            elapsed   = time.time() - t_start
            sweep_yaw = math.sin(elapsed * math.pi / 5) * SEARCH_YAW_RATE
            self._safe_hover(yaw_rate=sweep_yaw, z=CRUISE_ALT)
            time.sleep(0.05)

    # ── LOCK ──────────────────────────────────────────────────────────────────

    def lock_on_gate(self):
        """
        Hover in place for LOCK_DURATION while the camera keeps reporting a
        gate.  Returns True if still detected at the end, False otherwise.
        """
        print(f'  [LOCK] holding {LOCK_DURATION:.1f}s to confirm')
        t_end    = time.time() + LOCK_DURATION
        last_det = None
        while time.time() < t_end and not self._stop:
            det = self._cam.latest_detection
            if det is not None:
                last_det = det
            self._safe_hover(z=CRUISE_ALT)
            time.sleep(0.05)

        if last_det is not None:
            print(f'  [LOCK] confirmed (cx={last_det.cx:.0f}'
                  f'  size={last_det.size:.0f}px) -> APPROACH')
            return True
        print('  [LOCK] lost during hover -> SEARCH')
        return False

    # ── APPROACH ─────────────────────────────────────────────────────────────

    def approach_gate(self):
        """
        IBVS: read pre-computed pixel errors from the camera thread and turn
        them directly into velocity commands — no CV here.

          e_x -> vy  (strafe)
          e_y -> dz  (climb/descend)
          e_z -> vx  (forward; never backward)

        Returns True when gate is close AND centred (-> TRANSIT).
        Returns False if gate is lost too long (-> SEARCH).
        """
        print('  [APPROACH] IBVS toward gate')
        target_z = CRUISE_ALT
        lost     = 0

        while not self._stop:
            det = self._cam.latest_detection   # errors already computed

            if det is None:
                lost += 1
                if lost > LOST_TOLERANCE:
                    print('  [APPROACH] gate lost -> SEARCH')
                    return False
                self._safe_hover(z=target_z)
                time.sleep(0.05)
                continue

            lost = 0

            # Termination: gate fills frame and is centred
            if (det.size >= GATE_SIZE_CLOSE
                    and abs(det.e_x) < ALIGN_TOL_X
                    and abs(det.e_y) < ALIGN_TOL_Y):
                print(f'  [APPROACH] aligned & close'
                      f' (sz={det.size:.0f}px ex={det.e_x:+.0f} ey={det.e_y:+.0f})'
                      f' -> TRANSIT')
                return True

            # Proportional control — pure arithmetic, no CV
            v_strafe  = clamp(KP_VY * det.e_x, -MAX_VY, MAX_VY)
            v_climb   = clamp(KP_VZ * det.e_y, -MAX_VZ_DELTA, MAX_VZ_DELTA)
            v_forward = clamp(KP_VX * det.e_z,  0.0, MAX_VX)

            # Throttle forward speed by lateral alignment
            mis_x = abs(det.e_x) / _CX_MID
            mis_y = abs(det.e_y) / _CY_MID
            align = clamp(1.0 - (mis_x + mis_y) / ALIGN_SCALE_DENOM, 0.0, 1.0)
            v_forward *= align

            target_z = clamp(CRUISE_ALT + v_climb,
                             CRUISE_ALT - MAX_VZ_DELTA,
                             CRUISE_ALT + MAX_VZ_DELTA)

            print(f'  [IBVS] ex={det.e_x:+.0f} ey={det.e_y:+.0f}'
                  f' sz={det.size:.0f}  align={align:.2f}'
                  f'  -> vx={v_forward:.2f} vy={v_strafe:+.2f} z={target_z:.2f}')

            self._safe_hover(vx=v_forward, vy=v_strafe, z=target_z)
            time.sleep(0.05)

        return False

    # ── TRANSIT ───────────────────────────────────────────────────────────────

    def transit_gate(self):
        """
        Push forward through the gate.  Correct lateral/vertical error from
        the camera thread as long as the gate is still in frame; then finish
        the push open-loop once it disappears.
        """
        print('  [TRANSIT] flying through gate')
        target_z = CRUISE_ALT
        t_end    = time.time() + TRANSIT_TIME

        while time.time() < t_end and not self._stop:
            det = self._cam.latest_detection
            if det is not None:
                v_strafe = clamp(KP_VY * det.e_x, -MAX_VY, MAX_VY)
                v_climb  = clamp(KP_VZ * det.e_y, -MAX_VZ_DELTA, MAX_VZ_DELTA)
                target_z = clamp(CRUISE_ALT + v_climb,
                                 CRUISE_ALT - MAX_VZ_DELTA,
                                 CRUISE_ALT + MAX_VZ_DELTA)
                self._safe_hover(vx=TRANSIT_VX, vy=v_strafe, z=target_z)
            else:
                self._safe_hover(vx=TRANSIT_VX, z=target_z)
            time.sleep(0.05)

        print('  [TRANSIT] done')

    # ── mission ────────────────────────────────────────────────────────────────

    def run_vision_lap(self):
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

                while not self._stop:
                    self.search_for_gate()
                    if self._stop or self.lock_on_gate():
                        break

                if self._stop:
                    break

                while not self._stop:
                    if self.approach_gate():
                        self.transit_gate()
                        break
                    print('  gate lost mid-approach — searching again')
                    while not self._stop:
                        self.search_for_gate()
                        if self._stop or self.lock_on_gate():
                            break

            msg = 'All gates complete' if not self._stop else 'Emergency stop'
            print(f'\n[MISSION] {msg} — landing')

        except Exception as e:
            print(f'\n[MISSION] Unhandled exception: {e} — landing now')

        finally:
            try:
                self.land()
            except Exception as e:
                print(f'Land failed ({e}) — cutting motors')
                self._stop_motors()


# ── emergency stop ─────────────────────────────────────────────────────────────

def emergency_stop_listener(ctrl: GateController, cam: UdpVideoThread, cf: Crazyflie):
    """
    ESC -> set _stop; mission loop exits and its finally block lands cleanly.
    Q   -> cut motors immediately (drone drops).
    """
    def on_press(key):
        if key == keyboard.Key.esc:
            print('\n[STOP] ESC — landing')
            ctrl._stop = True
            return False
        try:
            if key.char == 'q':
                print('\n[STOP] Q — cutting motors immediately')
                ctrl._stop = True
                ctrl._stop_motors()
                return False
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
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

    cam = UdpVideoThread()
    cam.start()
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

    stop_thread = threading.Thread(
        target=emergency_stop_listener, args=(ctrl, cam, cf), daemon=True)
    stop_thread.start()

    try:
        ctrl.run_vision_lap()
    finally:
        if fpv is not None:
            fpv.stop()
        cam.stop()
        cf.close_link()
