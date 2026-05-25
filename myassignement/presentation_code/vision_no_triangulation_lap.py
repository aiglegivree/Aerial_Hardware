# -*- coding: utf-8 -*-
"""
Vision lap — no triangulation (Image-Based Visual Servoing).

Per gate:  SEARCH → APPROACH → TRANSIT
  SEARCH    — rotate CCW until a gate is seen and centred in frame.
  APPROACH  — IBVS: strafe + altitude + yaw align until park conditions hold,
              then push forward; ends when the gate fills the frame.
  TRANSIT   — open-loop forward push for TRANSIT_TIME at the locked altitude.

Repeats for N_GATES gates.

Keys: 'q' = motor cut, ESC = controlled landing.
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

# ── flight ─────────────────────────────────────────────────────────────────────

CRUISE_ALT       = 1.25  # m
TAKEOFF_DURATION = 3.0   # s
LAND_DURATION    = 3.0   # s

MAX_VZ_STEP = 0.005

# IBVS gains
KP_VX        = 0.012  # size error  (GATE_SIZE_CLOSE - size) → forward speed
KP_VY        = 0.005  # lateral pixel error (cx - cx_mid)    → strafe
KP_VZ        = 0.005  # vertical pixel error (cy_mid - cy)   → altitude delta
MAX_VX       = 0.20
MAX_VY       = 0.10
MAX_VZ_DELTA = 0.5

ALIGN_TOL_ARM    = 0.10
ALIGN_TOL_DISARM = 0.20
PARK_HOLD_S      = 0.2

KP_YAW_ALIGN   = 8.0
MAX_YAW_ALIGN  = 8.0
ALIGN_DEADBAND = 0.08

CENTER_TOL_PX = 40

TRANSIT_VX   = 0.10
TRANSIT_TIME = 10.0

SEARCH_YAW_RATE = 10.0
LOST_TOLERANCE  = 50

N_GATES = 5

GATE_SIZE_MIN   = 30   # px
GATE_SIZE_CLOSE = 180  # px — gate fills frame, switch to TRANSIT


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
        sock.settimeout(0.5)
        sock.sendto(UDP_START_MAGIC, (UDP_AIDECK_IP, UDP_AIDECK_PORT))

        buffer = bytearray()
        expected_size = 0
        receiving = False

        while self._running:
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                sock.sendto(UDP_START_MAGIC, (UDP_AIDECK_IP, UDP_AIDECK_PORT))
                continue
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

        if img is None:
            return None
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


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

    @staticmethod
    def _detect_overlay(frame):
        result = get_gate_detection(frame)
        if result[0] is None:
            return None, None, []
        cx, cy, size, lh, rh, box_pts = result
        rect = ((cx, cy), (size, size), 0)
        return None, rect, []

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
            disp = frame.copy()

            _mask, best_rect, rejected = self._detect_overlay(frame)
            cv2.line(disp, (CAM_WIDTH // 2, 0), (CAM_WIDTH // 2, CAM_HEIGHT),
                     (60, 60, 60), 1)
            cv2.line(disp, (0, CAM_HEIGHT // 2), (CAM_WIDTH, CAM_HEIGHT // 2),
                     (60, 60, 60), 1)

            for rect in rejected:
                box = cv2.boxPoints(rect).astype(np.int32)
                cv2.drawContours(disp, [box], 0, (0, 0, 180), 1)

            if best_rect is not None:
                (rcx, rcy), (rw, rh), _ = best_rect
                size = max(rw, rh)
                color = (0, 255, 0) if size >= GATE_SIZE_CLOSE else (0, 200, 255)
                box = cv2.boxPoints(best_rect).astype(np.int32)
                cv2.drawContours(disp, [box], 0, color, 2)
                cv2.drawMarker(disp, (int(rcx), int(rcy)), color,
                               cv2.MARKER_CROSS, 14, 2)
                cv2.putText(disp,
                            f'cx={rcx:.0f} cy={rcy:.0f} size={size:.0f}',
                            (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                cv2.putText(disp, 'no gate', (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if self._ctrl is not None:
                s = self._ctrl._state
                cv2.putText(disp,
                            f"x={s['x']:+.2f} y={s['y']:+.2f} z={s['z']:.2f} yaw={s['yaw']:+.0f}",
                            (5, CAM_HEIGHT - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (255, 255, 255), 1)

            if FPV_SCALE != 1:
                disp = cv2.resize(disp,
                                  (CAM_WIDTH * FPV_SCALE, CAM_HEIGHT * FPV_SCALE),
                                  interpolation=cv2.INTER_NEAREST)
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('Q'):
                if self._ctrl is not None:
                    print('\n[STOP] Q (FPV window) — cutting motors immediately')
                    self._ctrl._stop = True
                    self._ctrl._stop_motors()
                break
            elif key == 27:  # ESC
                if self._ctrl is not None:
                    print('\n[STOP] ESC (FPV window) — landing')
                    self._ctrl._stop = True
                break
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass


# ── helpers ────────────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _fill_holes(mask):
    """Fill regions fully enclosed by white. Bail out on degenerate cases."""
    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    scratch = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, scratch, (0, 0), 255)
    flood = padded[1:-1, 1:-1]

    exterior_pixels = cv2.countNonZero(flood) - cv2.countNonZero(mask)
    total = h * w
    if exterior_pixels < 0.5 * total:
        return mask

    return mask | cv2.bitwise_not(flood)


def _gate_edge_heights(contour):
    """Approximate gate as quad, return (left_edge_height, right_edge_height) px."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    if len(approx) != 4:
        return None, None

    pts = approx.reshape(4, 2).astype(np.float32)
    sorted_by_x = pts[np.argsort(pts[:, 0])]
    left_pts  = sorted_by_x[:2]
    right_pts = sorted_by_x[2:]

    left_h  = float(np.linalg.norm(left_pts[0]  - left_pts[1]))
    right_h = float(np.linalg.norm(right_pts[0] - right_pts[1]))
    return left_h, right_h


# Bright LED gates: high V, low-to-moderate S, any H.
GATE_HSV_LOWER = np.array([0,   0,   200], dtype=np.uint8)
GATE_HSV_UPPER = np.array([255, 100, 255], dtype=np.uint8)


def get_gate_detection(frame):
    if frame is None:
        return None, None, None, None, None, None

    if len(frame.shape) == 3:
        bgr = frame
    else:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GATE_HSV_LOWER, GATE_HSV_UPPER)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = _fill_holes(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = 0.0
    best_gate = None
    best_rect = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4500:
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

        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area <= 0:
            continue
        solidity = area / hull_area

        size_half = max(rw, rh) / 2
        edge_clipped = (rcx - size_half < 4 or
                        rcx + size_half > CAM_WIDTH  - 4 or
                        rcy - size_half < 4 or
                        rcy + size_half > CAM_HEIGHT - 4)

        if edge_clipped:
            rect_min, solid_min = 0.55, 0.55
        else:
            rect_min, solid_min = 0.80, 0.85

        if rectangularity < rect_min:
            continue
        if solidity < solid_min:
            continue

        score = rectangularity * solidity * area
        if score > best_score:
            best_score = score
            left_h, right_h = _gate_edge_heights(cnt)
            best_rect = rect
            best_gate = (rcx, rcy, max(rw, rh), left_h, right_h)

    # Fallback: large edge-touching contours
    if best_gate is None:
        FALLBACK_AREA_MIN = 700
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < FALLBACK_AREA_MIN:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            touches_edge = (x <= 2 or y <= 2 or
                            x + w >= CAM_WIDTH  - 2 or
                            y + h >= CAM_HEIGHT - 2)
            if not touches_edge:
                continue

            rect = cv2.minAreaRect(cnt)
            (rcx, rcy), (rw, rh), _ = rect
            if rw < 10 or rh < 10:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 50:
                continue

            if area > best_score:
                best_score = area
                left_h, right_h = _gate_edge_heights(cnt)
                best_rect = rect
                best_gate = (rcx, rcy, max(rw, rh), left_h, right_h)

    if best_gate is not None:
        box_pts = cv2.boxPoints(best_rect)
        return best_gate[0], best_gate[1], best_gate[2], best_gate[3], best_gate[4], box_pts
    return None, None, None, None, None, None


# ── flight controller ──────────────────────────────────────────────────────────

class GateController:

    def __init__(self, cf: Crazyflie, cam: UdpVideoThread = None):
        self._cf  = cf
        self._cam = cam

        self.is_connected = False
        self._stop        = False
        self._state       = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}

        lg = LogConfig(name='State', period_in_ms=100)
        lg.add_variable('stateEstimate.x', 'float')
        lg.add_variable('stateEstimate.y', 'float')
        lg.add_variable('stateEstimate.z', 'float')
        lg.add_variable('stabilizer.yaw',  'float')
        lg.add_variable('pm.vbat', 'float')
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

        vbat = data['pm.vbat']
        self._state['vbat'] = vbat
        if vbat < 3.2 and time.time() - getattr(self, '_last_vbat_warn', 0) > 1.0:
            print(f'  [LOW BATTERY] vbat={vbat:.2f} V')
            self._last_vbat_warn = time.time()

    # ── hover / stop ─────────────────────────────────────────────────────────

    def _hover(self, vx=0.0, vy=0.0, yaw_rate=0.0, z=CRUISE_ALT):
        self._cf.commander.send_hover_setpoint(vx, vy, yaw_rate, z)

    def _stop_motors(self):
        self._cf.commander.send_stop_setpoint()

    # ── take-off / landing ───────────────────────────────────────────────────

    def takeoff(self, target_z=CRUISE_ALT):
        print(f'Taking off to {target_z:.2f} m')
        steps = int(TAKEOFF_DURATION / 0.1)
        for i in range(steps):
            self._hover(z=target_z * (i / steps))
            time.sleep(0.1)
        for _ in range(20):
            self._hover(z=target_z)
            time.sleep(0.1)

    def land(self):
        print('Landing')
        current_z = max(self._state['z'], CRUISE_ALT)
        steps = int(LAND_DURATION / 0.1)
        for i in range(steps):
            self._hover(z=current_z * (1.0 - i / steps))
            time.sleep(0.1)
        self._stop_motors()

    # ── state: SEARCH ────────────────────────────────────────────────────────

    def search_for_gate(self):
        """CCW rotation until a gate is seen, then proportional yaw to centre it."""
        print('  [SEARCH] rotating CCW...')
        cx_mid   = CAM_WIDTH / 2.0
        ALIGN_PX = 50

        while not self._stop:
            cx, cy, size, left_h, right_h, _ = get_gate_detection(self._cam.latest_frame)
            if cx is None:
                self._hover(yaw_rate=SEARCH_YAW_RATE, z=CRUISE_ALT)
            else:
                e_x = cx_mid - cx
                if abs(e_x) < ALIGN_PX:
                    print('  [SEARCH] gate centred → APPROACH')
                    return

                yaw_rate = clamp(SEARCH_YAW_RATE * e_x / cx_mid,
                                 -SEARCH_YAW_RATE, SEARCH_YAW_RATE)
                self._hover(yaw_rate=yaw_rate, z=CRUISE_ALT)

            time.sleep(0.05)

    # ── state: APPROACH ──────────────────────────────────────────────────────

    def approach_gate(self):
        """IBVS: strafe + climb + yaw-align, then push forward once park
        conditions (centred + edges symmetric) hold for PARK_HOLD_S.
        Locks altitude the first time the gate is centred and holds it.
        Returns (True, locked_z) on TRANSIT or (False, None) if lost."""
        print('  [APPROACH] IBVS toward gate (park-first)')
        cx_mid   = CAM_WIDTH  / 2.0
        cy_mid   = CAM_HEIGHT / 2.0
        target_z = CRUISE_ALT
        locked_z = None
        lost     = 0

        park_armed = False
        park_t0    = None

        EDGE_MARGIN = 8

        last_v_forward = 0.0
        last_v_strafe  = 0.0
        last_yaw_rate  = 0.0

        while not self._stop:
            cx, cy, size, left_h, right_h, _ = get_gate_detection(self._cam.latest_frame)

            if cx is None:
                lost += 1
                if lost > LOST_TOLERANCE:
                    print('  [APPROACH] gate lost — back to SEARCH')
                    return False, None

                z_hold = locked_z if locked_z is not None else target_z
                self._hover(vx=last_v_forward, vy=last_v_strafe,
                            yaw_rate=last_yaw_rate, z=z_hold)
                time.sleep(0.05)
                continue

            lost = 0

            if size >= GATE_SIZE_CLOSE:
                print(f'  [APPROACH] gate fills frame (size={size:.0f}px) → TRANSIT')
                if locked_z is None:
                    locked_z = self._state['z']
                    print(f'  [APPROACH] altitude not yet locked — using current z={locked_z:.2f}')
                return True, locked_z

            e_x = cx_mid - cx
            e_y = cy_mid - cy
            e_z = GATE_SIZE_CLOSE - size

            asymmetry = 0.0
            have_align = False
            if left_h is not None and right_h is not None:
                edge_clipped = (cx - size/2 < EDGE_MARGIN or
                                cx + size/2 > CAM_WIDTH - EDGE_MARGIN)
                if not edge_clipped:
                    long_edge = max(left_h, right_h)
                    if long_edge > 1e-3:
                        asymmetry = (left_h - right_h) / long_edge
                        have_align = True

            yaw_rate = 0.0
            if have_align and abs(asymmetry) > ALIGN_DEADBAND:
                yaw_rate = clamp(KP_YAW_ALIGN * asymmetry,
                                 -MAX_YAW_ALIGN, MAX_YAW_ALIGN)

            v_strafe = clamp(KP_VY * e_x, -MAX_VY, MAX_VY)

            centered = abs(e_x) < CENTER_TOL_PX and abs(e_y) < CENTER_TOL_PX
            aligned = (not have_align) or abs(asymmetry) < ALIGN_TOL_ARM

            if park_armed:
                disarm = have_align and abs(asymmetry) > ALIGN_TOL_DISARM
                if disarm or not centered:
                    park_armed = False
                    park_t0    = None
                    print('  [PARK] lost alignment — pausing forward motion')
            else:
                parked_now = centered and aligned
                if parked_now:
                    if park_t0 is None:
                        park_t0 = time.time()
                    elif time.time() - park_t0 >= PARK_HOLD_S:
                        park_armed = True
                        print('  [PARK] held — arming forward motion')
                else:
                    park_t0 = None

            v_forward = clamp(KP_VX * e_z, 0.0, MAX_VX) if park_armed else 0.0

            if locked_z is None:
                if centered:
                    locked_z = self._state['z']
                    print(f'  [APPROACH] gate centred → locking altitude z={locked_z:.2f} m '
                          f'(no further altitude corrections until next SEARCH)')
                else:
                    desired_dz = clamp(KP_VZ * e_y, -MAX_VZ_DELTA, MAX_VZ_DELTA)
                    target_z_desired = clamp(CRUISE_ALT + desired_dz,
                                             CRUISE_ALT - MAX_VZ_DELTA,
                                             CRUISE_ALT + MAX_VZ_DELTA)
                    dz_step = clamp(target_z_desired - target_z, -MAX_VZ_STEP, MAX_VZ_STEP)
                    target_z += dz_step

            z_cmd = locked_z if locked_z is not None else target_z

            status = 'GO' if park_armed else 'park'
            align_str = f'{asymmetry:+.2f}' if have_align else '  -- '
            z_tag = 'LOCK' if locked_z is not None else ' adj'
            print(f'  [IBVS|{status}] ex={e_x:+.0f} ey={e_y:+.0f} '
                  f'asym={align_str} size={size:.0f}px → '
                  f'vx={v_forward:.2f} vy={v_strafe:+.2f} '
                  f'yaw={yaw_rate:+.1f}°/s z={z_cmd:.2f}[{z_tag}]')

            last_v_forward = v_forward
            last_v_strafe  = v_strafe
            last_yaw_rate  = yaw_rate
            self._hover(vx=v_forward, vy=v_strafe, yaw_rate=yaw_rate, z=z_cmd)
            time.sleep(0.05)

        return False, None

    # ── state: TRANSIT ───────────────────────────────────────────────────────

    def transit_gate(self, z=None):
        """Open-loop forward push at the locked altitude to clear the gate."""
        z_cmd = z if z is not None else CRUISE_ALT
        print(f'  [TRANSIT] flying through gate at z={z_cmd:.2f} m')
        t_end = time.time() + TRANSIT_TIME
        while time.time() < t_end and not self._stop:
            self._hover(vx=TRANSIT_VX, z=z_cmd)
            time.sleep(0.05)
        print('  [TRANSIT] done')

    # ── vision mission ───────────────────────────────────────────────────────

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

                self.search_for_gate()
                if self._stop:
                    break

                while not self._stop:
                    success, locked_z = self.approach_gate()
                    if success:
                        self.transit_gate(z=locked_z)
                        break
                    print('  gate lost — searching again')
                    self.search_for_gate()

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
    """ESC → request controlled land; Q → cut motors immediately."""
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

    try:
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    except Exception as e:
        print(f'[WARN] pynput keyboard listener failed ({e}) — use Q/ESC in the FPV window')


# ── entry point ────────────────────────────────────────────────────────────────

def main():
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
        return

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
        return

    fpv = None
    if FPV_ENABLED:
        fpv = FpvViewerThread(cam, ctrl)
        fpv.start()
        print('FPV viewer started')

    threading.Thread(
        target=emergency_stop_listener, args=(ctrl, cam, cf), daemon=True).start()

    try:
        ctrl.run_vision_lap()
    finally:
        if fpv is not None:
            fpv.stop()
        cam.stop()
        cf.close_link()


if __name__ == '__main__':
    main()
