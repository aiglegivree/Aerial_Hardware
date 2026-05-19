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

Part 2 — position waypoints (Lighthouse world frame):
  per gate: pre-gate point → (through) → post-gate point.
  Each waypoint is reached with a speed-limited, interpolated
  send_position_setpoint stream. Boundary safety clamps any target
  whose radius exceeds 2 m.

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

CONTROL_URI = uri_helper.uri_from_env(default='radio://0/10/2M/E7E7E7E708')

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

# ── flight ─────────────────────────────────────────────────────────────────────

CRUISE_ALT       = 0.6   # m
TAKEOFF_DURATION = 3.0   # s
LAND_DURATION    = 3.0   # s

KP_VX        = 0.004  # (GATE_SIZE_CLOSE - size) → vx
KP_VY        = 0.003  # cx_err  → vy
KP_VZ        = 0.002  # cy_err  → Δz
MAX_VX       = 0.4    # m/s
MAX_VY       = 0.3    # m/s
MAX_VZ_DELTA = 0.3    # m

TRANSIT_VX   = 0.5    # m/s
TRANSIT_TIME = 2.0    # s

SEARCH_YAW_RATE = 15.0  # deg/s CCW
SEARCH_TIMEOUT  = 10.0  # s — double yaw rate after this
LOST_TOLERANCE  = 15    # consecutive no-detection frames before re-search

BOUNDARY_RADIUS      = 2.0  # m — hard limit
BOUNDARY_SOFT_RADIUS = 1.7  # m — start braking here

N_GATES = 5  # Part 1 vision gates to search for 

# ── Part 2: position-based mission ─────────────────────────────────────────────
#
# Fill GATE_POSITIONS once the instructor gives you the exact gate positions.
# Each entry is the gate CENTRE in the Lighthouse world frame, in metres:
#     (x, y, z)              — approach direction derived from gate-to-gate line
#     (x, y, z, yaw_deg)     — yaw_deg = direction the gate faces; the drone
#                              flies through along this axis (preferred)
#
# Gates must be flown counter-clockwise (same order as listed here).
GATE_POSITIONS = [
    # (x, y, z)  or  (x, y, z, yaw_deg)
    # e.g. (1.20, 0.40, 0.80, 90.0),
]

N_LAPS           = 2     # number of timed laps
PRE_GATE_OFFSET  = 0.6   # m — waypoint placed before the gate along approach axis
POST_GATE_OFFSET = 0.6   # m — waypoint placed after the gate (clears the frame)
APPROACH_SPEED   = 0.4   # m/s — interpolated setpoint speed toward pre-gate (start slow!)
TRANSIT_SPEED    = 0.6   # m/s — speed through the gate (pre → post)
GOTO_TOL         = 0.15  # m — arrival tolerance for a waypoint
GOTO_SETTLE_TIME = 2.5   # s — extra time allowed after the nominal travel duration
POSITION_RATE_HZ = 20.0  # setpoint streaming rate for position mode


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

def get_gate_detection(frame):
    """
    Uses grayscale adaptive thresholding instead of HSV color filtering.
    Returns: cx, cy, size (or None, None, None if nothing found)
    """
    if frame is None:
        return None, None, None

    # Ensure image is grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Adaptive thresholding: finds dark objects on light backgrounds (or vice versa)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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
        Send hover setpoint, clamping outward velocity near the 2 m boundary.

        Inside BOUNDARY_SOFT_RADIUS  → no clamping.
        Between soft and hard radius → outward component linearly reduced to 0.
        At BOUNDARY_RADIUS           → outward component fully blocked;
                                        tangential + inward motion preserved.
        """
        x   = self._state['x']
        y   = self._state['y']
        yaw = math.radians(self._state['yaw'])

        # Body → world
        wx = vx * math.cos(yaw) - vy * math.sin(yaw)
        wy = vx * math.sin(yaw) + vy * math.cos(yaw)

        r = math.sqrt(x**2 + y**2)
        if r > BOUNDARY_SOFT_RADIUS:
            ox = x / r if r > 1e-6 else 0.0
            oy = y / r if r > 1e-6 else 0.0
            outward = wx * ox + wy * oy
            if outward > 0:
                scale  = clamp(
                    1.0 - (r - BOUNDARY_SOFT_RADIUS) /
                          (BOUNDARY_RADIUS - BOUNDARY_SOFT_RADIUS),
                    0.0, 1.0
                )
                factor = 1.0 - scale
                wx    -= factor * outward * ox
                wy    -= factor * outward * oy

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
            sweep_yaw_rate = math.sin(elapsed * math.pi) * SEARCH_YAW_RATE
            self._safe_hover(yaw_rate=sweep_yaw_rate, z=CRUISE_ALT)

            time.sleep(0.05)

    # ── state: APPROACH ──────────────────────────────────────────────────────

    def approach_gate(self):
        """
        Visual servo toward the fitted rectangle centre.
        Returns True  when size > GATE_SIZE_CLOSE  (→ TRANSIT).
        Returns False when gate lost               (→ SEARCH).
        """
        print('  [APPROACH] flying toward gate')
        cx_mid   = CAM_WIDTH  / 2.0
        cy_mid   = CAM_HEIGHT / 2.0
        lost     = 0
        target_z = CRUISE_ALT

        while not self._stop:
            cx, cy, size = get_gate_detection(self._cam.latest_frame)

            if cx is None:
                lost += 1
                if lost > LOST_TOLERANCE:
                    print('  [APPROACH] gate lost — back to SEARCH')
                    return False
                self._safe_hover(vx=0.1, z=target_z)
                time.sleep(0.05)
                continue

            lost = 0

            if size > GATE_SIZE_CLOSE:
                print(f'  [APPROACH] close enough (size={size:.0f}px) → TRANSIT')
                return True

            lat_err  = cx - cx_mid
            vy       = clamp(-KP_VY * lat_err, -MAX_VY, MAX_VY)

            vert_err = cy_mid - cy                          # +ve → gate above → climb
            target_z = clamp(
                CRUISE_ALT + KP_VZ * vert_err,
                CRUISE_ALT - MAX_VZ_DELTA,
                CRUISE_ALT + MAX_VZ_DELTA,
            )

            vx       = clamp(KP_VX * (GATE_SIZE_CLOSE - size), 0.05, MAX_VX)
            yaw_corr = -0.05 * lat_err                      # deg/s

            self._safe_hover(vx=vx, vy=vy, yaw_rate=yaw_corr, z=target_z)
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

    def run_vision_lap(self):
        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2)

        try:
            self.takeoff()

            for gate_idx in range(N_GATES):
                if self._stop:
                    break
                print(f'\n=== Gate {gate_idx + 1} / {N_GATES} ===')
                while not self._stop:
                    self.search_for_gate()
                    if self._stop:
                        break
                    if self.approach_gate():
                        self.transit_gate()
                        break
                    # gate lost during approach → retry search

            msg = 'All gates complete' if not self._stop else 'Emergency stop'
            print(f'\n{msg} — landing')

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
        """Clamp a target (x, y) so its radius never exceeds the 2 m boundary."""
        r = math.hypot(x, y)
        if r > BOUNDARY_RADIUS:
            s = BOUNDARY_RADIUS / r
            print(f'  [BOUND] target r={r:.2f} m clamped onto {BOUNDARY_RADIUS} m circle')
            return x * s, y * s
        return x, y

    def goto(self, tx, ty, tz, yaw_deg, speed=APPROACH_SPEED, tol=GOTO_TOL):
        """
        Fly to (tx, ty, tz) with send_position_setpoint, streaming a
        speed-limited interpolated setpoint so the move is smooth and not a
        full-speed dash. Returns True on arrival, False on stop / timeout.
        """
        tx, ty = self._clamp_to_boundary(tx, ty)

        sx, sy, sz = self._state['x'], self._state['y'], self._state['z']
        dist = math.sqrt((tx - sx)**2 + (ty - sy)**2 + (tz - sz)**2)
        duration = max(dist / max(speed, 1e-3), 0.3)

        dt = 1.0 / POSITION_RATE_HZ
        t0 = time.time()
        while not self._stop:
            elapsed = time.time() - t0
            frac = min(elapsed / duration, 1.0)

            # Interpolated (moving) setpoint between start and target
            ix = sx + frac * (tx - sx)
            iy = sy + frac * (ty - sy)
            iz = sz + frac * (tz - sz)
            self._cf.commander.send_position_setpoint(ix, iy, iz, yaw_deg)

            cx, cy, cz = self._state['x'], self._state['y'], self._state['z']
            d = math.sqrt((tx - cx)**2 + (ty - cy)**2 + (tz - cz)**2)
            if frac >= 1.0 and d < tol:
                return True
            if elapsed > duration + GOTO_SETTLE_TIME:
                print(f'  [GOTO] timeout near ({tx:.2f},{ty:.2f},{tz:.2f}) '
                      f'remaining={d:.2f} m')
                return False
            time.sleep(dt)
        return False

    def _build_gate_waypoints(self, gates, home_xy):
        """
        From the list of gate centres build, per gate, a pre-gate point, the
        gate centre and a post-gate point, all colinear along the approach
        axis so a straight pre→post move passes cleanly through the gate.
        """
        waypoints = []
        for i, g in enumerate(gates):
            gx, gy, gz = g[0], g[1], g[2]

            if len(g) >= 4 and g[3] is not None:
                # Explicit gate facing yaw provided — fly along that axis.
                yaw = math.radians(g[3])
                dirx, diry = math.cos(yaw), math.sin(yaw)
            else:
                # Derive approach direction from the previous waypoint.
                prev = home_xy if i == 0 else (gates[i - 1][0], gates[i - 1][1])
                dx, dy = gx - prev[0], gy - prev[1]
                n = math.hypot(dx, dy)
                dirx, diry = (dx / n, dy / n) if n > 1e-6 else (1.0, 0.0)

            yaw_deg = math.degrees(math.atan2(diry, dirx))
            pre  = (gx - PRE_GATE_OFFSET * dirx,  gy - PRE_GATE_OFFSET * diry,  gz)
            post = (gx + POST_GATE_OFFSET * dirx, gy + POST_GATE_OFFSET * diry, gz)
            waypoints.append({
                'pre': pre, 'gate': (gx, gy, gz), 'post': post, 'yaw': yaw_deg,
            })
        return waypoints

    def run_fast_lap(self, gates, n_laps=N_LAPS):
        """
        Part 2: gate positions are known. Fly N laps through the gates as fast
        as possible using position setpoints. Only the best lap counts, so
        lap times are printed at the end.
        """
        if not gates:
            print('GATE_POSITIONS is empty — fill it in before running Part 2.')
            return

        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2)

        lap_times = []
        try:
            self.takeoff(CRUISE_ALT)
            home_xy = (self._state['x'], self._state['y'])
            waypoints = self._build_gate_waypoints(gates, home_xy)

            for lap in range(n_laps):
                if self._stop:
                    break
                print(f'\n=== Lap {lap + 1} / {n_laps} ===')
                t_lap = time.time()

                for i, wp in enumerate(waypoints):
                    if self._stop:
                        break
                    print(f'  Gate {i + 1}/{len(waypoints)}  '
                          f'centre=({wp["gate"][0]:.2f},{wp["gate"][1]:.2f},'
                          f'{wp["gate"][2]:.2f})  yaw={wp["yaw"]:.0f}°')
                    # Approach the pre-gate point, then drive straight through
                    # to the post-gate point (the line passes through the gate).
                    self.goto(*wp['pre'],  wp['yaw'], speed=APPROACH_SPEED)
                    self.goto(*wp['post'], wp['yaw'], speed=TRANSIT_SPEED,
                              tol=GOTO_TOL * 2)

                # Return to the takeoff region to close the lap
                if not self._stop:
                    self.goto(home_xy[0], home_xy[1], CRUISE_ALT,
                              self._state['yaw'], speed=APPROACH_SPEED)

                lap_times.append(time.time() - t_lap)
                print(f'  Lap {lap + 1} time: {lap_times[-1]:.2f} s')

            print(f'\nLap times: {[f"{t:.2f}s" for t in lap_times]}')
            if lap_times:
                print(f'Best lap: {min(lap_times):.2f} s')

        except Exception as e:
            print(f'\nUnhandled exception during position mission: {e} — landing now')

        finally:
            try:
                self.land()
            except Exception as e:
                print(f'Land failed ({e}) — cutting motors')
                self._stop_motors()


# ── emergency stop ─────────────────────────────────────────────────────────────

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
    threading.Thread(
        target=emergency_stop_listener,
        args=(ctrl, cam, cf),
        daemon=True
    ).start()

    try:
        if MISSION == 'vision':
            ctrl.run_vision_lap()
        else:
            ctrl.run_fast_lap(GATE_POSITIONS, N_LAPS)
    finally:
        if cam is not None:
            cam.stop()
        cf.close_link()
