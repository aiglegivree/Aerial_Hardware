# -*- coding: utf-8 -*-
"""
Gate traversal controller — Crazyflie + Lighthouse (POSITION mode only)
========================================================================

Part 2 — raw waypoint streaming (Lighthouse world frame):
  Build one waypoint list per lap (start → pre-gate, gate centre, post-gate
  ×N → return). Stream each waypoint via send_position_setpoint and advance
  to the next once the drone is within WAYPOINT_REACH_TOL. The Crazyflie's
  onboard position controller smooths the motion between waypoints; no
  polynomial trajectory fit needed.

Boundary safety:
  stateEstimate.x/y from Lighthouse → enforce arena hard limits.

Press 'q' at any time for emergency stop (cuts motors).
Press ESC for a controlled landing.
"""

import csv
import logging
import math
import os
import time
import threading
import warnings

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

# ── Part 2: position-based mission ─────────────────────────────────────────────
#
# Gate positions are loaded from gates_info.csv beside this script.
# CSV columns: Gate,x,y,z,theta,width,height
# Each gate entry is (x, y, z, theta, width, height). theta is already radians.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GATES_INFO_CSV = os.path.join(SCRIPT_DIR, 'gates_info.csv')


def load_gate_positions(csv_path=GATES_INFO_CSV):
    required = {'x', 'y', 'z', 'theta', 'width', 'height'}
    gates = []

    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f'{csv_path} is empty')

            fieldnames = {name.strip() for name in reader.fieldnames}
            missing = required - fieldnames
            if missing:
                missing_cols = ', '.join(sorted(missing))
                raise ValueError(f'{csv_path} is missing column(s): {missing_cols}')

            for row_idx, row in enumerate(reader, start=2):
                row = {k.strip(): (v.strip() if v is not None else '') for k, v in row.items()}
                if not any(row.values()):
                    continue
                gate_id = row.get('Gate', row_idx)
                try:
                    gate_no = int(gate_id)
                    gate = (
                        float(row['x']),
                        float(row['y']),
                        float(row['z']),
                        float(row['theta']),
                        float(row['width']),
                        float(row['height']),
                    )
                except ValueError as e:
                    raise ValueError(f'Invalid numeric value in {csv_path}, row {row_idx}') from e
                gates.append((gate_no, gate))
    except FileNotFoundError:
        print(f'Gate CSV not found: {csv_path}')
        return []

    gates.sort(key=lambda item: item[0])
    gate_positions = [gate for _, gate in gates]
    print(f'Loaded {len(gate_positions)} gate(s) from {csv_path}')
    return gate_positions


GATE_POSITIONS = load_gate_positions()

N_LAPS           = 2     # number of timed laps
PRE_GATE_OFFSET  = 0.3   # m — waypoint placed before the gate along its approach axis
POST_GATE_OFFSET = 0.3   # m — waypoint placed after the gate (clears the frame)

POSITION_RATE_HZ   = 20.0  # setpoint streaming rate
WAYPOINT_REACH_TOL = 0.15  # m — final-waypoint reached tolerance
PURSUIT_LOOKAHEAD  = 0.35  # m — carrot distance ahead of drone along the path
                            #     larger = smoother + faster, smaller = tighter tracking


# ── helpers ────────────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ── flight controller ──────────────────────────────────────────────────────────

class GateController:

    def __init__(self, cf: Crazyflie):
        self._cf  = cf

        self.is_connected = False
        self._stop        = False
        self._state       = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}

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

    # ── boundary-safe hover ──────────────────────────────────────────────────

    def _safe_hover(self, vx=0.0, vy=0.0, yaw_rate=0.0, z=CRUISE_ALT):
        """
        Send hover setpoint, blocking outward velocity at the arena boundary.
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
        (list of (x, y, z, yaw_rad)).
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
        gates = list(gates)
        if not gates:
            print('GATE_POSITIONS is empty — fill it in before running Part 2.')
            return
        print(f'Position mission will use {len(gates)} gate(s).')

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

def emergency_stop_listener(ctrl: GateController, cf: Crazyflie):
    """ESC → controlled landing. Q → immediate motor kill."""

    def on_press(key):
        if key == keyboard.Key.esc:
            print('\n[EMERGENCY STOP] landing')
            ctrl._stop = True
            try:
                ctrl.land()
            except Exception as e:
                print(f'Land failed ({e}) — cutting motors')
                ctrl._stop_motors()
            finally:
                cf.close_link()
            return False

        try:
            if key.char == 'q':
                print('\n[EMERGENCY STOP] cutting motors immediately')
                ctrl._stop = True
                ctrl._stop_motors()
                cf.close_link()
                return False
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


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
        exit(1)

    ctrl = GateController(cf)
    if not ctrl.is_connected:
        print('Log setup failed — exiting')
        cf.close_link()
        exit(1)

    emergency_stop_thread = threading.Thread(
        target=emergency_stop_listener, args=(ctrl, cf), daemon=True)
    emergency_stop_thread.start()

    try:
        ctrl.run_fast_lap(GATE_POSITIONS, N_LAPS)
    finally:
        cf.close_link()


if __name__ == '__main__':
    main()
