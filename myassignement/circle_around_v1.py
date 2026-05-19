# -*- coding: utf-8 -*-
"""
Step 1 — Take off, fly one 1 m CCW lap, land.
=============================================

Crazyflie + AI-deck + Lighthouse. Same architecture as circle_around.py
(separate Crazyradio control link and UDP video link from the AI-deck).

Mission:
    1. Take off to 1.5 m at the current (start_x, start_y).
    2. Fly one full counter-clockwise lap on a circle of radius 1 m whose
       centre is 1 m in front of the take-off position (along the start yaw).
    3. Land.

Position setpoints are used throughout — Lighthouse is accurate enough
(<1 cm) that closing the loop on absolute (x, y, z, yaw) is more reliable
than tracking a velocity profile.

Press 'q' for emergency stop (a second 'q' cuts motors immediately).
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

# ── flight ─────────────────────────────────────────────────────────────────────

CRUISE_ALT       = 1.5    # m — hover altitude for this test
TAKEOFF_DURATION = 3.0    # s — linear ramp from start_z to CRUISE_ALT
LAND_DURATION    = 3.0    # s
SETPOINT_PERIOD  = 0.05   # s (20 Hz)

CIRCLE_RADIUS   = 1.0     # m
CIRCLE_OFFSET   = 1.0     # m — centre is this far in front of start, along start yaw
CIRCLE_OMEGA    = 0.35    # rad/s — angular speed along the circle (CCW)

FRAME_SAVE_DIR = os.path.join('myassignement', 'frames')

# state machine

STATE_TAKEOFF = 'takeoff'
STATE_CIRCLE  = 'circle'
STATE_LAND    = 'land'
STATE_DONE    = 'done'


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
    """Reads AI-deck UDP packets, decodes JPEG, exposes the latest frame."""

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
        os.makedirs(FRAME_SAVE_DIR, exist_ok=True)
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
                    timestamp = int(time.time() * 1000)
                    frame_path = os.path.join(FRAME_SAVE_DIR, f'{timestamp}.jpg')
                    cv2.imwrite(frame_path, frame)
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
            return img
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return None


# ── helpers ────────────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ── flight controller ──────────────────────────────────────────────────────────

class GateController:

    def __init__(self, cf: Crazyflie, cam: UdpVideoThread):
        self._cf  = cf
        self._cam = cam

        self.is_connected = False
        self._stop        = False
        self._state_ready = threading.Event()
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
        self._state['yaw'] = data['stabilizer.yaw']  # degrees
        self._state_ready.set()

    # ── setpoint wrappers ────────────────────────────────────────────────────

    def _send_pos(self, x, y, z, yaw_deg):
        self._cf.commander.send_position_setpoint(x, y, z, yaw_deg)

    def _stop_motors(self):
        self._cf.commander.send_stop_setpoint()

    # ── take-off / landing using position setpoints ──────────────────────────

    def takeoff(self, start_x, start_y, start_yaw_deg, target_z=CRUISE_ALT):
        print(f'Taking off to {target_z:.2f} m at ({start_x:.2f}, {start_y:.2f})')
        start_z = self._state['z']
        steps = max(1, int(TAKEOFF_DURATION / SETPOINT_PERIOD))
        for i in range(1, steps + 1):
            if self._stop:
                return
            z = start_z + (target_z - start_z) * (i / steps)
            self._send_pos(start_x, start_y, z, start_yaw_deg)
            time.sleep(SETPOINT_PERIOD)
        # settle at target
        for _ in range(20):
            if self._stop:
                return
            self._send_pos(start_x, start_y, target_z, start_yaw_deg)
            time.sleep(SETPOINT_PERIOD)

    def land(self):
        print('Landing')
        x = self._state['x']
        y = self._state['y']
        yaw = self._state['yaw']
        start_z = max(self._state['z'], 0.1)
        steps = max(1, int(LAND_DURATION / SETPOINT_PERIOD))
        for i in range(1, steps + 1):
            z = start_z * (1.0 - i / steps)
            self._send_pos(x, y, max(z, 0.0), yaw)
            time.sleep(SETPOINT_PERIOD)
        self._stop_motors()

    # ── 1 m radius CCW lap ───────────────────────────────────────────────────

    def circle_lap(self, start_x, start_y, start_yaw_deg, z=CRUISE_ALT,
                   radius=CIRCLE_RADIUS, offset=CIRCLE_OFFSET, omega=CIRCLE_OMEGA):
        """One full CCW lap around a circle whose centre is `offset` metres
        in front of (start_x, start_y) along the start yaw. The drone faces
        the direction of motion (tangent to the circle)."""
        yaw_rad = math.radians(start_yaw_deg)
        cx = start_x + offset * math.cos(yaw_rad)
        cy = start_y + offset * math.sin(yaw_rad)

        # The drone currently sits at (start_x, start_y), i.e. on the circle
        # at angle theta0 = yaw + pi as seen from the centre.
        theta0 = yaw_rad + math.pi
        duration = 2.0 * math.pi / omega

        # Tangent direction for CCW motion is theta + pi/2.
        initial_tangent_yaw_deg = math.degrees(theta0 + math.pi / 2.0)

        print(f'Circle: centre=({cx:.2f}, {cy:.2f}), r={radius:.2f} m, '
              f'lap={duration:.1f} s')

        # Rotate in place to the initial tangent heading before starting.
        for _ in range(40):  # ~2 s
            if self._stop:
                return
            self._send_pos(start_x, start_y, z, initial_tangent_yaw_deg)
            time.sleep(SETPOINT_PERIOD)

        t0 = time.time()
        while not self._stop:
            t = time.time() - t0
            if t >= duration:
                break
            theta = theta0 + omega * t  # CCW: theta increases
            x = cx + radius * math.cos(theta)
            y = cy + radius * math.sin(theta)
            yaw_target_rad = theta + math.pi / 2.0  # CCW tangent
            self._send_pos(x, y, z, math.degrees(yaw_target_rad))
            time.sleep(SETPOINT_PERIOD)

        # Rotate back to the start heading and hold briefly before landing.
        for _ in range(40):  # ~2 s
            if self._stop:
                return
            self._send_pos(start_x, start_y, z, start_yaw_deg)
            time.sleep(SETPOINT_PERIOD)

    # ── mission ──────────────────────────────────────────────────────────────

    def run_mission(self):
        # Reset the Kalman estimator (Lighthouse will repopulate it).
        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2.0)

        # Make sure we have at least one log sample before reading start state.
        if not self._state_ready.wait(timeout=3.0):
            print('No state estimate received — aborting')
            return

        start_x   = self._state['x']
        start_y   = self._state['y']
        start_yaw = self._state['yaw']  # degrees
        print(f'Start state: x={start_x:.2f}  y={start_y:.2f}  '
              f'z={self._state["z"]:.2f}  yaw={start_yaw:.1f}')

        state = STATE_TAKEOFF
        try:
            while not self._stop and state != STATE_DONE:

                if state == STATE_TAKEOFF:
                    print('[STATE] TAKEOFF')
                    self.takeoff(start_x, start_y, start_yaw)
                    state = STATE_CIRCLE

                elif state == STATE_CIRCLE:
                    print('[STATE] CIRCLE')
                    self.circle_lap(start_x, start_y, start_yaw)
                    state = STATE_LAND

                elif state == STATE_LAND:
                    print('[STATE] LAND')
                    self.land()
                    state = STATE_DONE

            print('Mission complete')

        except Exception as e:
            print(f'Unhandled exception: {e}')
            try:
                self.land()
            except Exception:
                self._stop_motors()


# ── emergency stop ─────────────────────────────────────────────────────────────

def emergency_stop_listener(ctrl: GateController, cam: UdpVideoThread, cf: Crazyflie):
    """Press 'q' for a controlled landing; press it again to cut motors."""
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
                cam.stop()
                cf.close_link()
            return False

        elif stop_count[0] >= 2:
            print('\n[EMERGENCY STOP] cutting motors immediately')
            ctrl._stop_motors()
            cam.stop()
            cf.close_link()
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cflib.crtp.init_drivers()

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

    cam = UdpVideoThread()
    cam.start()

    ctrl = GateController(cf, cam)
    if not ctrl.is_connected:
        print('Log setup failed — exiting')
        cam.stop()
        cf.close_link()
        exit(1)

    threading.Thread(
        target=emergency_stop_listener,
        args=(ctrl, cam, cf),
        daemon=True
    ).start()

    try:
        ctrl.run_mission()
    finally:
        cam.stop()
        cf.close_link()
