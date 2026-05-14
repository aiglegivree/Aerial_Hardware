# -*- coding: utf-8 -*-
"""
Gate traversal controller — Crazyflie + AI deck + Lighthouse
=============================================================

Split links (like FPV example):
    • Control/logs via Crazyradio (CRTP)
    • Video via AI-deck UDP stream (JPEG)

Architecture:
    UdpVideoThread — listens to UDP packets from AI-deck,
                                     decodes JPEG frames, stores latest in a shared slot.
    GateController — sends hover setpoints at ~20 Hz over radio,
                                     reads Lighthouse state from the log callback,
                                     reads camera detections from UdpVideoThread.

Mission:
    takeoff → 180° yaw pan → land

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

CRUISE_ALT       = 0.6   # m
TAKEOFF_DURATION = 3.0   # s
LAND_DURATION    = 3.0   # s

PAN_YAW_RATE = 30.0  # deg/s
PAN_ANGLE    = 180.0 # deg

FRAME_SAVE_DIR = os.path.join('myassignement', 'frames')

# state machine

STATE_TAKEOFF = 'takeoff'
STATE_PAN     = 'pan'
STATE_FORWARD = 'forward'
STATE_BACK    = 'back'
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
    """
    Reads AI-deck UDP packets, decodes JPEG, stores grayscale frames via .latest_frame.
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
        self._cf.commander.send_hover_setpoint(vx, vy, yaw_rate, z)

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

    # ── yaw pan ───────────────────────────────────────────────────────────

    def pan_around(self, angle_deg=PAN_ANGLE, yaw_rate_deg=PAN_YAW_RATE):
        print(f'Panning {angle_deg:.0f}° at {yaw_rate_deg:.1f}°/s')
        duration = abs(angle_deg) / max(abs(yaw_rate_deg), 1e-6)
        yaw_rate = yaw_rate_deg if angle_deg >= 0 else -abs(yaw_rate_deg)
        t_end = time.time() + duration
        while time.time() < t_end and not self._stop:
            self._safe_hover(yaw_rate=yaw_rate, z=CRUISE_ALT)
            time.sleep(0.05)

    # ── mission ──────────────────────────────────────────────────────────────

    def run_mission(self):
        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2)

        # try:
        #     self.takeoff()
        #     if not self._stop:
        #         self.pan_around()
        #     msg = 'Pan complete' if not self._stop else 'Emergency stop'
        #     print(f'\n{msg} — landing')
        #     # go to 1 m in from
        #     self._cf.commander.send_position_setpoint(1.0, 0.0, CRUISE_ALT, 0.0)
        #     time.sleep(5.0)
        #     self._cf.commander.send_position_setpoint(0.0, 0.0, CRUISE_ALT, 0.0)
        #     time.sleep(5.0)
        #     # land
        #     self.land()

        # except Exception as e:
        #     print(f'\nUnhandled exception during mission: {e} — landing now')

        # finally:
        #     # Always attempt a controlled landing, whatever happened above.
        #     # _stop_motors() is NOT called here — land() does a gradual descent
        #     # and only cuts motors at the end, so the drone doesn't just drop.
        #     try:
        #         self.land()
        #     except Exception as e:
        #         print(f'Land failed ({e}) — cutting motors')
        #         self._stop_motors()

        state = STATE_TAKEOFF

        # timers / flags
        state_start = time.time()

        try:

            while not self._stop and state != STATE_DONE:

                # ─────────────────────────────────────────────
                # TAKEOFF
                # ─────────────────────────────────────────────
                if state == STATE_TAKEOFF:

                    print('[STATE] TAKEOFF')

                    self.takeoff()

                    state = STATE_PAN
                    state_start = time.time()

                # ─────────────────────────────────────────────
                # PAN
                # ─────────────────────────────────────────────
                elif state == STATE_PAN:

                    print('[STATE] PAN')

                    self.pan_around()

                    state = STATE_FORWARD
                    state_start = time.time()

                # ─────────────────────────────────────────────
                # FORWARD
                # ─────────────────────────────────────────────
                elif state == STATE_FORWARD:

                    print('[STATE] FORWARD')

                    duration = 3.0
                    t_end = time.time() + duration

                    while time.time() < t_end and not self._stop:
                        self._safe_hover(vx=0.2, z=CRUISE_ALT)
                        time.sleep(0.05)

                    state = STATE_BACK
                    state_start = time.time()

                # ─────────────────────────────────────────────
                # BACK
                # ─────────────────────────────────────────────
                elif state == STATE_BACK:

                    print('[STATE] BACK')

                    duration = 3.0
                    t_end = time.time() + duration

                    while time.time() < t_end and not self._stop:
                        self._safe_hover(vx=-0.2, z=CRUISE_ALT)
                        time.sleep(0.05)

                    state = STATE_LAND
                    state_start = time.time()

                # ─────────────────────────────────────────────
                # LAND
                # ─────────────────────────────────────────────
                elif state == STATE_LAND:

                    print('[STATE] LAND')

                    self.land()

                    state = STATE_DONE

            print('Mission complete')

        except Exception as e:

            print(f'Unhandled exception: {e}')

            try:
                self.land()
            except:
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

    # Start UDP camera thread (AI-deck WiFi stream)
    cam = UdpVideoThread()
    cam.start()

    # Camera thread runs asynchronously (no blocking wait, like FPV control)

    # Build controller (sets up log variables)
    ctrl = GateController(cf, cam)
    if not ctrl.is_connected:
        print('Log setup failed — exiting')
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
        ctrl.run_mission()
    finally:
        cam.stop()
        cf.close_link()