#!/usr/bin/env python3
"""
Autonomous FPV — Crazyflie + AI-deck

Combines the live FPV Qt window from crazyflie_fpv_example.py with the
autonomous state-machine flight from circle_around.py.

  Video    : UDP stream from the AI-deck (shown in the Qt window)
  Control  : cflib over Crazyradio, driven by a worker thread that runs
             the takeoff -> pan -> forward -> back -> land state machine.

Keys:
  Space  : emergency stop (cut motors, stop mission)
  Esc    : same as Space + close window
"""
import contextlib
import math
import os
import socket
import struct
import sys
import threading
import time

import cflib.crtp
import cv2
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from PyQt6 import QtCore, QtGui, QtWidgets


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


# --- Configure these for your setup ---
URI         = 'radio://0/20/2M/E7E7E7E708'
AIDECK_IP   = '192.168.4.1'
AIDECK_PORT = 5000
LOCAL_PORT  = 5001
START_MAGIC = b'FER'

CPX_HEADER_SIZE  = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE  = 11
IMG_WIDTH        = 324
IMG_HEIGHT       = 244
MIN_JPEG_BYTES   = 5000

# --- Flight parameters (from circle_around.py) ---
CRUISE_ALT       = 0.6   # m
TAKEOFF_DURATION = 3.0   # s
LAND_DURATION    = 3.0   # s
PAN_YAW_RATE     = 30.0  # deg/s
PAN_ANGLE        = 180.0 # deg

# --- State-machine labels ---
STATE_TAKEOFF = 'takeoff'
STATE_PAN     = 'pan'
STATE_FORWARD = 'forward'
STATE_BACK    = 'back'
STATE_LAND    = 'land'
STATE_DONE    = 'done'


class UdpVideoThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(('0.0.0.0', LOCAL_PORT))
        sock.sendto(START_MAGIC, (AIDECK_IP, AIDECK_PORT))

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
                if w == IMG_WIDTH and h == IMG_HEIGHT and 0 < size < 65536:
                    expected_size = size
                    buffer = bytearray()
                    receiving = True
                    continue

            if not receiving:
                continue

            buffer.extend(payload)

            if len(buffer) >= expected_size:
                self._decode_and_emit(buffer)
                receiving = False

    def _decode_and_emit(self, buffer):
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
        if img is None or img.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
            return
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.frame_ready.emit(img)


class MissionThread(QtCore.QThread):
    """Runs the autonomous state machine off the Qt event loop."""

    state_changed = QtCore.pyqtSignal(str)
    mission_done  = QtCore.pyqtSignal()

    def __init__(self, cf: Crazyflie, parent=None):
        super().__init__(parent)
        self._cf    = cf
        self._stop  = False
        self._state = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
        self._log_ready = threading.Event()
        self._setup_log()

    # ── log setup ────────────────────────────────────────────────────────────
    def _setup_log(self):
        lg = LogConfig(name='State', period_in_ms=100)
        lg.add_variable('stateEstimate.x', 'float')
        lg.add_variable('stateEstimate.y', 'float')
        lg.add_variable('stateEstimate.z', 'float')
        lg.add_variable('stabilizer.yaw',  'float')
        try:
            self._cf.log.add_config(lg)
            lg.data_received_cb.add_callback(self._log_cb)
            lg.start()
        except Exception as e:
            print(f'Log setup failed: {e}')

    def _log_cb(self, timestamp, data, logconf):
        self._state['x']   = data['stateEstimate.x']
        self._state['y']   = data['stateEstimate.y']
        self._state['z']   = data['stateEstimate.z']
        self._state['yaw'] = data['stabilizer.yaw']
        self._log_ready.set()

    # ── setpoint helpers ─────────────────────────────────────────────────────
    def _safe_hover(self, vx=0.0, vy=0.0, yaw_rate=0.0, z=CRUISE_ALT):
        self._cf.commander.send_hover_setpoint(vx, vy, yaw_rate, z)

    def _stop_motors(self):
        self._cf.commander.send_stop_setpoint()

    # ── primitives (lifted from circle_around.py) ────────────────────────────
    def takeoff(self, target_z=CRUISE_ALT):
        print(f'Taking off to {target_z:.2f} m')
        steps = int(TAKEOFF_DURATION / 0.1)
        for i in range(steps):
            if self._stop:
                return
            self._safe_hover(z=target_z * (i / steps))
            time.sleep(0.1)
        for _ in range(20):
            if self._stop:
                return
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

    def pan_around(self, angle_deg=PAN_ANGLE, yaw_rate_deg=PAN_YAW_RATE):
        print(f'Panning {angle_deg:.0f} deg at {yaw_rate_deg:.1f} deg/s')
        duration = abs(angle_deg) / max(abs(yaw_rate_deg), 1e-6)
        yaw_rate = yaw_rate_deg if angle_deg >= 0 else -abs(yaw_rate_deg)
        t_end = time.time() + duration
        while time.time() < t_end and not self._stop:
            self._safe_hover(yaw_rate=yaw_rate, z=CRUISE_ALT)
            time.sleep(0.05)

    # ── public API ───────────────────────────────────────────────────────────
    def request_stop(self):
        self._stop = True

    # ── mission ──────────────────────────────────────────────────────────────
    def run(self):
        # Reset Kalman estimator before flying
        self._cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        self._cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2.0)

        # Wait briefly for a log sample (so land() has a valid z)
        self._log_ready.wait(timeout=5.0)

        state = STATE_TAKEOFF
        self.state_changed.emit(state)

        try:
            while not self._stop and state != STATE_DONE:

                if state == STATE_TAKEOFF:
                    print('[STATE] TAKEOFF')
                    self.takeoff()
                    state = STATE_PAN
                    self.state_changed.emit(state)

                elif state == STATE_PAN:
                    print('[STATE] PAN')
                    self.pan_around()
                    state = STATE_FORWARD
                    self.state_changed.emit(state)

                elif state == STATE_FORWARD:
                    print('[STATE] FORWARD')
                    t_end = time.time() + 3.0
                    while time.time() < t_end and not self._stop:
                        self._safe_hover(vx=0.2, z=CRUISE_ALT)
                        time.sleep(0.05)
                    state = STATE_BACK
                    self.state_changed.emit(state)

                elif state == STATE_BACK:
                    print('[STATE] BACK')
                    t_end = time.time() + 3.0
                    while time.time() < t_end and not self._stop:
                        self._safe_hover(vx=-0.2, z=CRUISE_ALT)
                        time.sleep(0.05)
                    state = STATE_LAND
                    self.state_changed.emit(state)

                elif state == STATE_LAND:
                    print('[STATE] LAND')
                    self.land()
                    state = STATE_DONE
                    self.state_changed.emit(state)

            print('Mission complete')

        except Exception as e:
            print(f'Unhandled exception: {e}')
            try:
                self.land()
            except Exception:
                self._stop_motors()

        finally:
            self.mission_done.emit()


class FPVWindow(QtWidgets.QWidget):
    _connected_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Crazyflie FPV — Autonomous')

        self.image_label  = QtWidgets.QLabel('Waiting for video...')
        self.status_label = QtWidgets.QLabel(f'Connecting to {URI}...')
        self.state_label  = QtWidgets.QLabel('State: —')
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.state_label)

        self.video = UdpVideoThread(self)
        self.video.frame_ready.connect(self._show_frame)
        self.video.start()

        cflib.crtp.init_drivers()
        self.cf = Crazyflie(rw_cache='cache')
        self._connected_signal.connect(self._on_connected)
        self.cf.connected.add_callback(self._connected_signal.emit)
        self.cf.open_link(URI)

        self.mission = None

    def _on_connected(self, uri):
        self.status_label.setText(f'Connected to {uri} — starting mission')
        try:
            self.cf.supervisor.send_arming_request(True)
        except Exception as e:
            print(f'Arming request failed (continuing): {e}')

        self.mission = MissionThread(self.cf, self)
        self.mission.state_changed.connect(
            lambda s: self.state_label.setText(f'State: {s}'))
        self.mission.mission_done.connect(
            lambda: self.status_label.setText('Mission complete'))
        self.mission.start()

    def _show_frame(self, img):
        if img.ndim == 2:
            h, w = img.shape
            qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
        else:
            h, w, _ = img.shape
            qimg = QtGui.QImage(img.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg.scaled(w * 2, h * 2)))

    def _emergency_stop(self):
        if self.mission is not None:
            self.mission.request_stop()
        try:
            self.cf.commander.send_stop_setpoint()
        except Exception:
            pass
        self.status_label.setText('EMERGENCY STOP — motors cut')

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        k = event.key()
        if k == QtCore.Qt.Key.Key_Space:
            self._emergency_stop()
        elif k == QtCore.Qt.Key.Key_Escape:
            self._emergency_stop()
            self.close()

    def closeEvent(self, event):
        if self.mission is not None:
            self.mission.request_stop()
            self.mission.wait(2000)
        try:
            self.cf.commander.send_stop_setpoint()
        except Exception:
            pass
        try:
            self.cf.close_link()
        except Exception:
            pass
        event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = FPVWindow()
    win.show()
    sys.exit(app.exec())
