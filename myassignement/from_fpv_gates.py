#!/usr/bin/env python3
"""
FPV-based autonomous gate runner.

This keeps the UDP camera path and Qt display style from crazyflie_fpv_example.py,
but runs the lap-1 gate mission logic from hardware_lap1.py in a background
thread. The important difference from hardware_lap1.py is that camera reception is
driven by the same QThread pattern as the FPV example, which is known to stream
frames during manual flight.

Keys:
  Q      controlled landing
  Space  immediate stop setpoint, then close
"""

import contextlib
import os
import socket
import struct
import sys
import threading
import time

import cflib.crtp
import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper

from hardware_lap1 import Lap1Controller


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


# Same defaults as the FPV example in this folder.
URI = uri_helper.uri_from_env(default='radio://0/10/2M/E7E7E7E701')
AIDECK_IP = '192.168.4.1'
AIDECK_PORT = 5000
LOCAL_PORT = 5001
START_MAGIC = b'FER'

CPX_HEADER_SIZE = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE = 11
IMG_WIDTH = 324
IMG_HEIGHT = 244
MIN_JPEG_BYTES = 5000


class UdpGateVideoThread(QtCore.QThread):
    """FPV-style UDP receiver that also exposes the latest grayscale frame."""

    frame_ready = QtCore.pyqtSignal(np.ndarray)
    status_ready = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self._lock = threading.Lock()
        self._frame = None
        self._frame_ts = 0.0
        self._stats = {
            'packets': 0,
            'headers': 0,
            'decode_attempts': 0,
            'frames': 0,
            'decode_failures': 0,
            'timeouts': 0,
            'last_packet_ts': 0.0,
            'last_header_ts': 0.0,
        }

    def stop(self):
        self._running = False

    @property
    def latest_frame_with_ts(self):
        with self._lock:
            return self._frame, self._frame_ts

    def stats(self):
        with self._lock:
            stats = dict(self._stats)
            stats['has_frame'] = self._frame is not None
            stats['frame_ts'] = self._frame_ts
            return stats

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(('0.0.0.0', LOCAL_PORT))
        sock.settimeout(1.0)
        sock.sendto(START_MAGIC, (AIDECK_IP, AIDECK_PORT))

        buffer = bytearray()
        expected_size = 0
        receiving = False
        last_status = 0.0

        while self._running:
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                with self._lock:
                    self._stats['timeouts'] += 1
                sock.sendto(START_MAGIC, (AIDECK_IP, AIDECK_PORT))
                continue
            except OSError:
                break

            now = time.time()
            with self._lock:
                self._stats['packets'] += 1
                self._stats['last_packet_ts'] = now

            if len(data) < CPX_HEADER_SIZE:
                continue
            payload = data[CPX_HEADER_SIZE:]

            if len(payload) >= IMG_HEADER_SIZE and payload[0] == IMG_HEADER_MAGIC:
                _, w, h, _, _, size = struct.unpack('<BHHBBI', payload[:IMG_HEADER_SIZE])
                if w == IMG_WIDTH and h == IMG_HEIGHT and 0 < size < 65536:
                    with self._lock:
                        self._stats['headers'] += 1
                        self._stats['last_header_ts'] = now
                    expected_size = size
                    buffer = bytearray()
                    receiving = True
                    continue

            if not receiving:
                continue

            buffer.extend(payload)

            if len(buffer) >= expected_size:
                with self._lock:
                    self._stats['decode_attempts'] += 1
                frame = self._decode(buffer)
                if frame is None:
                    with self._lock:
                        self._stats['decode_failures'] += 1
                else:
                    with self._lock:
                        self._frame = frame
                        self._frame_ts = time.time()
                        self._stats['frames'] += 1
                    self.frame_ready.emit(frame)
                receiving = False

            if now - last_status >= 1.0:
                self.status_ready.emit(self._format_status())
                last_status = now

        sock.close()

    def _decode(self, buffer):
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
        if img is None or img.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
            return None
        if img.ndim == 2:
            return img
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return None

    def _format_status(self):
        st = self.stats()
        now = time.time()
        frame_age = now - st['frame_ts'] if st['frame_ts'] else float('inf')
        pkt_age = now - st['last_packet_ts'] if st['last_packet_ts'] else float('inf')
        return (
            f'pkts={st["packets"]} headers={st["headers"]} '
            f'frames={st["frames"]} fail={st["decode_failures"]} '
            f'timeouts={st["timeouts"]} frame_age={frame_age:.1f}s '
            f'pkt_age={pkt_age:.1f}s'
        )


class GateRunnerWindow(QtWidgets.QWidget):
    connected_signal = QtCore.pyqtSignal(str)
    connection_failed_signal = QtCore.pyqtSignal(str)
    disconnected_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('FPV Gates')

        self.image_label = QtWidgets.QLabel('Waiting for video...')
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_label = QtWidgets.QLabel(f'Connecting to {URI}...')
        self.camera_label = QtWidgets.QLabel('Camera: starting...')

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.camera_label)

        self.video = UdpGateVideoThread(self)
        self.video.frame_ready.connect(self._show_frame)
        self.video.status_ready.connect(self._show_camera_status)
        self.video.start()

        self.cf = Crazyflie(rw_cache='cache')
        self.ctrl = None
        self.mission_thread = None
        self._stop_requested = False

        self.monitor_timer = QtCore.QTimer(self)
        self.monitor_timer.timeout.connect(self._monitor_mission)
        self.monitor_timer.setInterval(500)
        self.monitor_timer.start()

        self.connected_signal.connect(self._on_connected)
        self.connection_failed_signal.connect(self._on_connection_failed)
        self.disconnected_signal.connect(self._on_disconnected)
        self.cf.connected.add_callback(self.connected_signal.emit)
        self.cf.connection_failed.add_callback(
            lambda uri, msg: self.connection_failed_signal.emit(msg)
        )
        self.cf.disconnected.add_callback(self.disconnected_signal.emit)
        self.cf.open_link(URI)

    def _on_connected(self, uri):
        self.status_label.setText(f'Connected to {uri}; starting autonomous gates')
        try:
            self.cf.supervisor.send_arming_request(True)
        except Exception:
            pass
        self.ctrl = Lap1Controller(self.cf, self.video)
        if not self.ctrl.is_connected:
            self.status_label.setText('Log setup failed; mission not started')
            return
        self.mission_thread = threading.Thread(
            target=self.ctrl.run_mission,
            daemon=True,
            name='GateMissionThread',
        )
        self.mission_thread.start()

    def _on_connection_failed(self, msg):
        self.status_label.setText(f'Connection failed: {msg}')

    def _on_disconnected(self, uri):
        self.status_label.setText(f'Disconnected from {uri}')

    def _show_camera_status(self, text):
        self.camera_label.setText(f'Camera: {text}')

    def _show_frame(self, gray):
        h, w = gray.shape
        qimg = QtGui.QImage(gray.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8).copy()
        self.image_label.setPixmap(
            QtGui.QPixmap.fromImage(qimg).scaled(
                w * 2,
                h * 2,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _request_stop(self):
        self._stop_requested = True
        if self.ctrl is not None:
            self.ctrl._stop = True
            self.status_label.setText('Stop requested; landing...')
        else:
            self.close()

    def _immediate_stop(self):
        self._stop_requested = True
        if self.ctrl is not None:
            self.ctrl._stop = True
        try:
            self.cf.commander.send_stop_setpoint()
        except Exception:
            pass
        self.close()

    def _monitor_mission(self):
        if (
            self._stop_requested
            and self.mission_thread is not None
            and not self.mission_thread.is_alive()
        ):
            self.close()

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        if event.key() == QtCore.Qt.Key.Key_Q:
            self._request_stop()
            return
        if event.key() == QtCore.Qt.Key.Key_Space:
            self._immediate_stop()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.ctrl is not None:
            self.ctrl._stop = True
        self.monitor_timer.stop()
        self.video.stop()
        self.video.wait(1000)
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
    cflib.crtp.init_drivers()
    app = QtWidgets.QApplication(sys.argv)
    win = GateRunnerWindow()
    win.resize(760, 640)
    win.show()
    sys.exit(app.exec())
