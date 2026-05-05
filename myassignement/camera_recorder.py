#!/usr/bin/env python3
import logging
import struct
import sys
import threading
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import cflib.crtp
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from cflib.cpx import CPXFunction
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper

logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings("ignore", message=".*supervisor subsystem requires CRTP.*")

URI = uri_helper.uri_from_env(default="tcp://192.168.4.1:5000")
CAM_WIDTH = 324
CAM_HEIGHT = 244
FPS = 20.0


class ImageThread(threading.Thread):
    def __init__(self, cpx, callback):
        super().__init__(daemon=True)
        self._cpx = cpx
        self._cb = callback
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            while self._running:
                packet = self._cpx.receivePacket(CPXFunction.APP)
                header = packet.data[0:11]
                magic, width, height, depth, fmt, size = struct.unpack("<BHHBBI", header)
                if magic != 0xBC:
                    continue

                buf = bytearray()
                while self._running and len(buf) < size:
                    buf.extend(self._cpx.receivePacket(CPXFunction.APP).data)

                if self._running:
                    # Make the frame own its memory before handing it to Qt.
                    self._cb(np.frombuffer(buf[:size], dtype=np.uint8).copy())
        except Exception:
            traceback.print_exc()


class CameraWindow(QtWidgets.QWidget):
    frame_received = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crazyflie Camera Recorder")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self.image_label = QtWidgets.QLabel("Waiting for video...")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_label = QtWidgets.QLabel("Connecting...")
        self.help_label = QtWidgets.QLabel("Press R to start recording, Q to stop and save.")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.help_label)
        self.setLayout(layout)

        self._last_frame_bgr = None
        self._writer = None
        self._is_recording = False
        self._output_path = None

        self.frame_received.connect(self._handle_frame)

        cflib.crtp.init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache="cache")
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.open_link(URI)

        if not self.cf.link:
            raise RuntimeError("Could not connect to the Crazyflie video link.")

        self._img_thread = ImageThread(self.cf.link.cpx, self.frame_received.emit)
        self._img_thread.start()

    def _connected(self, uri):
        self.status_label.setText(f"Connected to {uri}")

    def _disconnected(self, uri):
        self.status_label.setText(f"Disconnected from {uri}")

    def _handle_frame(self, img):
        if img.size != CAM_WIDTH * CAM_HEIGHT:
            self.status_label.setText(f"Unexpected frame size: {img.size} bytes")
            return

        bayer = img.reshape((CAM_HEIGHT, CAM_WIDTH))
        frame_bgr = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)
        self._last_frame_bgr = frame_bgr

        if self._is_recording and self._writer is not None:
            self._writer.write(frame_bgr)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = frame_rgb.shape
        qimage = QtGui.QImage(
            frame_rgb.data,
            width,
            height,
            width * channels,
            QtGui.QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(qimage).scaled(
            width * 2,
            height * 2,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)

    def _start_recording(self):
        if self._is_recording:
            self.status_label.setText(f"Already recording to {self._output_path.name}")
            return

        if self._last_frame_bgr is None:
            self.status_label.setText("No camera frame yet. Wait for video before pressing R.")
            return

        output_dir = Path(__file__).resolve().parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_path = output_dir / f"camera_recording_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(self._output_path),
            fourcc,
            FPS,
            (self._last_frame_bgr.shape[1], self._last_frame_bgr.shape[0]),
        )

        if not self._writer.isOpened():
            self._writer = None
            self._output_path = None
            self.status_label.setText("Failed to open MP4 writer.")
            return

        self._is_recording = True
        self.status_label.setText(f"Recording to {self._output_path.name}")

    def _stop_recording(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        saved_path = self._output_path
        self._is_recording = False
        self._output_path = None
        return saved_path

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return

        key = event.key()
        if key == QtCore.Qt.Key.Key_R:
            self._start_recording()
            return

        if key == QtCore.Qt.Key.Key_Q:
            saved_path = self._stop_recording()
            if saved_path is not None:
                self.status_label.setText(f"Saved video to {saved_path.name}")
            else:
                self.status_label.setText("Stopped without an active recording.")
            self.close()
            return

        super().keyPressEvent(event)

    def closeEvent(self, event):
        self._stop_recording()
        if hasattr(self, "_img_thread") and self._img_thread is not None:
            self._img_thread.stop()
        if hasattr(self, "cf") and self.cf is not None:
            self.cf.close_link()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CameraWindow()
    window.resize(800, 700)
    window.show()
    sys.exit(app.exec())
