#!/usr/bin/env python3
import logging
import msvcrt
import struct
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from threading import Event

import cv2
import cflib.crtp
import numpy as np
from cflib.cpx import CPXFunction
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper

logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings("ignore", message=".*supervisor subsystem requires CRTP.*")

URI = uri_helper.uri_from_env(default="tcp://192.168.4.1:5000")
CAM_WIDTH = 324
CAM_HEIGHT = 244
FPS = 20.0


def receive_frame(cpx):
    while True:
        packet = cpx.receivePacket(CPXFunction.APP)
        magic, width, height, depth, fmt, size = struct.unpack("<BHHBBI", packet.data[0:11])
        if magic != 0xBC:
            continue

        buf = bytearray()
        while len(buf) < size:
            buf.extend(cpx.receivePacket(CPXFunction.APP).data)

        frame = np.frombuffer(buf[:size], dtype=np.uint8)
        if frame.size != CAM_WIDTH * CAM_HEIGHT:
            raise RuntimeError(f"Unexpected frame size: {frame.size} bytes")

        bayer = frame.reshape((CAM_HEIGHT, CAM_WIDTH))
        return cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)


def main():
    cflib.crtp.init_drivers()
    cf = Crazyflie(ro_cache=None, rw_cache="cache")
    connected_event = Event()
    failed_event = Event()
    connection_error = {"message": None}

    def on_connected(uri):
        connected_event.set()

    def on_connection_failed(uri, errmsg):
        connection_error["message"] = errmsg
        failed_event.set()

    def on_disconnected(uri):
        if not connected_event.is_set():
            connection_error["message"] = "Disconnected before the camera stream was ready."
            failed_event.set()

    cf.connected.add_callback(on_connected)
    cf.connection_failed.add_callback(on_connection_failed)
    cf.disconnected.add_callback(on_disconnected)
    cf.open_link(URI)

    if not connected_event.wait(timeout=10):
        if failed_event.is_set() and connection_error["message"]:
            raise RuntimeError(connection_error["message"])
        raise RuntimeError("Timed out waiting for the Crazyflie connection.")

    writer = None
    output_path = None

    print("Connected. Press R to start recording. Press Q to stop and save.")

    try:
        while True:
            frame_bgr = receive_frame(cf.link.cpx)

            if writer is not None:
                writer.write(frame_bgr)

            key = None
            if msvcrt.kbhit():
                key = msvcrt.getwch().lower()

            if key == "r" and writer is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path(__file__).resolve().parent / f"camera_recording_{timestamp}.mp4"
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    FPS,
                    (frame_bgr.shape[1], frame_bgr.shape[0]),
                )
                if not writer.isOpened():
                    writer.release()
                    writer = None
                    output_path = None
                    print("Failed to open MP4 writer.")
                else:
                    print(f"Recording started: {output_path.name}")

            if key == "q":
                if writer is not None:
                    writer.release()
                    writer = None
                    print(f"Recording saved: {output_path}")
                else:
                    print("Stopped without an active recording.")
                break

    finally:
        if writer is not None:
            writer.release()
        cf.close_link()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
