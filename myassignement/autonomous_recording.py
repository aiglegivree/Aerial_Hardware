import logging
import time
import threading
import struct
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from pynput import keyboard

import cflib.crtp
from cflib.cpx import CPXFunction
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper

# We use the TCP URI because it supports both flight control and CPX image streaming
URI = uri_helper.uri_from_env(default='tcp://192.168.4.1:5000')
CAM_WIDTH = 324
CAM_HEIGHT = 244

logging.basicConfig(level=logging.ERROR)

class ImageThread(threading.Thread):
    def __init__(self, cpx, callback):
        super().__init__(daemon=True)
        self._cpx = cpx
        self._cb = callback

    def run(self):
        while True:
            try:
                p = self._cpx.receivePacket(CPXFunction.APP)
                [magic, width, height, depth, fmt, size] = struct.unpack('<BHHBBI', p.data[0:11])
                if magic == 0xBC:
                    buf = bytearray()
                    while len(buf) < size:
                        buf.extend(self._cpx.receivePacket(CPXFunction.APP).data)
                    self._cb(np.frombuffer(buf, dtype=np.uint8))
            except Exception as e:
                pass


def emergency_stop_callback(cf):
    def on_press(key):
        try:
            if key.char == 'q':
                print("Emergency stop triggered!")
                cf.commander.send_stop_setpoint()
                cf.close_link()
                return False
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == '__main__':
    cflib.crtp.init_drivers()
    cf = Crazyflie(ro_cache=None, rw_cache='./cache')

    is_connected = False
    def _connected(link_uri):
        global is_connected
        print(f"Connected to {link_uri}")
        is_connected = True

    cf.connected.add_callback(_connected)

    print('Connecting to %s' % URI)
    cf.open_link(URI)

    # Wait until fully connected
    while not is_connected:
        time.sleep(0.1)

    cf.supervisor.send_arming_request(True)
    time.sleep(1.0)

    # Setup timestamped folder for images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = Path(__file__).resolve().parent / 'frames' / timestamp
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"Recording images to {frames_dir}")

    frame_index = 0
    def _update_image(img):
        global frame_index
        bayer = img.reshape((CAM_HEIGHT, CAM_WIDTH))
        color = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)
        frame_path = frames_dir / f'frame_{frame_index:06d}.png'
        cv2.imwrite(str(frame_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        frame_index += 1

    # Start getting camera images
    img_thread = ImageThread(cf.link.cpx, _update_image)
    img_thread.start()

    # Reset estimation before taking off
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

    # Listen for emergency kill switch
    emergency_stop_thread = threading.Thread(target=emergency_stop_callback, args=(cf,), daemon=True)
    emergency_stop_thread.start()

    print("Starting control sequence...")
    print("Press 'q' at any time to kill motors and land gracefully.")

    try:
        # Take-off
        for y in range(10):
            cf.commander.send_hover_setpoint(0, 0, 0, y / 25)
            time.sleep(0.1)
        for _ in range(20):
            cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
            time.sleep(0.1)

        # Circle from origin around (-5, 0) with radius 5m
        for t in range(100):
            x = -5 + 5 * np.cos(2 * np.pi * t / 100)
            y = 5 * np.sin(2 * np.pi * t / 100)
            cf.commander.send_position_setpoint(x, y, 0, 0.4)
            time.sleep(0.1)

        # Return to hover
        for _ in range(20):
            cf.commander.send_hover_setpoint(0, 0, 0, 0.4)
            time.sleep(0.1)

        # Land
        for y in range(10):
            cf.commander.send_hover_setpoint(0, 0, 0, (10 - y) / 25)
            time.sleep(0.1)

    except Exception as e:
        print(f"Error during flight: {e}")
    finally:
        cf.commander.send_stop_setpoint()
        time.sleep(0.5)
        cf.close_link()
        print("Flight completed and link closed.")
