import os
import cv2
from cv_detection2 import detect_gate, render_detection

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

img_path = os.path.join(
    BASE_DIR,
    "flight_frames/20260524_134149/frame_00062.jpg"
)

print("Loading:", img_path)

img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Cannot load image: {img_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

result = detect_gate(gray)

print("STATUS:", result["status"])

vis = render_detection(img, result)

cv2.imshow("result", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()