import cv2
import numpy as np

# ── detection constants ────────────────────────────────────────────────────────
BRIGHTNESS_THRESHOLD = 180

K1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  20))
K2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,  5))

GATE_SIZE_MIN   = 30    # px — below this → noise, ignore
GATE_SIZE_CLOSE = 180   # px — above this → close enough → TRANSIT

# ── detection pipeline ─────────────────────────────────────────────────────────

def get_gate_mask(img, threshold=BRIGHTNESS_THRESHOLD, k1=K1, k2=K2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
    return mask

def _clean_filled(filled, close_size=20, open_size=20):
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size,  open_size))
    out = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kc)
    out = cv2.morphologyEx(out,    cv2.MORPH_OPEN,  ko)
    return out

def _smooth_filled(filled, blur_size=21, threshold=127):
    blurred = cv2.GaussianBlur(filled.astype(np.float32), (blur_size, blur_size), 0)
    _, out  = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    return out.astype(np.uint8)

def find_gate_rectangles(mask, min_area=1000, max_area=1e6):
    kh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  20))
    kv = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,  5))
    dilated = cv2.morphologyEx(mask,    cv2.MORPH_CLOSE, kh)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kv)

    filled     = dilated.copy()
    h, w       = filled.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, flood_mask, (0, 0), 255)
    filled = cv2.bitwise_not(filled)
    filled = cv2.bitwise_or(filled, dilated)
    filled = _clean_filled(filled)
    filled = _smooth_filled(filled)

    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue
        rect            = cv2.minAreaRect(cnt)
        center, (rw, rh), angle = rect
        if max(rw, rh) / (min(rw, rh) + 1e-6) > 4.0:
            continue
        rects.append({
            'center': np.array(center),
            'size':   (rw, rh),
            'angle':  angle,
            'box':    np.int32(cv2.boxPoints(rect)),
            'area':   area,
        })
    return rects

def get_gate_detection(frame):
    """
    Return (cx, cy, size) for the closest visible gate, or (None, None, 0).
    size = max side of the fitted rectangle in pixels (grows as you approach).
    """
    if frame is None:
        return None, None, 0

    rects = find_gate_rectangles(get_gate_mask(frame))
    if not rects:
        return None, None, 0

    best = max(rects, key=lambda r: max(r['size']))
    cx, cy = best['center']
    size   = max(best['size'])

    return (float(cx), float(cy), float(size)) if size >= GATE_SIZE_MIN \
           else (None, None, 0)
