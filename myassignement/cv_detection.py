import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ── camera intrinsics ──────────────────────────────────────────────────────────

K = np.array([
    [191.42732876, 0.,           164.312971  ],
    [0.,           188.25579883, 136.49871826],
    [0.,           0.,           1.          ],
])
DIST = np.array([-5.56509139e-01, 1.36094480e+00, 3.25533387e-02,
                 -9.90039274e-04, -1.38102499e+00])

# ── preprocessing ──────────────────────────────────────────────────────────────

def preprocess_fill(gray, threshold=200,
                    close_short=5, close_long=20,
                    dilate_extra=3, open_size=3):
    """
    threshold → 2× asymmetric morph-close → seal dilation →
    flood-fill background → invert → OR with closed rim → open.

    Returns a dict with keys: raw, closed, seal, interior, filled.
    """
    H, W = gray.shape
    _, raw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    k_vert  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_short, close_long))
    k_horiz = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_long, close_short))
    closed = cv2.morphologyEx(raw,    cv2.MORPH_CLOSE, k_vert)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, k_horiz)

    seal = closed
    if dilate_extra >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_extra, dilate_extra))
        seal = cv2.dilate(closed, k)

    fill_src = seal.copy()
    mask_buf = np.zeros((H + 2, W + 2), np.uint8)
    for seed in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)]:
        if fill_src[seed[1], seed[0]] == 0:
            cv2.floodFill(fill_src, mask_buf, seed, 255)

    interior = cv2.bitwise_not(fill_src)
    full = cv2.bitwise_or(interior, closed)

    if open_size >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        full = cv2.morphologyEx(full, cv2.MORPH_OPEN, k)

    return {'raw': raw, 'closed': closed,
            'seal': seal, 'interior': interior, 'filled': full}

# ── corner detection ───────────────────────────────────────────────────────────

def order_corners(c):
    """Return corners ordered [TL, TR, BR, BL]."""
    c = np.asarray(c, dtype=np.float32)
    idx = np.argsort(c[:, 1])
    top, bot = c[idx[:2]], c[idx[2:]]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bot[np.argsort(bot[:, 0])]
    return np.array([tl, tr, br, bl])


def find_corners_by_curvature(curve, n_corners=4,
                               smooth_sigma=4.0, window_frac=0.05, nms_frac=0.12):
    """
    Detect n_corners corners on a closed contour using curvature peaks.

    Returns (corners, smoothed_curve, curvature_array).
    corners is None if fewer than n_corners peaks are found.
    """
    pts = np.asarray(curve, dtype=np.float64).reshape(-1, 2)
    N = len(pts)
    if N < 8:
        return None, None, None

    x = gaussian_filter1d(pts[:, 0], smooth_sigma, mode='wrap')
    y = gaussian_filter1d(pts[:, 1], smooth_sigma, mode='wrap')
    smoothed = np.column_stack([x, y])

    w = max(2, int(window_frac * N))
    v1 = smoothed - np.roll(smoothed,  w, axis=0)
    v2 = np.roll(smoothed, -w, axis=0) - smoothed
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot   = (v1 * v2).sum(axis=1)
    curv  = np.abs(np.arctan2(cross, dot))

    nms_dist = max(3, int(nms_frac * N))
    selected, avail = [], np.ones(N, dtype=bool)
    while len(selected) < n_corners and avail.any():
        masked = np.where(avail, curv, -1.0)
        i = int(np.argmax(masked))
        if masked[i] <= 0:
            break
        selected.append(i)
        for j in range(-nms_dist, nms_dist + 1):
            avail[(i + j) % N] = False

    if len(selected) < n_corners:
        return None, smoothed, curv
    return smoothed[selected], smoothed, curv

# ── full detector ──────────────────────────────────────────────────────────────

def detect_gate(gray, k=K, dist_coeffs=DIST,
                threshold=200, close_short=5, close_long=20,
                dilate_extra=3, open_size=3,
                border_pad=5, min_area=300, close_coverage=0.60,
                smooth_sigma=4.0, window_frac=0.05, nms_frac=0.12):
    """
    Detect a gate in a grayscale frame and return a result dict.

    Result keys:
      status          — 'ok' | 'no_gate' | 'no_corners' | 'commit_to_pass'
      quad_pix        — (4,2) float32 corners in pixel coords [TL,TR,BR,BL], or None
      quad_norm       — (4,2) float32 undistorted normalised coords, or None
      prep            — dict from preprocess_fill
      selected        — chosen contour, or None
      rejected_small  — list of contours rejected by area
      rejected_border — list of contours rejected by border touch
      accepted        — list of contours that passed both filters
      commit_to_pass  — contour used for commit_to_pass, or None
      smoothed        — smoothed contour points used for corner detection
      curvature       — curvature array
    """
    H, W = gray.shape
    img_area = float(W * H)

    prep = preprocess_fill(gray, threshold=threshold,
                            close_short=close_short, close_long=close_long,
                            dilate_extra=dilate_extra, open_size=open_size)
    mask = prep['filled']

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    accepted, small_c, border_c = [], [], []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            small_c.append(c)
            continue
        x, y, w, h = cv2.boundingRect(c)
        on_border = (x < border_pad or y < border_pad
                     or x + w > W - border_pad or y + h > H - border_pad)
        if on_border:
            border_c.append(c)
        else:
            accepted.append((c, area))

    result = {
        'prep': prep,
        'rejected_small': small_c,
        'rejected_border': border_c,
        'accepted': [c for c, _ in accepted],
        'selected': None, 'commit_to_pass': None,
        'smoothed': None, 'curvature': None,
        'quad_pix': None, 'quad_norm': None,
        'status': 'no_gate',
    }

    if not accepted:
        for c in border_c:
            if cv2.contourArea(c) / img_area > close_coverage:
                result['commit_to_pass'] = c
                result['status'] = 'commit_to_pass'
                break
        return result

    contour, _ = max(accepted, key=lambda x: x[1])
    result['selected'] = contour

    corners, smoothed, curv = find_corners_by_curvature(
        contour, 4, smooth_sigma=smooth_sigma,
        window_frac=window_frac, nms_frac=nms_frac)
    result['smoothed']  = smoothed
    result['curvature'] = curv

    if corners is None:
        result['status'] = 'no_corners'
        return result

    corners = order_corners(corners)
    result['quad_pix']  = corners
    result['quad_norm'] = cv2.undistortPoints(
        corners.reshape(-1, 1, 2).astype(np.float32), k, dist_coeffs
    ).reshape(-1, 2)
    result['status'] = 'ok'
    return result

# ── visualisation ──────────────────────────────────────────────────────────────

def render_detection(gray, result):
    """
    Draw detection overlays on gray and return an RGB image.

    gray    — raw grayscale frame
    magenta — commit-to-pass contour
    red     — border-rejected contours
    green   — accepted (not selected) contours
    bright green — selected gate contour
    cyan    — smoothed contour used for corner search
    yellow  — detected corner dots + quad polygon
    """
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if result['rejected_small']:
        cv2.drawContours(vis, result['rejected_small'], -1, (128, 128, 128), 1)
    if result['rejected_border']:
        cv2.drawContours(vis, result['rejected_border'], -1, (0, 0, 255), 2)    # → red

    sel = result['selected']
    others = [c for c in result['accepted'] if sel is None or c is not sel]
    if others:
        cv2.drawContours(vis, others, -1, (0, 120, 0), 1)                       # → dark green

    if result['commit_to_pass'] is not None:
        cv2.drawContours(vis, [result['commit_to_pass']], -1, (255, 0, 255), 2) # → magenta
    if sel is not None:
        cv2.drawContours(vis, [sel], -1, (0, 220, 0), 2)                        # → bright green

    if result['smoothed'] is not None:
        pts = np.int32(result['smoothed']).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (255, 255, 0), 1)                       # BGR→RGB cyan

    if result['quad_pix'] is not None:
        for (x, y) in result['quad_pix']:
            cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 255), -1)             # BGR→RGB yellow
        pts = np.int32(result['quad_pix']).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 255, 255), 1)                       # BGR→RGB yellow

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
