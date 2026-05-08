import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
from collections import deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions

# ── Download model if missing ─────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task (~3 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

# ── MediaPipe 0.10.x HandLandmarker ──────────────────────────────────────────
base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options   = HandLandmarkerOptions(
    base_options=base_opts,
    num_hands=1,
    min_hand_detection_confidence=0.75,
    min_hand_presence_confidence=0.65,
    min_tracking_confidence=0.65,
    running_mode=mp_vision.RunningMode.VIDEO,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ── Config ────────────────────────────────────────────────────────────────────
PINCH_THRESH = 0.06
TRAIL_LEN    = 24
BRUSH_SIZES  = [4, 8, 14, 22]
COLORS = {
    "cyan":   (255, 210,  0),
    "pink":   (160,  60, 230),
    "green":  ( 50, 220, 100),
    "yellow": (  0, 220, 240),
    "white":  (255, 255, 255),
    "red":    ( 40,  40, 230),
}
COLOR_LIST  = list(COLORS.values())
COLOR_NAMES = list(COLORS.keys())

# ── State ─────────────────────────────────────────────────────────────────────
canvas     = None
brush_idx  = 1
color_idx  = 0
erase_mode = False
last_pt    = None
smooth_pt  = None
trail      = deque(maxlen=TRAIL_LEN)
undo_stack = []
show_help  = True
fps_buf    = deque(maxlen=30)

# ── Helpers ───────────────────────────────────────────────────────────────────
def lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def dist_norm(a, b):
    return np.hypot(a.x - b.x, a.y - b.y)

def push_undo():
    undo_stack.append(canvas.copy())
    if len(undo_stack) > 30:
        undo_stack.pop(0)

def draw_skeleton(frame, landmarks, w, h):
    for a, b in HAND_CONNECTIONS:
        pa = lm_px(landmarks[a], w, h)
        pb = lm_px(landmarks[b], w, h)
        cv2.line(frame, pa, pb, (30, 60, 100), 1)
    for lm in landmarks:
        cv2.circle(frame, lm_px(lm, w, h), 2, (40, 80, 120), -1)

def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=0.55):
    overlay = img.copy()
    x1, y1 = pt1; x2, y2 = pt2
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, thickness)
    for cx, cy in [(x1+radius,y1+radius),(x2-radius,y1+radius),
                   (x1+radius,y2-radius),(x2-radius,y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def draw_ui(frame, fps, drawing, pinching):
    h, w = frame.shape[:2]
    cur_color = COLOR_LIST[color_idx]

    draw_rounded_rect(frame, (0,0), (w,56), (15,15,25), radius=0, alpha=0.75)
    cv2.putText(frame, f"FPS {fps:02d}", (16,36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,120,160), 1, cv2.LINE_AA)

    sx = 110
    for i, (col, name) in enumerate(zip(COLOR_LIST, COLOR_NAMES)):
        cx = sx + i*46
        cv2.circle(frame, (cx, 28), 14, col, -1)
        cv2.circle(frame, (cx, 28), 14,
                   (200,200,200) if i==color_idx else (60,60,80),
                   2 if i==color_idx else 1)
        if i == color_idx:
            cv2.putText(frame, name, (cx-20, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,220), 1, cv2.LINE_AA)

    bx = sx + len(COLOR_LIST)*46 + 24
    for i, (sz, lbl) in enumerate(zip(BRUSH_SIZES, ["S","M","L","XL"])):
        bx2 = bx + i*44
        cv2.circle(frame, (bx2, 28), sz//2+2,
                   cur_color if i==brush_idx else (80,80,100), -1)
        cv2.putText(frame, lbl, (bx2-6, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200,200,220) if i==brush_idx else (80,80,100), 1, cv2.LINE_AA)

    tag = "ERASE" if erase_mode else "DRAW"
    tag_col = (60,60,200) if erase_mode else cur_color
    tx = w - 130
    draw_rounded_rect(frame,(tx-4,8),(tx+110,48), tag_col, radius=8, alpha=0.35)
    cv2.putText(frame, tag, (tx+4, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, tag_col, 2, cv2.LINE_AA)

    draw_rounded_rect(frame, (0,h-42), (w,h), (15,15,25), radius=0, alpha=0.70)
    pen_status = "DRAWING" if drawing else ("PEN UP (pinch)" if pinching else "LIFT INDEX FINGER TO DRAW")
    cv2.putText(frame, pen_status, (16, h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0,220,80) if drawing else (120,120,180), 1, cv2.LINE_AA)
    cv2.putText(frame, "H=help  C=clear  U=undo  E=erase  W=save  Q=quit",
                (w//2-220, h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70,90,110), 1, cv2.LINE_AA)

    if show_help:
        hx, hy = 16, 70
        lines = [
            "  AIR WRITING - CONTROLS  ",
            "",
            "  Gesture:",
            "  Index finger up  = draw",
            "  Pinch to camera  = lift pen",
            "",
            "  Keys:",
            "  1-6   select colour",
            "  S/M/L/X  brush size",
            "  E     toggle erase",
            "  U     undo",
            "  C     clear canvas",
            "  W     save PNG",
            "  H     toggle help",
            "  Q     quit",
        ]
        draw_rounded_rect(frame,(hx-8,hy-8),(hx+290,hy+len(lines)*22+8),(20,20,35),radius=12,alpha=0.82)
        for j, line in enumerate(lines):
            col = (0,200,255) if j==0 else (180,200,220)
            cv2.putText(frame, line, (hx+4, hy+j*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

def draw_trail(frame):
    for i, pt in enumerate(trail):
        if pt is None:
            continue
        alpha = (i+1)/TRAIL_LEN
        r = max(1, int(BRUSH_SIZES[brush_idx]*0.4*alpha))
        col = tuple(int(c*alpha) for c in COLOR_LIST[color_idx])
        cv2.circle(frame, pt, r, col, -1)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global canvas, brush_idx, color_idx, erase_mode, last_pt, smooth_pt, show_help

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    canvas = np.zeros((actual_h, actual_w, 3), dtype=np.uint8)

    cv2.namedWindow("Air Writing", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Air Writing", actual_w, actual_h)

    t_prev   = time.time()
    frame_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_ms += 33
        result = detector.detect_for_video(mp_img, frame_ms)

        drawing  = False
        pinching = False
        tip_pt   = None

        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            draw_skeleton(frame, lms, w, h)

            idx_tip   = lms[8]
            idx_mid   = lms[6]
            thumb_tip = lms[4]

            tip_raw   = lm_px(idx_tip, w, h)

            # EMA smoothing - reduces jitter
            if smooth_pt is None:
                smooth_pt = tip_raw
            else:
                ALPHA = 0.4  # lower = smoother but more lag
                smooth_pt = (
                    int(ALPHA * tip_raw[0] + (1-ALPHA) * smooth_pt[0]),
                    int(ALPHA * tip_raw[1] + (1-ALPHA) * smooth_pt[1]),
                )

            tip_px    = smooth_pt
            tip_pt    = tip_raw  # cursor shows real position

            pinch_d   = dist_norm(idx_tip, thumb_tip)
            idx_base  = lms[5]
            finger_up = idx_tip.y < idx_base.y - 0.05

            pinching = pinch_d < PINCH_THRESH
            drawing  = finger_up and not pinching
            trail.append(tip_raw)

            cur_color = COLOR_LIST[color_idx]
            cur_size  = BRUSH_SIZES[brush_idx]

            if drawing:
                if last_pt is None:
                    last_pt = tip_px
                else:
                    if erase_mode:
                        cv2.line(canvas, last_pt, tip_px, (0,0,0), cur_size*4)
                    else:
                        cv2.line(canvas, last_pt, tip_px, cur_color, cur_size)
                        cv2.circle(canvas, tip_px, cur_size//2, cur_color, -1)
                    last_pt = tip_px
            else:
                last_pt = None
                smooth_pt = None
                trail.append(None)
        else:
            last_pt = None
            trail.append(None)

        t_now = time.time()
        fps_buf.append(1.0/(t_now - t_prev + 1e-9))
        t_prev = t_now
        fps = int(np.mean(fps_buf))

        display = cv2.addWeighted(frame, 0.45, canvas, 1.0, 0)

        if tip_pt:
            col = (0,80,255) if pinching else COLOR_LIST[color_idx]
            sz  = BRUSH_SIZES[brush_idx]
            cv2.circle(display, tip_pt, sz+6,  tuple(c//3 for c in col), -1)
            cv2.circle(display, tip_pt, sz+2,  col, -1)
            cv2.circle(display, tip_pt, sz+10, col, 1)

        draw_trail(display)
        draw_ui(display, fps, drawing, pinching)
        cv2.imshow("Air Writing", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            push_undo(); canvas[:] = 0
        elif key == ord('u'):
            if undo_stack: canvas[:] = undo_stack.pop()
        elif key == ord('e'):
            erase_mode = not erase_mode
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('w'):
            fname = f"airwriting_{int(time.time())}.png"
            cv2.imwrite(fname, canvas)
            print(f"Saved -> {fname}")
        elif ord('1') <= key <= ord('6'):
            color_idx = key - ord('1')
        elif key == ord('s'): brush_idx = 0
        elif key == ord('m'): brush_idx = 1
        elif key == ord('l'): brush_idx = 2
        elif key == ord('x'): brush_idx = 3

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()
