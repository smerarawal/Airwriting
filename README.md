# ✦ Air Writing — Fingertip CV Project

Draw in the air with your index finger using MediaPipe hand tracking + OpenCV.

---

## Requirements

- Python 3.9 – 3.11  (MediaPipe does **not** support Python 3.12+ yet)
- A webcam

---

## Setup (one time)

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Run

```bash
python airwriting.py
```

---

## Gestures & Controls

| Action | How |
|---|---|
| **Draw** | Raise index finger up |
| **Lift pen** | Pinch index finger + thumb together |
| **Change colour** | Press `1` `2` `3` `4` `5` `6` |
| **Brush size** | Press `S` (small) `M` (medium) `L` (large) `X` (XL) |
| **Erase mode** | Press `E` |
| **Undo** | Press `U` |
| **Clear canvas** | Press `C` |
| **Save PNG** | Press `W` (saved to current folder) |
| **Toggle help** | Press `H` |
| **Quit** | Press `Q` |

---

## Tips for best results

- Good, even lighting on your hand
- Keep your hand 30–60 cm from the camera
- Move slowly and deliberately for clean strokes
- Pinch clearly to end a stroke before starting a new one

---

## Troubleshooting

**Webcam not opening** — Make sure no other app (Teams, Zoom, etc.) is using the camera.

**MediaPipe install fails** — Make sure you are on Python 3.9–3.11. Run `python --version` to check.

**Laggy tracking** — Lower `model_complexity` from `1` to `0` in `airwriting.py` line 13.
