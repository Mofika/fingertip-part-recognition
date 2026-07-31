"""
Fingertip Part Recognition — Research Prototype
================================================

Answers the research question: can a Vision AI, combined with real-time hand
tracking, identify the specific PART of an object the user's index finger is
pointing at (not just classify the whole object)?

ARCHITECTURE (three specialists, not one generalist)
----------------------------------------------------
  Stage 1  MediaPipe Hands  -> exact index-fingertip pixel   (local, 30fps)
  Stage 2  Dwell gate       -> decides WHEN to ask           (local)
  Stage 3  Gemini via Vertex-> names object + part           (cloud, GCP credit)

Two problems this code solves explicitly:
  * OCCLUSION - the finger covers the part it points at. Handled by (a) sampling
    slightly AHEAD of the fingertip along the finger's axis, and (b) keeping a
    "clean plate" frame from before the hand entered the scene.
  * TIMING - the vision call is slow relative to 30fps video, so it runs on a
    background thread and only fires when the finger dwells.

EXPERIMENT MODES (set with the 'm' key at runtime)
--------------------------------------------------
  MODE_A  Baseline: send the RAW frame, ask "what part is the finger touching?"
          (Vision AI alone - no hand tracking assistance)
  MODE_B  Pipeline: send a MARKED frame with a dot at the computed target point.
          (Hand tracking + Vision AI)

  The accuracy gap between A and B is the result of your study.

SETUP
-----
  Auth: gcloud auth application-default login   (no API key — uses Vertex)
  py -3.11 -m pip install opencv-python mediapipe numpy google-genai
  py -3.11 fingertip.py

CONTROLS
--------
  q       quit
  m       toggle MODE_A / MODE_B
  SPACE   force a capture now (ignore dwell gate)
  y / n   log the last prediction as correct / incorrect
  c       clear the cached answer
  [ / ]   decrease / increase the point-ahead offset
"""

import csv
import os
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from hand_tracker import HandTracker
from gestures import classify_gesture, GestureStabiliser, debug_finger_states
from vertex_vision import ask_vertex     # <-- vision now runs on Vertex/GCP credit

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

# --- Dwell gate --------------------------------------------------------------
DWELL_MS = 500          # finger must hold still this long before we ask
DWELL_RADIUS_PX = 25    # "still" = stays inside this radius
RECALL_RADIUS_PX = 60   # move further than this from the last query -> ask again

# --- Occlusion handling ------------------------------------------------------
POINT_AHEAD_PX = 45     # sample this far beyond the fingertip, along finger axis
                        # adjustable at runtime with [ and ]
CLEAN_PLATE_FRAMES = 90 # rolling buffer of hand-free frames (~3s at 30fps)
CLEAN_PLATE_MAX_AGE_S = 3.0   # older than this, the scene may have changed
CROP_HALF_SIZE = 220    # crop a square of this half-width around the target

# --- Debugging ---------------------------------------------------------------
DEBUG_SHOW_QUERY = True
DEBUG_SAVE_DIR = "debug_queries"

# --- Timing ------------------------------------------------------------------
# Vertex has real quota (paid by GCP credit), so this is only about pacing the
# UX, not dodging a free-tier limit. 2s feels responsive; lower if you like.
MIN_SECONDS_BETWEEN_CALLS = 2.0

# --- Output ------------------------------------------------------------------
LOG_PATH = "experiment_log.csv"

MODE_A, MODE_B = "A_vlm_only", "B_pipeline"

# MediaPipe hand landmark indices
INDEX_MCP = 5   # base knuckle of the index finger
INDEX_TIP = 8   # tip of the index finger


# ----------------------------------------------------------------------------
# STAGE 3: the Vision AI prompts
# ----------------------------------------------------------------------------

PROMPT_MODE_B = (
    "A red dot has been drawn on this image. "
    "First identify the object shown. Then identify the specific PART of that "
    "object located at the red dot. Then give ONE short factual sentence "
    "about that part (its function, material, or purpose). "
    "Answer in exactly this format and nothing else: OBJECT | PART | DETAIL\n"
    "Example: lion | eye | Lions have excellent night vision, roughly six "
    "times more sensitive to low light than a human eye.\n"
    "If you cannot determine the part, answer: OBJECT | unknown | unknown"
)

PROMPT_MODE_A = (
    "A person's index finger is pointing at or touching part of an object in "
    "this image. First identify the object. Then identify the specific PART of "
    "that object the finger is indicating. Then give ONE short factual "
    "sentence about that part (its function, material, or purpose). "
    "Answer in exactly this format and nothing else: OBJECT | PART | DETAIL\n"
    "Example: lion | eye | Lions have excellent night vision, roughly six "
    "times more sensitive to low light than a human eye.\n"
    "If you cannot determine the part, answer: OBJECT | unknown | unknown"
)

# --- LEARNING MODE prompt (toggle with 'L') ---------------------------------
# Asks for a full teaching explanation instead of a one-line detail. Still uses
# the OBJECT | PART | DETAIL shape so the same parser works — DETAIL just holds
# the longer explanation. The explanation panel wraps and shows all of it.
PROMPT_LEARNING = (
    "A red dot marks a specific part of an object in this image. "
    "You are a knowledgeable teacher. Identify the object and the exact part at "
    "the red dot, then explain that part clearly for a learner: what it does "
    "(its function), what it is typically made of (material), and how it works "
    "(the mechanism), in 2-4 sentences total. "
    "Answer in exactly this format and nothing else: OBJECT | PART | EXPLANATION\n"
    "Example: circuit board | capacitor | A capacitor stores electrical energy "
    "temporarily. It is usually made of two conductive plates separated by an "
    "insulating material. It charges and discharges to smooth and filter voltage, "
    "keeping the current steady in the circuit.\n"
    "If you cannot determine the part, still explain what is at the red dot as "
    "best you can, in the same format."
)

# --- COMPARE prompt (two-finger gesture) ------------------------------------
# Sent with TWO images side by side (A on top, B on bottom), each with a red dot.
PROMPT_COMPARE = (
    "Two images are stacked vertically. The TOP image has a red dot on part A. "
    "The BOTTOM image has a red dot on part B. "
    "Identify part A and part B, then explain the key DIFFERENCES between them "
    "(function, material, or purpose) in 2-4 sentences. "
    "Answer in exactly this format and nothing else: A | B | DIFFERENCES\n"
    "Example: capacitor | resistor | A capacitor stores energy while a resistor "
    "limits current. Capacitors use conductive plates and an insulator; resistors "
    "use a resistive material. One smooths voltage, the other controls current flow."
)


# ----------------------------------------------------------------------------
# STAGE 2: geometry + occlusion handling
# ----------------------------------------------------------------------------

def compute_target_point(landmarks, w, h, ahead_px=POINT_AHEAD_PX):
    """
    People point AT things, so the intended target sits just beyond the
    fingernail. Extrapolate along the index finger's axis (knuckle -> tip).

    ahead_px too large -> the target overshoots the object onto the background
    (you get "wall | surface"). Too small -> the finger occludes the part.
    Tune it live with the [ and ] keys.

    Returns (tip_xy, target_xy) in pixel coordinates.
    """
    mcp = landmarks[INDEX_MCP]
    tip = landmarks[INDEX_TIP]

    tip_xy = np.array([tip.x * w, tip.y * h])
    mcp_xy = np.array([mcp.x * w, mcp.y * h])

    direction = tip_xy - mcp_xy
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        target_xy = tip_xy
    else:
        target_xy = tip_xy + (direction / norm) * ahead_px

    target_xy[0] = np.clip(target_xy[0], 0, w - 1)
    target_xy[1] = np.clip(target_xy[1], 0, h - 1)

    return tip_xy.astype(int), target_xy.astype(int)


def build_query_image(clean_frame, live_frame, target_xy, mode):
    """
    MODE_B: use the clean plate (pre-occlusion view, if available) so the part
            under the finger is actually visible, draw a marker at the target,
            and crop around it.
    MODE_A: send the raw live frame with the finger in it, no marker.
    """
    if mode == MODE_A:
        return live_frame.copy()

    base = clean_frame.copy() if clean_frame is not None else live_frame.copy()
    x, y = int(target_xy[0]), int(target_xy[1])

    cv2.circle(base, (x, y), 11, (255, 255, 255), -1)
    cv2.circle(base, (x, y), 8, (0, 0, 255), -1)

    h, w = base.shape[:2]
    x0, x1 = max(0, x - CROP_HALF_SIZE), min(w, x + CROP_HALF_SIZE)
    y0, y1 = max(0, y - CROP_HALF_SIZE), min(h, y + CROP_HALF_SIZE)
    crop = base[y0:y1, x0:x1]

    return crop if crop.size else base


def make_marked_crop(frame, target_xy):
    """A marked+cropped single image for one side of a comparison."""
    return build_query_image(None, frame, target_xy, MODE_B)


def stack_two(img_a, img_b):
    """Stack two images vertically to the same width for the compare prompt."""
    wa, wb = img_a.shape[1], img_b.shape[1]
    w = min(wa, wb)
    ra = cv2.resize(img_a, (w, int(img_a.shape[0] * w / wa)))
    rb = cv2.resize(img_b, (w, int(img_b.shape[0] * w / wb)))
    return np.vstack([ra, rb])


def save_annotated_screenshot(display):
    """Fist gesture — save the current annotated view to disk."""
    os.makedirs("screenshots", exist_ok=True)
    path = os.path.join("screenshots",
                        f"shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    cv2.imwrite(path, display)
    print(f"screenshot saved: {path}")
    return path


# ----------------------------------------------------------------------------
# Experiment logging
# ----------------------------------------------------------------------------

class ExperimentLog:
    def __init__(self, path):
        self.path = path
        self.last_row = None
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow([
                "timestamp", "mode", "predicted_object", "predicted_part",
                "predicted_detail", "raw_response", "latency_ms",
                "target_x", "target_y", "used_clean_plate", "correct",
            ])
            self.f.flush()

    def record(self, mode, obj, part, detail, raw, latency_ms, target, used_clean):
        self.last_row = [
            datetime.now().isoformat(timespec="seconds"), mode, obj, part,
            detail, raw, round(latency_ms), int(target[0]), int(target[1]),
            int(used_clean), "",
        ]

    def commit(self, correct=""):
        if self.last_row is None:
            return
        self.last_row[-1] = correct
        self.w.writerow(self.last_row)
        self.f.flush()
        self.last_row = None

    def close(self):
        self.commit()
        self.f.close()


# ----------------------------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")

    hands = HandTracker(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    log = ExperimentLog(LOG_PATH)

    # State
    mode = MODE_B
    learning_mode = False                  # toggle with 'L' — detailed explanations
    gesture_stab = GestureStabiliser(frames_needed=5)
    compare_slot_a = None                  # holds first captured image for compare
    current_gesture = None                 # stable gesture name for display
    clean_plate = None
    clean_plate_time = 0.0
    ahead_px = POINT_AHEAD_PX
    last_query_img = None
    clean_buffer = deque(maxlen=CLEAN_PLATE_FRAMES)
    dwell_anchor = None
    dwell_start = None
    last_query_point = None
    last_call_time = 0.0
    answer = {"object": None, "part": None, "detail": None,
              "error": None, "pending": False}
    lock = threading.Lock()

    def worker(query_img, prompt, target, used_clean, mode_used):
        t0 = time.time()
        obj, part, detail, raw, err = ask_vertex(query_img, prompt)
        latency = (time.time() - t0) * 1000
        with lock:
            answer["object"], answer["part"] = obj, part
            answer["detail"] = detail
            answer["error"] = err
            answer["pending"] = False
        if err is None:
            log.record(mode_used, obj, part, detail, raw, latency,
                       target, used_clean)

    print("Running. Keys: q quit | m mode | L learning | SPACE force | y/n log | c clear | [ ] offset")
    print("Gestures: index=identify | two fingers=compare | fist=screenshot | thumbs up=confirm")

    cv2.namedWindow("Fingertip Part Recognition", cv2.WINDOW_NORMAL)

    # Video recording — saves the annotated display feed to an mp4
    record_path = f"output_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(record_path, fourcc, 24.0, (1280, 720))
    print(f"Recording to: {record_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)

        hand_present = bool(result.multi_hand_landmarks)
        display = frame.copy()
        target_xy = None

        if not hand_present:
            clean_buffer.append(frame.copy())
            clean_plate = clean_buffer[-1]
            clean_plate_time = time.time()
            dwell_anchor, dwell_start = None, None
            gesture_stab.reset()
            current_gesture = None
        else:
            lm = result.multi_hand_landmarks[0].landmark
            tip_xy, target_xy = compute_target_point(lm, w, h, ahead_px)

            # ---- Gesture classification + debounce ----------------------
            raw_gesture = classify_gesture(lm)
            triggered = gesture_stab.update(raw_gesture)
            current_gesture = gesture_stab.stable

            # Live finger-state debug (T I M R P => gesture)
            cv2.putText(display, debug_finger_states(lm), (14, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # FIST -> save annotated screenshot (one-shot)
            if triggered == "fist":
                save_annotated_screenshot(display)

            # THUMBS UP -> confirm the current answer (one-shot); logs as correct
            if triggered == "thumbsup":
                with lock:
                    has_answer = answer["object"] is not None
                if has_answer:
                    log.commit("1")
                    print("confirmed (thumbs up): logged correct")

            # COMPARE (two fingers) -> capture A, then B, then compare (one-shot)
            if triggered == "compare" and not answer["pending"]:
                plate_age = time.time() - clean_plate_time
                plate = clean_plate if (clean_plate is not None
                                        and plate_age <= CLEAN_PLATE_MAX_AGE_S) else None
                base_for_crop = plate if plate is not None else frame
                marked = make_marked_crop(base_for_crop, target_xy)

                if compare_slot_a is None:
                    compare_slot_a = marked
                    print("compare: captured A — now point at B and show two fingers")
                    with lock:
                        answer.update({"object": "compare", "part": "A captured",
                                       "detail": "Point at the second part and "
                                                 "show two fingers again.",
                                       "error": None, "pending": False})
                else:
                    stacked = stack_two(compare_slot_a, marked)
                    compare_slot_a = None
                    last_query_img = stacked.copy()
                    with lock:
                        answer["pending"] = True
                        answer["error"] = None
                    last_call_time = time.time()
                    threading.Thread(
                        target=worker,
                        args=(stacked, PROMPT_COMPARE, target_xy, False, "compare"),
                        daemon=True,
                    ).start()
                    print("compare: captured B — comparing...")

            cv2.circle(display, tuple(tip_xy), 8, (0, 255, 255), 2)
            cv2.line(display, tuple(tip_xy), tuple(target_xy), (0, 255, 255), 2)
            cv2.circle(display, tuple(target_xy), 10, (0, 0, 255), -1)

            now = time.time()
            if dwell_anchor is None or np.linalg.norm(target_xy - dwell_anchor) > DWELL_RADIUS_PX:
                dwell_anchor = target_xy.astype(float)
                dwell_start = now

            dwelled = (now - dwell_start) * 1000 >= DWELL_MS

            moved_enough = (
                last_query_point is None
                or np.linalg.norm(target_xy - last_query_point) > RECALL_RADIUS_PX
            )

            cooled_down = (now - last_call_time) >= MIN_SECONDS_BETWEEN_CALLS

            should_ask = (dwelled and moved_enough and cooled_down
                          and not answer["pending"]
                          and current_gesture == "point")

            if not dwelled and dwell_start:
                pct = min(1.0, (now - dwell_start) * 1000 / DWELL_MS)
                cv2.ellipse(display, tuple(target_xy), (22, 22), -90, 0,
                            int(360 * pct), (0, 255, 0), 3)

            if should_ask:
                plate_age = time.time() - clean_plate_time
                plate_fresh = (clean_plate is not None
                               and plate_age <= CLEAN_PLATE_MAX_AGE_S)
                plate = clean_plate if plate_fresh else None

                used_clean = plate is not None and mode == MODE_B
                if learning_mode:
                    prompt = PROMPT_LEARNING
                else:
                    prompt = PROMPT_MODE_B if mode == MODE_B else PROMPT_MODE_A
                query_img = build_query_image(plate, frame, target_xy, mode)
                last_query_img = query_img.copy()

                if DEBUG_SAVE_DIR:
                    os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)
                    stamp = datetime.now().strftime("%H%M%S")
                    cv2.imwrite(os.path.join(DEBUG_SAVE_DIR,
                                             f"query_{stamp}.jpg"), query_img)

                with lock:
                    answer["pending"] = True
                    answer["error"] = None
                last_call_time = now
                last_query_point = target_xy.copy()

                threading.Thread(
                    target=worker,
                    args=(query_img, prompt, target_xy, used_clean, mode),
                    daemon=True,
                ).start()

        # ---- Overlay -------------------------------------------------------
        with lock:
            pending, obj, part, detail, err = (
                answer["pending"], answer["object"], answer["part"],
                answer["detail"], answer["error"],
            )

        if pending:
            msg, col = "asking vision AI...", (0, 200, 255)
        elif err:
            msg, col = f"error: {err[:60]}", (0, 80, 255)
        elif obj and part:
            msg, col = f"This is the {obj}'s {part}", (0, 255, 120)
        else:
            msg, col = "point at something and hold still", (150, 150, 150)

        # In learning mode the explanation is long, so allow more lines.
        wrap_chars = 74 if learning_mode else 62
        max_lines = 8 if learning_mode else 3
        detail_lines = []
        if detail and not pending and not err:
            words, line = detail.split(), ""
            for word in words:
                trial = f"{line} {word}".strip()
                if len(trial) > wrap_chars:
                    detail_lines.append(line)
                    line = word
                else:
                    line = trial
            if line:
                detail_lines.append(line)
            detail_lines = detail_lines[:max_lines]

        panel_h = 78 + (26 * len(detail_lines) if detail_lines else 0)
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.78, display, 0.22, 0, display)

        banner = f"MODE {mode}" + ("  |  LEARNING" if learning_mode else "")
        banner_col = (120, 220, 255) if learning_mode else (180, 180, 180)
        cv2.putText(display, banner, (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_col, 2)

        # Gesture indicator
        gesture_label = {
            "point": "POINT - identify", "compare": "TWO FINGERS - compare",
            "fist": "FIST - screenshot", "thumbsup": "THUMBS UP - confirm",
        }.get(current_gesture, "")
        if gesture_label:
            cv2.putText(display, gesture_label, (w - 340, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
        if compare_slot_a is not None:
            cv2.putText(display, "compare: A captured - point at B",
                        (14, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 255), 2)
        cv2.putText(display, msg, (14, 62), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, col, 2)

        for i, line in enumerate(detail_lines):
            cv2.putText(display, line, (16, 90 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1)

        plate_age = time.time() - clean_plate_time
        if clean_plate is not None and plate_age <= CLEAN_PLATE_MAX_AGE_S:
            cv2.putText(display, f"plate: fresh {plate_age:.1f}s", (w - 320, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1)
        else:
            cv2.putText(display, "plate: STALE - move hand out of frame",
                        (w - 430, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 140, 255), 1)

        cv2.putText(display, f"ahead: {ahead_px}px  ([ / ])", (w - 320, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Record the annotated feed. Resize to the writer's fixed size so a
        # camera that doesn't deliver exactly 1280x720 still records cleanly.
        rec = display if (display.shape[1], display.shape[0]) == (1280, 720) \
              else cv2.resize(display, (1280, 720))
        video_writer.write(rec)

        cv2.imshow("Fingertip Part Recognition", display)

        if DEBUG_SHOW_QUERY and last_query_img is not None:
            cv2.imshow("DEBUG: image sent to API", last_query_img)

        # ---- Keys ----------------------------------------------------------
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("m"):
            mode = MODE_A if mode == MODE_B else MODE_B
            last_query_point = None
            print(f"switched to {mode}")
        elif k == ord("l"):
            learning_mode = not learning_mode
            last_query_point = None
            with lock:
                answer.update({"object": None, "part": None,
                               "detail": None, "error": None})
            print(f"learning mode: {'ON' if learning_mode else 'OFF'}")
        elif k == ord("c"):
            with lock:
                answer.update({"object": None, "part": None,
                               "detail": None, "error": None})
            last_query_point = None
        elif k == ord("["):
            ahead_px = max(0, ahead_px - 10)
            print(f"point-ahead offset: {ahead_px}px")
        elif k == ord("]"):
            ahead_px = min(200, ahead_px + 10)
            print(f"point-ahead offset: {ahead_px}px")
        elif k == ord("y"):
            log.commit("1")
            print("logged: correct")
        elif k == ord("n"):
            log.commit("0")
            print("logged: incorrect")
        elif k == ord(" ") and target_xy is not None and not answer["pending"]:
            used_clean = clean_plate is not None and mode == MODE_B
            if learning_mode:
                prompt = PROMPT_LEARNING
            else:
                prompt = PROMPT_MODE_B if mode == MODE_B else PROMPT_MODE_A
            query_img = build_query_image(clean_plate, frame, target_xy, mode)
            with lock:
                answer["pending"] = True
            last_call_time = time.time()
            last_query_point = target_xy.copy()
            threading.Thread(target=worker,
                             args=(query_img, prompt, target_xy, used_clean, mode),
                             daemon=True).start()

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    hands.close()
    log.close()
    print(f"\nResults written to {LOG_PATH}")


if __name__ == "__main__":
    main()