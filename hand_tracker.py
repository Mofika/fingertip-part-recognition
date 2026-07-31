"""
hand_tracker.py — drop-in replacement for the legacy `mp.solutions.hands`.

WHY THIS EXISTS
---------------
MediaPipe's old "Solutions" API (mp.solutions.hands.Hands) has been removed in
recent releases in favour of the "Tasks" API. This module wraps the Tasks API
and re-exposes the same interface the old code used, so the main script needs
only a two-line change:

    OLD:  import mediapipe as mp
          hands = mp.solutions.hands.Hands(max_num_hands=1, ...)
          result = hands.process(rgb)
          if result.multi_hand_landmarks:
              lm = result.multi_hand_landmarks[0].landmark

    NEW:  from hand_tracker import HandTracker
          hands = HandTracker(max_num_hands=1)
          result = hands.process(rgb)          # <- identical from here down
          if result.multi_hand_landmarks:
              lm = result.multi_hand_landmarks[0].landmark

Landmark indices are unchanged: 5 = index knuckle, 8 = index fingertip.
Each landmark still has .x, .y, .z as normalised floats (0.0 - 1.0).

MODEL FILE
----------
The Tasks API needs a model file, `hand_landmarker.task` (~7 MB). This module
downloads it automatically on first run and caches it next to this file.
If your machine blocks the download, fetch it manually from:
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
and place it in the same folder as this script.
"""

import os
import time
import urllib.request
import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model(path=None):
    """Download the hand landmarker model once and cache it locally."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            MODEL_FILENAME)
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path

    print(f"[hand_tracker] downloading model to {path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as e:
        raise RuntimeError(
            f"Could not download {MODEL_FILENAME} ({e}).\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it to:\n  {path}"
        ) from e
    print("[hand_tracker] model ready.")
    return path


# --- Shim objects that mimic the old result structure ------------------------

class _LandmarkList:
    """Mimics the old `multi_hand_landmarks[i]`, which had a `.landmark` list."""
    __slots__ = ("landmark",)

    def __init__(self, landmarks):
        self.landmark = landmarks


class _Result:
    """Mimics the old `hands.process()` return value."""
    __slots__ = ("multi_hand_landmarks", "multi_handedness")

    def __init__(self, hand_landmarks, handedness):
        self.multi_hand_landmarks = hand_landmarks or None
        self.multi_handedness = handedness or None


# --- The tracker -------------------------------------------------------------

class HandTracker:
    """
    Same constructor arguments as the old Hands() class, so existing calls
    keep working. `model_complexity` is accepted but ignored — the Tasks API
    selects this via the model file instead.
    """

    def __init__(self,
                 max_num_hands=1,
                 min_detection_confidence=0.6,
                 min_tracking_confidence=0.5,
                 min_presence_confidence=0.5,
                 model_complexity=None,          # accepted, unused
                 model_path=None):

        model_path = _ensure_model(model_path)

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        # VIDEO mode requires strictly increasing timestamps in milliseconds.
        self._t0 = time.time()
        self._last_ts = -1

    def process(self, rgb_frame):
        """
        Takes an RGB numpy frame (same as the old API) and returns an object
        with `.multi_hand_landmarks`, matching the old result shape.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        ts = int((time.time() - self._t0) * 1000)
        if ts <= self._last_ts:          # timestamps must strictly increase
            ts = self._last_ts + 1
        self._last_ts = ts

        out = self._landmarker.detect_for_video(mp_image, ts)

        hands = [_LandmarkList(h) for h in out.hand_landmarks] if out.hand_landmarks else []
        handed = out.handedness if getattr(out, "handedness", None) else []
        return _Result(hands, handed)

    def close(self):
        self._landmarker.close()

    # Allow use as a context manager
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    # Quick self-test: open the webcam and print the fingertip position.
    import cv2

    tracker = HandTracker(max_num_hands=1)
    cap = cv2.VideoCapture(0)
    print("Self-test running — show your hand. Press q to quit.")
    cv2.namedWindow("hand_tracker self-test", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("hand_tracker self-test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = tracker.process(rgb)

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            h, w = frame.shape[:2]
            x, y = int(lm[8].x * w), int(lm[8].y * h)
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"index tip: {x},{y}", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "no hand detected", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

        cv2.imshow("hand_tracker self-test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()