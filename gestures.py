"""
gestures.py — hand gesture detection with debouncing for the fingertip app.

Classifies the current hand pose into one of:
    "point"     index only         -> identify the pointed part
    "compare"   index + middle     -> two-finger compare (capture A, then B)
    "fist"      no fingers up       -> save annotated screenshot
    "thumbsup"  thumb only          -> confirm the identified object
    "other"     anything else       -> ignored
"""

from collections import deque
import math

TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCPS = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
THUMB_TIP, THUMB_IP, THUMB_MCP, WRIST = 4, 3, 2, 0


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def get_extended_fingers(lm):
    """
    Robust finger-extension detection.

    Four fingers: a finger is extended if its tip is farther from the wrist
    than its PIP joint is — this works regardless of hand rotation, unlike a
    raw tip.y < pip.y test.

    Thumb: extended if the tip is far from the index-finger MCP (thumb sticks
    out sideways). Uses a distance ratio so it scales with hand size.
    """
    wrist = lm[WRIST]
    fingers = {}
    for name in ("index", "middle", "ring", "pinky"):
        tip_d = _dist(lm[TIPS[name]], wrist)
        pip_d = _dist(lm[PIPS[name]], wrist)
        # extended if tip clearly beyond the pip (with a small margin)
        fingers[name] = tip_d > pip_d * 1.05

    # Hand scale = wrist to index MCP, used to normalise the thumb test
    scale = _dist(lm[MCPS["index"]], wrist) + 1e-6
    thumb_tip_to_index_mcp = _dist(lm[THUMB_TIP], lm[MCPS["index"]])
    # Thumb extended if its tip is far from the index knuckle (sticking out)
    fingers["thumb"] = (thumb_tip_to_index_mcp / scale) > 0.75
    return fingers


def classify_gesture(lm):
    """Map the extended-finger pattern to a named gesture."""
    f = get_extended_fingers(lm)
    idx, mid, ring, pinky, thumb = (
        f["index"], f["middle"], f["ring"], f["pinky"], f["thumb"]
    )
    n_fingers = sum([idx, mid, ring, pinky])

    # Fist: no fingers extended, thumb also curled
    if n_fingers == 0 and not thumb:
        return "fist"
    # Thumbs up: thumb out, all four fingers curled
    if thumb and n_fingers == 0:
        return "thumbsup"
    # Point: index only (thumb may be in or out — pointing often has thumb tucked)
    if idx and mid is False and ring is False and pinky is False:
        return "point"
    # Compare: index + middle up, ring & pinky down (peace sign)
    if idx and mid and not ring and not pinky:
        return "compare"
    return "other"


def debug_finger_states(lm):
    """Return a short string of finger states for on-screen debugging."""
    f = get_extended_fingers(lm)
    def mark(b): return "U" if b else "-"
    return (f"T:{mark(f['thumb'])} I:{mark(f['index'])} "
            f"M:{mark(f['middle'])} R:{mark(f['ring'])} P:{mark(f['pinky'])}"
            f"  => {classify_gesture(lm)}")


class GestureStabiliser:
    """Debounces raw per-frame gestures into stable + one-shot events."""

    ONE_SHOT = {"fist", "thumbsup", "compare"}

    def __init__(self, frames_needed=4):
        self.frames_needed = frames_needed
        self.history = deque(maxlen=frames_needed)
        self.stable = None
        self._fired_for_current_hold = False

    def update(self, raw_gesture):
        self.history.append(raw_gesture)
        just_triggered = None
        if len(self.history) == self.frames_needed and len(set(self.history)) == 1:
            new_stable = self.history[0]
            if new_stable != self.stable:
                self.stable = new_stable
                self._fired_for_current_hold = False
            if (self.stable in self.ONE_SHOT
                    and not self._fired_for_current_hold):
                just_triggered = self.stable
                self._fired_for_current_hold = True
        return just_triggered

    def reset(self):
        self.history.clear()
        self.stable = None
        self._fired_for_current_hold = False