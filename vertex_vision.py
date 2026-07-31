"""
vertex_vision.py — Gemini vision via VERTEX AI / Agent Platform (GCP credits).

Authenticates through Application Default Credentials (the gcloud login you
already did) — NO API key. Billed to project raakesh-test-13052026, paid by
your GCP credit.

Drop-in replacement for ask_gemini: ask_vertex() has the same signature and
returns (object, part, detail, raw_text, error).

REQUIREMENTS (all done already, listed for reference)
-----------------------------------------------------
1. Agent Platform / Vertex AI API enabled on the project.
2. gcloud auth application-default login  (credentials saved).
3. py -3.11 -m pip install google-genai

TEST (no camera):
   py -3.11 vertex_vision.py
"""

import cv2

from google import genai
from google.genai import types

# ---- GCP SETTINGS ----------------------------------------------------------
PROJECT_ID = "raakesh-test-13052026"
LOCATION = "us-central1"

# Vertex model name. If this 404s, open Model Garden (Agent Platform -> Models)
# and use the exact ID shown there. Alternatives to try, in order:
#   "gemini-2.5-flash-lite"
#   "gemini-2.5-flash"
#   "gemini-flash-latest"
MODEL = "gemini-2.5-flash-lite"
# ----------------------------------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is None:
        # vertexai=True routes through Vertex/Agent Platform (GCP credit),
        # NOT the consumer API. No api_key argument — uses ADC from gcloud login.
        _client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
        )
    return _client


def encode_jpeg_bytes(frame_bgr, quality=85):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def ask_vertex(frame_bgr, prompt, model=MODEL):
    """
    Same signature/return as ask_gemini:
        returns (object, part, detail, raw_text, error)
    Authenticates via ADC (gcloud login) — no key in code.
    """
    try:
        img_bytes = encode_jpeg_bytes(frame_bgr)
    except Exception as e:
        return None, None, None, None, f"encode error: {e}"

    try:
        client = _get_client()
        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                # 400 fits a short paragraph of explanation (learning mode).
                # Experiment mode uses far fewer; the cap just sets the ceiling.
                max_output_tokens=400,
                temperature=0.2,
                # Disable "thinking" — part naming is perception, not reasoning.
                # Keeps answers un-truncated and output-token cost down.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (resp.text or "").strip()
    except Exception as e:
        msg = str(e)
        if "PERMISSION_DENIED" in msg or "403" in msg:
            return None, None, None, None, ("403 — your account may need the "
                                            "'Vertex AI User' role on the project. " + msg[:120])
        if "not found" in msg.lower() or "404" in msg:
            return None, None, None, None, (f"model '{model}' not found in {LOCATION} — "
                                            "check the exact name in Model Garden. " + msg[:120])
        if "default credentials" in msg.lower() or "could not automatically" in msg.lower():
            return None, None, None, None, ("No credentials — run: "
                                            "gcloud auth application-default login")
        if "billing" in msg.lower() or "RESOURCE_PROJECT" in msg:
            return None, None, None, None, ("Billing/quota issue on the project — "
                                            "confirm billing is linked. " + msg[:120])
        return None, None, None, None, f"{type(e).__name__}: {msg[:160]}"

    # Parse "OBJECT | PART | DETAIL"
    fields = [s.strip() for s in text.split("|")]
    if len(fields) >= 3:
        obj, part, detail = fields[0].lower(), fields[1].lower(), fields[2]
    elif len(fields) == 2:
        obj, part, detail = fields[0].lower(), fields[1].lower(), ""
    else:
        obj, part, detail = text.lower(), "unparsed", ""
    return obj, part, detail, text, None


# ---------------------------------------------------------------------------
# Standalone test — no camera. Confirms Vertex auth + model + vision.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    print(f"Project: {PROJECT_ID}")
    print(f"Region:  {LOCATION}")
    print(f"Model:   {MODEL}\n")

    # Test image: green circle on white, red dot marker
    img = np.full((240, 240, 3), 255, dtype=np.uint8)
    cv2.circle(img, (120, 120), 70, (0, 180, 0), -1)
    cv2.circle(img, (120, 120), 8, (0, 0, 255), -1)

    test_prompt = ("A red dot is drawn on this image. Identify the object, the "
                   "part at the red dot, and one short fact. "
                   "Format: OBJECT | PART | DETAIL")

    print("Sending test image via Vertex AI...")
    obj, part, detail, raw, err = ask_vertex(img, test_prompt)

    if err:
        print("\nERROR:", err)
        print("\nIf model not found -> try another MODEL name (see top of file).")
        print("If 403 -> grant your account the 'Vertex AI User' role in IAM.")
        print("If credentials -> gcloud auth application-default login")
    else:
        print("\n*** SUCCESS — running on Vertex AI / GCP credit ***")
        print("object:", obj)
        print("part:  ", part)
        print("detail:", detail)
        print("raw:   ", raw)