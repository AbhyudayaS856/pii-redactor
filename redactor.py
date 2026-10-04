"""
redactor.py - reads text from an image with OCR, finds PII, blacks it out,
and hides faces using a deep-learning face detector (YuNet).

Usage:  python redactor.py sample.png
Output: sample_redacted.png
"""
import os
import sys

import cv2
import easyocr

from detectors import find_pii, mask

FACE_MODEL = os.path.join("models", "face_detection_yunet_2023mar.onnx")

_reader = None
_face_detector = None


def get_reader():
    """Load the OCR model once (it's slow to load, fast to reuse)."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def get_face_detector():
    """Load the YuNet face detection model once."""
    global _face_detector
    if _face_detector is None:
        if not os.path.exists(FACE_MODEL):
            raise FileNotFoundError(
                f"Face model not found at {FACE_MODEL}. "
                "Download it into the 'models' folder first."
            )
        # 0.7 = only accept detections the model is at least 70% sure about
        _face_detector = cv2.FaceDetectorYN.create(FACE_MODEL, "", (320, 320), 0.7)
    return _face_detector


def hide_faces(image, out, findings):
    """
    Find faces and PIXELATE them. Pixelation is safer than a light blur,
    because blurred faces can sometimes be partly recovered.
    """
    h, w = image.shape[:2]
    detector = get_face_detector()
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    if faces is None:
        return

    for f in faces:
        x, y, fw, fh = [int(v) for v in f[:4]]
        pad = int(0.25 * fh)  # cover hair and chin too, not just the face
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + fw + pad, w), min(y + fh + pad, h)
        region = out[y1:y2, x1:x2]
        if region.size == 0:
            continue
        tiny = cv2.resize(region, (8, 8), interpolation=cv2.INTER_LINEAR)
        out[y1:y2, x1:x2] = cv2.resize(tiny, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        findings.append({
            "type": "FACE",
            "value (masked)": "[pixelated]",
            "confidence": round(float(f[14]), 2),
        })


def span_to_box(bbox, text, start, end):
    """
    OCR gives us a box for a whole line of text. If only characters
    start..end are sensitive, estimate where they sit horizontally,
    so we black out just that part.
    """
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    n = max(len(text), 1)
    sx = x1 + (x2 - x1) * start / n
    ex = x1 + (x2 - x1) * end / n
    return int(sx), int(y1), int(ex), int(y2)


def redact_image(image, pad=4):
    out = image.copy()
    findings = []

    # 1. Text-based PII (Aadhaar, PAN, phone, ...)
    for bbox, text, conf in get_reader().readtext(image):
        for label, s, e, value in find_pii(text):
            x1, y1, x2, y2 = span_to_box(bbox, text, s, e)
            cv2.rectangle(out, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), (0, 0, 0), -1)
            findings.append({
                "type": label,
                "value (masked)": mask(value),
                "confidence": round(float(conf), 2),
            })

    # 2. Faces
    hide_faces(image, out, findings)

    return out, findings


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python redactor.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        print(f"Could not open {path}. Check the file name.")
        sys.exit(1)

    redacted, found = redact_image(img)
    out_path = path.rsplit(".", 1)[0] + "_redacted.png"
    cv2.imwrite(out_path, redacted)

    print(f"Found {len(found)} item(s):")
    for f in found:
        print(" ", f)
    print(f"Saved -> {out_path}")
