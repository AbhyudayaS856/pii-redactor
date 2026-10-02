"""
redactor.py - reads text from an image with OCR, finds PII, blacks it out.

Usage:  python redactor.py sample.png
Output: sample_redacted.png
"""
import sys

import cv2
import easyocr

from detectors import find_pii, mask

_reader = None


def get_reader():
    """Load the OCR model once (it's slow to load, fast to reuse)."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


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
    for bbox, text, conf in get_reader().readtext(image):
        for label, s, e, value in find_pii(text):
            x1, y1, x2, y2 = span_to_box(bbox, text, s, e)
            cv2.rectangle(out, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), (0, 0, 0), -1)
            findings.append({
                "type": label,
                "value (masked)": mask(value),
                "ocr confidence": round(float(conf), 2),
            })
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
