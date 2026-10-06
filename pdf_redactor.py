"""
pdf_redactor.py - redacts every page of a PDF.

How it works:
  1. Each PDF page is rendered into an image (like taking a high-quality photo of it).
  2. That image goes through the same redact_image() used for photos.
  3. The redacted images are put back together into a new PDF.

Why turn pages into images? Drawing a black box over PDF text often leaves
the text underneath still selectable and copyable - a famous real-world
redaction mistake. Rebuilding the PDF from images guarantees the hidden
data is truly gone.

Usage:  python pdf_redactor.py sample.pdf
Output: sample_redacted.pdf
"""
import sys

import cv2
import numpy as np
import pymupdf

from redactor import redact_image


def redact_pdf(pdf_bytes, dpi=200, progress=None):
    """
    Returns (redacted_pdf_bytes, findings, page_images).
    progress: optional function called as progress(done_pages, total_pages).
    """
    src = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    out = pymupdf.open()
    findings, page_images = [], []

    for i, page in enumerate(src):
        # 1. Render the page to an image (higher dpi = sharper text for OCR)
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 2. Redact it
        redacted, found = redact_image(img)
        for item in found:
            item["page"] = i + 1
        findings += found
        page_images.append(redacted)

        # 3. Add the redacted image as a page of the new PDF, same size as the original
        ok, buf = cv2.imencode(".png", redacted)
        new_page = out.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=buf.tobytes())

        if progress:
            progress(i + 1, len(src))

    return out.tobytes(), findings, page_images


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_redactor.py <file.pdf>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "rb") as f:
        data = f.read()

    redacted_pdf, found, _ = redact_pdf(
        data, progress=lambda done, total: print(f"  page {done}/{total} done")
    )

    out_path = path.rsplit(".", 1)[0] + "_redacted.pdf"
    with open(out_path, "wb") as f:
        f.write(redacted_pdf)

    print(f"Found {len(found)} item(s):")
    for item in found:
        print(" ", item)
    print(f"Saved -> {out_path}")
