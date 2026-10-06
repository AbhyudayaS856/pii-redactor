"""
app.py - web interface. Run: streamlit run app.py
"""
import cv2
import numpy as np
import streamlit as st

from pdf_redactor import redact_pdf
from redactor import redact_image

st.set_page_config(page_title="PII Redactor", page_icon="🔒", layout="wide")
st.title("🔒 PII Redactor")
st.caption(
    "Detects and hides Aadhaar, PAN, phone numbers, emails, names, addresses and faces "
    "in images and PDFs. Runs fully offline."
)

uploaded = st.file_uploader("Upload an image or PDF", type=["png", "jpg", "jpeg", "pdf"])


def show_findings(findings):
    if findings:
        st.subheader(f"Found {len(findings)} sensitive item(s)")
        st.dataframe(findings, width="stretch")
    else:
        st.info("No sensitive information detected.")


if uploaded and uploaded.name.lower().endswith(".pdf"):
    # ---------------- PDF ----------------
    bar = st.progress(0, text="Redacting pages...")

    def update(done, total):
        bar.progress(done / total, text=f"Redacted page {done} of {total}")

    pdf_bytes, findings, pages = redact_pdf(uploaded.read(), progress=update)
    bar.empty()

    st.download_button("⬇ Download redacted PDF", pdf_bytes, "redacted.pdf", "application/pdf")
    show_findings(findings)

    st.subheader("Preview")
    for i, page in enumerate(pages[:5]):
        st.image(page, channels="BGR", caption=f"Page {i + 1} (redacted)")
    if len(pages) > 5:
        st.caption(f"Showing 5 of {len(pages)} pages. Download the PDF to see all.")

elif uploaded:
    # ---------------- Image ----------------
    data = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    with st.spinner("Scanning for personal information..."):
        redacted, findings = redact_image(img)

    col1, col2 = st.columns(2)
    col1.image(img, channels="BGR", caption="Original")
    col2.image(redacted, channels="BGR", caption="Redacted")

    show_findings(findings)

    ok, buf = cv2.imencode(".png", redacted)
    st.download_button("⬇ Download redacted image", buf.tobytes(), "redacted.png", "image/png")
