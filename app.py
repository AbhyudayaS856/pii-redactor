"""
app.py - web interface. Run: streamlit run app.py
"""
import cv2
import numpy as np
import streamlit as st

from redactor import redact_image

st.set_page_config(page_title="PII Redactor", page_icon="🔒", layout="wide")
st.title("🔒 PII Redactor")
st.caption("Detects and hides Aadhaar, PAN, phone numbers, emails and faces. Runs fully offline.")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    data = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    with st.spinner("Scanning for personal information..."):
        redacted, findings = redact_image(img)

    col1, col2 = st.columns(2)
    col1.image(img, channels="BGR", caption="Original")
    col2.image(redacted, channels="BGR", caption="Redacted")

    if findings:
        st.subheader(f"Found {len(findings)} sensitive item(s)")
        st.dataframe(findings, width="stretch")
    else:
        st.info("No sensitive information detected.")

    ok, buf = cv2.imencode(".png", redacted)
    st.download_button("⬇ Download redacted image", buf.tobytes(), "redacted.png", "image/png")
