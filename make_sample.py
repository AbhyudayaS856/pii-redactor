"""
make_sample.py - creates a FAKE ID-card image for testing.
Never test with your real Aadhaar/PAN. Run: python make_sample.py
"""
import random

from PIL import Image, ImageDraw, ImageFont

from detectors import verhoeff_check_digit


def fake_aadhaar():
    base = str(random.randint(2, 9)) + "".join(random.choices("0123456789", k=10))
    n = base + verhoeff_check_digit(base)
    return f"{n[:4]} {n[4:8]} {n[8:]}"


def fake_pan():
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return (
        "".join(random.choices(letters, k=5))
        + "".join(random.choices("0123456789", k=4))
        + random.choice(letters)
    )


lines = [
    "GOVERNMENT OF INDIA (SAMPLE - NOT REAL)",
    "Name: Ravi Kumar",
    "DOB: 14/08/2003",
    "Phone: +91 98765 43210",
    "Email: ravi.kumar@example.com",
    f"PAN: {fake_pan()}",
    f"Aadhaar: {fake_aadhaar()}",
]

img = Image.new("RGB", (900, 480), "white")
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("arial.ttf", 32)
except OSError:
    font = ImageFont.load_default(size=32)

y = 30
for line in lines:
    draw.text((40, y), line, fill="black", font=font)
    y += 60

img.save("sample.png")
print("Saved sample.png with:")
for line in lines:
    print(" ", line)
