"""
detectors.py - finds personal information (PII) in a line of text.

Each detector returns (label, start, end, value) where start/end are
character positions inside the text. We need positions so we can later
black out ONLY the sensitive part of a line, not the whole line.
"""
import re

# ---------------------------------------------------------------
# Verhoeff checksum - the algorithm UIDAI uses for Aadhaar numbers.
# The last digit of every real Aadhaar is a check digit. Validating it
# means a random 12-digit number (like an invoice ID) is NOT flagged.
# This cuts false positives by ~90%.
# ---------------------------------------------------------------
D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]
INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_valid(number: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = D[c][P[i % 8][int(ch)]]
    return c == 0


def verhoeff_check_digit(number: str) -> str:
    """Used to generate valid FAKE Aadhaar numbers for testing."""
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = D[c][P[(i + 1) % 8][int(ch)]]
    return str(INV[c])


# Set to False if you want to redact every 12-digit number, valid or not.
AADHAAR_STRICT = True

# ---------------------------------------------------------------
# Regex patterns. (?<!\d) and (?!\d) mean "not touching another digit".
# ---------------------------------------------------------------
PATTERNS = {
    "AADHAAR": re.compile(r"(?<!\d)[2-9]\d{3}\s?\d{4}\s?\d{4}(?!\d)"),
    "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{4}\s?\d{5}(?!\d)"),
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "IFSC": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "DOB": re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b"),
}


def find_pii(text: str):
    results = []
    for label, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            value = m.group()
            if label == "AADHAAR" and AADHAAR_STRICT:
                if not verhoeff_valid(re.sub(r"\s", "", value)):
                    continue
            results.append((label, m.start(), m.end(), value))
    return results


def mask(value: str) -> str:
    """Show only the last 4 characters, e.g. XXXX XXXX 9012."""
    keep = 4
    return "".join("X" if ch.isalnum() else ch for ch in value[:-keep]) + value[-keep:]


if __name__ == "__main__":
    # Quick self-test: run `python detectors.py`
    sample = "Name: Ravi Kumar DOB: 12/05/2003 PAN: ABCDE1234F Ph: +91 98765 43210"
    for r in find_pii(sample):
        print(r)
