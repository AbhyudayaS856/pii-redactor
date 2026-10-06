"""
generate_data.py - creates a labelled training dataset for the NER model.

Each example is one line of text, like what OCR reads from an ID card,
with a label for every word:
    O       = not sensitive
    B-NAME  = first word of a person's name,  I-NAME = rest of the name
    B-ADDR  = first word of an address,       I-ADDR = rest of the address

Run: python generate_data.py
Output: data/train.json and data/test.json

v2: added lowercase names, OCR noise, title prefixes, Karnataka-style
addresses and company names (negatives) after evaluation showed these
were weak spots.
"""
import json
import os
import random

from faker import Faker

fake = Faker("en_IN")  # Indian names and addresses

# Lines that appear on Indian documents but contain NO names or addresses.
# The model must learn to leave these alone.
NEGATIVE_LINES = [
    "GOVERNMENT OF INDIA", "Government of India",
    "Unique Identification Authority of India",
    "INCOME TAX DEPARTMENT", "Permanent Account Number Card",
    "Male", "Female", "MALE", "FEMALE", "Year of Birth",
    "Mera Aadhaar, Meri Pehchan", "Signature", "Valid till",
    "Date of Issue", "STATE BANK OF INDIA", "Statement of Account",
    "Opening Balance", "Transaction Details", "Reference Number",
    "Blood Group: O+", "Student Identity Card",
    "Visvesvaraya Technological University",
    "Department of Computer Science and Engineering",
    "Enrolment No.", "Help Line 1947", "This is a computer generated document",
]


# v2: company names containing person-like words, so the model learns they are NOT people.
# (Deliberately DIFFERENT from the companies in make_hard_test.py, to avoid data leakage.)
TRAIN_COMPANIES = [
    "Sundaram Finance", "Muthoot Finance Ltd", "Godrej Industries", "Birla Cement Works",
    "Murugappa Group", "Shriram Transport Finance", "Kalyan Jewellers", "Narayana Health",
    "Larsen and Toubro", "Hindustan Unilever Limited", "Prestige Estates Projects",
]

TRAIN_AREAS = ["Hebbal", "Banashankari", "Vijayanagar", "Marathahalli", "JP Nagar",
               "Electronic City", "Kuvempunagar", "Hampankatta", "Vidyanagar"]
TRAIN_CITIES = ["Bengaluru", "Mysuru", "Mangaluru", "Hubballi", "Belagavi"]

OCR_SWAPS = {"O": "0", "o": "0", "I": "l", "l": "1", "S": "5", "B": "8", "e": "c", "a": "o"}


def ocr_noise(text, rate=0.08):
    """Corrupt some letters like a bad scan, so the model handles imperfect OCR."""
    return "".join(OCR_SWAPS[c] if c in OCR_SWAPS and random.random() < rate else c for c in text)


def fake_address():
    if random.random() < 0.5:
        # Karnataka-style address
        return (f"No. {random.randint(1, 500)}, {random.choice(['1st', '2nd', '3rd'])} Stage, "
                f"{random.choice(TRAIN_AREAS)}, {random.choice(TRAIN_CITIES)}, "
                f"PIN {random.randint(560001, 591999)}")
    return fake.address().replace("\n", ", ")


def vary_case(text):
    """ID cards often print names in CAPITALS; typed forms are sometimes all lowercase."""
    r = random.random()
    if r < 0.3:
        return text.upper()
    if r < 0.4:
        return text.lower()
    return text


def tag_words(text, label):
    words = text.split()
    if label == "O":
        return words, ["O"] * len(words)
    return words, [f"B-{label}"] + [f"I-{label}"] * (len(words) - 1)


def build(parts):
    tokens, tags = [], []
    for text, label in parts:
        w, t = tag_words(text, label)
        tokens += w
        tags += t
    return {"tokens": tokens, "tags": tags}


def make_example():
    name = vary_case(fake.name())
    other = vary_case(fake.name())
    addr = vary_case(fake_address())
    rel = random.choice(["S/O", "D/O", "W/O", "C/O"])

    templates = [
        # --- lines WITH names / addresses ---
        [("Name:", "O"), (name, "NAME")],
        [(name, "NAME")],
        [(rel, "O"), (other, "NAME")],
        [("Father's Name:", "O"), (other, "NAME")],
        [("Address:", "O"), (addr, "ADDR")],
        [(addr, "ADDR")],
        [(rel, "O"), (other, "NAME"), (addr, "ADDR")],
        [("To,", "O"), (name, "NAME"), (addr, "ADDR")],
        [("Account Holder:", "O"), (name, "NAME")],
        [("Signature of", "O"), (name, "NAME")],
        [("Student Name:", "O"), (name, "NAME"), ("USN:", "O"), ("1KS21CS" + str(random.randint(100, 999)), "O")],
        [(random.choice(["Shri", "Dr.", "Ms.", "Kumari"]), "O"), (name, "NAME")],
        [("Transferred to", "O"), (name, "NAME")],
        [("Paid by", "O"), (name, "NAME"), ("on", "O"), (fake.date(pattern="%d/%m/%Y"), "O")],
        [("Permanent Address:", "O"), (addr, "ADDR")],
        # --- lines WITHOUT names / addresses (negatives) ---
        [(random.choice(NEGATIVE_LINES), "O")],
        [(random.choice(NEGATIVE_LINES), "O")],
        [("DOB:", "O"), (fake.date(pattern="%d/%m/%Y"), "O")],
        [("Phone:", "O"), (fake.phone_number(), "O")],
        [("Email:", "O"), (fake.email(), "O")],
        [(random.choice(TRAIN_COMPANIES), "O")],
        [("Company:", "O"), (random.choice(TRAIN_COMPANIES), "O")],
        [("Beneficiary:", "O"), (name, "NAME"), ("Bank:", "O"), (random.choice(TRAIN_COMPANIES), "O")],
    ]
    parts = random.choice(templates)
    if random.random() < 0.3:
        parts = [(ocr_noise(text), label) for text, label in parts]
    return build(parts)


def save(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=1)
    print(f"Saved {len(examples)} examples -> {path}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    random.seed(42)
    Faker.seed(42)
    save([make_example() for _ in range(6000)], "data/train.json")

    # Different seed, so the test set has names the model has never seen
    random.seed(7)
    Faker.seed(7)
    save([make_example() for _ in range(600)], "data/test.json")

    print("\nThree random examples:")
    with open("data/train.json", encoding="utf-8") as f:
        sample = random.sample(json.load(f), 3)
    for ex in sample:
        print(list(zip(ex["tokens"], ex["tags"])))
