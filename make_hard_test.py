"""
make_hard_test.py - builds a HARDER test set the model was never trained on.

It is "harder" in three ways:
  1. New formats   - "Payment received from <name> via UPI", "Nominee:", "Mr./Smt."
  2. Real Bengaluru-style addresses - "#12, 4th Main, 9th Cross, Jayanagar, ..."
  3. OCR noise     - scanners misread letters (O->0, l->1, S->5, ...)
plus tricky NEGATIVES: company names that contain person-like words
("Tata Consultancy Services", "Ashok Leyland Ltd").

Run: python make_hard_test.py   ->  data/hard_test.json
"""
import json
import random

from faker import Faker

from generate_data import build

fake = Faker("en_IN")

AREAS = ["Jayanagar", "Koramangala", "Indiranagar", "Malleshwaram", "Rajajinagar",
         "BTM Layout", "HSR Layout", "Basavanagudi", "Yelahanka", "Whitefield"]

COMPANIES = ["Tata Consultancy Services", "Infosys Limited", "Ashok Leyland Ltd",
             "Reliance Jio Infocomm", "Mahindra and Mahindra", "Bajaj Finserv",
             "Kotak Mahindra Bank", "Ramaiah Institute of Technology", "Wipro Technologies"]

OTHER_NEGATIVES = ["Account Type: Savings", "Amount: Rs. 4,500.00", "Mode: NEFT",
                   "Status: Successful", "Branch Code: 00412", "Closing Balance"]

# Letters OCR commonly confuses
OCR_SWAPS = {"O": "0", "o": "0", "I": "l", "l": "1", "S": "5", "B": "8", "e": "c", "a": "o"}


def ocr_noise(text, rate=0.08):
    """Randomly corrupt some letters, like a bad scan. Spaces are kept, so word count stays the same."""
    return "".join(OCR_SWAPS[c] if c in OCR_SWAPS and random.random() < rate else c for c in text)


def blr_address():
    return (f"#{random.randint(1, 999)}, {random.randint(1, 18)}th Main, "
            f"{random.randint(1, 30)}th Cross, {random.choice(AREAS)}, "
            f"Bengaluru - 5600{random.randint(10, 99)}")


def make_hard_example():
    name = fake.name()
    if random.random() < 0.2:
        name = name.lower()
    addr = blr_address() if random.random() < 0.6 else fake.address().replace("\n", ", ")

    templates = [
        [("Customer Name -", "O"), (name, "NAME")],
        [("Nominee:", "O"), (name, "NAME")],
        [("Payment received from", "O"), (name, "NAME"), ("via UPI", "O")],
        [("Mr.", "O"), (name, "NAME")],
        [("Smt.", "O"), (name, "NAME")],
        [("Residential Address:", "O"), (addr, "ADDR")],
        [(addr, "ADDR")],
        [("Ship to:", "O"), (name, "NAME"), (addr, "ADDR")],
        [(random.choice(COMPANIES), "O")],
        [("Employer:", "O"), (random.choice(COMPANIES), "O")],
        [(random.choice(OTHER_NEGATIVES), "O")],
    ]
    parts = random.choice(templates)
    if random.random() < 0.4:
        parts = [(ocr_noise(text), label) for text, label in parts]
    return build(parts)


if __name__ == "__main__":
    random.seed(123)
    Faker.seed(123)
    examples = [make_hard_example() for _ in range(500)]
    with open("data/hard_test.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=1)
    print(f"Saved {len(examples)} hard examples -> data/hard_test.json")
    for ex in random.sample(examples, 4):
        print(list(zip(ex["tokens"], ex["tags"])))
