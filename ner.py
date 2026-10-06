"""
ner.py - uses YOUR fine-tuned model (models/ner) to find names and addresses
in a line of text, and returns their character positions.
"""
import os
import re

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

NER_DIR = os.path.join("models", "ner")

# Names/addresses the model is less sure about than this are ignored.
# Chosen by running evaluate.py on the HARD test set (see results/).
MIN_CONFIDENCE = 0.5

_tokenizer = None
_model = None


def ner_available():
    return os.path.isdir(NER_DIR)


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(NER_DIR)
        _model = AutoModelForTokenClassification.from_pretrained(NER_DIR)
        _model.eval()
    return _tokenizer, _model


def find_names_addresses(text):
    """
    Returns a list of (label, start_char, end_char, value, confidence),
    where label is "NAME" or "ADDRESS".
    """
    # Each word plus where it starts and ends in the text
    words = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    if not words or not ner_available():
        return []

    tokenizer, model = _load()
    enc = tokenizer(
        [w for w, _, _ in words],
        is_split_into_words=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[0]
    probs = logits.softmax(dim=-1)

    # One tag + confidence per word (taken from the word's first sub-word piece)
    tags = ["O"] * len(words)
    conf = [0.0] * len(words)
    prev = None
    for k, w in enumerate(enc.word_ids()):
        if w is not None and w != prev:
            best = int(probs[k].argmax())
            tags[w] = model.config.id2label[best]
            conf[w] = float(probs[k][best])
        prev = w

    # Group B-/I- tags into whole names / addresses
    results = []
    i = 0
    while i < len(words):
        if tags[i] == "O":
            i += 1
            continue
        typ = tags[i][2:]
        j = i + 1
        while j < len(words) and tags[j] == f"I-{typ}":
            j += 1
        label = "NAME" if typ == "NAME" else "ADDRESS"
        value = " ".join(w for w, _, _ in words[i:j])
        confidence = min(conf[i:j])
        if confidence >= MIN_CONFIDENCE:
            results.append((label, words[i][1], words[j - 1][2], value, confidence))
        i = j
    return results


if __name__ == "__main__":
    for line in ["Name: Ravi Kumar", "S/O SURESH GOWDA", "Phone: +91 98765 43210"]:
        print(line, "->", find_names_addresses(line))
