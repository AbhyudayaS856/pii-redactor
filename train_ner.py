"""
train_ner.py - fine-tunes DistilBERT to find NAMES and ADDRESSES.

Run:    python train_ner.py
Needs:  data/train.json and data/test.json (from generate_data.py)
Output: models/ner/  (the trained model) and models/ner/metrics.json
"""
import json
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

BASE_MODEL = "distilbert-base-cased"  # "cased" keeps capital letters, which helps with names
OUT_DIR = os.path.join("models", "ner")
LABELS = ["O", "B-NAME", "I-NAME", "B-ADDR", "I-ADDR"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

MAX_LEN = 64      # max sub-word pieces per line
EPOCHS = 2        # passes over the training data
BATCH_SIZE = 16
LEARNING_RATE = 5e-5

torch.manual_seed(42)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def encode(examples, tokenizer):
    """
    The tokenizer splits words into sub-word pieces, e.g. "Kozhikode" -> "Ko", "##zh", "##ik", "##ode".
    We give each WORD's label to its FIRST piece and mark the other pieces with -100,
    which tells PyTorch to ignore them when calculating the loss.
    """
    enc = tokenizer(
        [ex["tokens"] for ex in examples],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    all_labels = []
    for i, ex in enumerate(examples):
        labels, prev = [], None
        for w in enc.word_ids(batch_index=i):
            if w is None or w == prev:
                labels.append(-100)
            else:
                labels.append(LABEL2ID[ex["tags"][w]])
            prev = w
        all_labels.append(labels)
    return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(all_labels))


def predict_words(model, tokenizer, examples, batch_size=64):
    """Return one predicted tag per WORD for each example."""
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            enc = tokenizer(
                [ex["tokens"] for ex in batch],
                is_split_into_words=True,
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            best = logits.argmax(dim=-1)
            for j, ex in enumerate(batch):
                tags, prev = ["O"] * len(ex["tokens"]), None
                for k, w in enumerate(enc.word_ids(batch_index=j)):
                    if w is not None and w != prev:
                        tags[w] = LABELS[best[j, k].item()]
                    prev = w
                results.append(tags)
    return results


def spans(tags):
    """Turn B-/I- tags into entities: {("NAME", start_word, end_word), ...}"""
    found, start, typ = set(), None, None
    for i, t in enumerate(tags + ["O"]):
        if t.startswith("I-") and start is not None and t[2:] == typ:
            continue
        if start is not None:
            found.add((typ, start, i))
            start = None
        if t != "O":
            start, typ = i, t[2:]
    return found


def evaluate(gold_examples, predicted_tags):
    """
    Entity-level scores: an entity only counts as correct if the WHOLE
    name/address is found with exactly the right boundaries.
    """
    scores = {}
    for typ in ["NAME", "ADDR", "ALL"]:
        tp = fp = fn = 0
        for ex, pred in zip(gold_examples, predicted_tags):
            g = {s for s in spans(ex["tags"]) if typ == "ALL" or s[0] == typ}
            p = {s for s in spans(pred) if typ == "ALL" or s[0] == typ}
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[typ] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
    return scores


if __name__ == "__main__":
    train_data = load("data/train.json")
    test_data = load("data/test.json")
    print(f"Train: {len(train_data)} examples | Test: {len(test_data)} examples")

    print(f"Downloading/loading {BASE_MODEL} (first time ~260 MB)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
    )

    loader = DataLoader(encode(train_data, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for step, (ids, mask, labels) in enumerate(loader, 1):
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()      # work out how to adjust the weights
            optimizer.step()         # adjust them
            optimizer.zero_grad()
            total += out.loss.item()
            if step % 25 == 0:
                print(f"  epoch {epoch} | step {step}/{len(loader)} | loss {total / step:.4f}")
        print(f"Epoch {epoch} done | avg loss {total / len(loader):.4f} | {time.time() - start:.0f}s elapsed")

    print("\nEvaluating on the unseen test set...")
    scores = evaluate(test_data, predict_words(model, tokenizer, test_data))
    for typ, s in scores.items():
        print(f"  {typ:5s}  precision {s['precision']:.3f}  recall {s['recall']:.3f}  F1 {s['f1']:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nModel saved -> {OUT_DIR}")

    # Quick sanity check on lines the model has never seen
    demo = [
        "Name: Abhishek Ramachandran".split(),
        "S/O PRAKASH NAIK".split(),
        "Address: 45, 3rd Cross, Jayanagar, Bengaluru-560041".split(),
        "GOVERNMENT OF INDIA".split(),
    ]
    for words, tags in zip(demo, predict_words(model, tokenizer, [{"tokens": d} for d in demo])):
        print(list(zip(words, tags)))
