"""
evaluate.py - measures how good the NER model REALLY is.

Tests on:  data/test.json       (easy: same style as training)
           data/hard_test.json  (hard: new formats, OCR noise, tricky company names)
at several CONFIDENCE THRESHOLDS: a name/address is only accepted if the
model is at least that confident about every word in it.

Run: python evaluate.py   ->  prints a table and saves results/evaluation.json
"""
import json
import os

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ner import NER_DIR
from train_ner import MAX_LEN, evaluate, load, spans

THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]


def predict_with_confidence(model, tokenizer, examples, batch_size=64):
    """For each example: (tags, confidences), one per word."""
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            enc = tokenizer([ex["tokens"] for ex in batch], is_split_into_words=True,
                            truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
            probs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.softmax(-1)
            for j, ex in enumerate(batch):
                n = len(ex["tokens"])
                tags, confs, prev = ["O"] * n, [1.0] * n, None
                for k, w in enumerate(enc.word_ids(batch_index=j)):
                    if w is not None and w != prev:
                        best = int(probs[j, k].argmax())
                        tags[w] = model.config.id2label[best]
                        confs[w] = float(probs[j, k, best])
                    prev = w
                results.append((tags, confs))
    return results


def apply_threshold(tags, confs, threshold):
    """Drop any whole name/address whose weakest word is below the threshold (same rule as ner.py)."""
    tags = list(tags)
    for typ, s, e in spans(tags):
        if min(confs[s:e]) < threshold:
            tags[s:e] = ["O"] * (e - s)
    return tags


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(NER_DIR)
    model = AutoModelForTokenClassification.from_pretrained(NER_DIR)

    all_results = {}
    for name, path in [("easy test", "data/test.json"), ("HARD test", "data/hard_test.json")]:
        data = load(path)
        preds = predict_with_confidence(model, tokenizer, data)
        print(f"\n=== {name} ({len(data)} examples) ===")
        print("threshold | precision  recall   F1    | NAME F1  ADDR F1")
        all_results[name] = {}
        for t in THRESHOLDS:
            tagged = [apply_threshold(tags, confs, t) for tags, confs in preds]
            s = evaluate(data, tagged)
            all_results[name][t] = s
            a = s["ALL"]
            print(f"   {t:.1f}    |   {a['precision']:.3f}    {a['recall']:.3f}  {a['f1']:.3f}  |"
                  f"  {s['NAME']['f1']:.3f}    {s['ADDR']['f1']:.3f}")

        if name == "HARD test":
            print("\nSome mistakes on the HARD test (threshold 0.0):")
            shown = 0
            for ex, (tags, _) in zip(data, preds):
                if spans(ex["tags"]) != spans(tags) and shown < 6:
                    print("  text:     ", " ".join(ex["tokens"]))
                    print("  expected: ", sorted(spans(ex["tags"])))
                    print("  predicted:", sorted(spans(tags)))
                    shown += 1

    os.makedirs("results", exist_ok=True)
    with open("results/evaluation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved -> results/evaluation.json")
