"""
TrustFilter baseline — CLIP-similarity image-text filtration.

Based on: "Trust the Model: Compact VLMs as In-Context Judges for Image-Text
Data Quality" (Humain, IEEE MLSP 2025). The paper's compact-VLM judge is the
kind of method that should BEAT this baseline; the baseline implemented here
is the CLIP-similarity scoring the paper argues against.

Task shape: N COCO pairs, a frozen seeded corruption of 30% of them, and a
scoring function. The lowest-scoring 30% are flagged as corrupted. Primary
metric = detection_f1 against the known corruption labels.

THE CONTRACT:
  - Only the SCORING section may be modified by experiments. Data loading,
    corruption, and evaluation are FROZEN (they are the ground truth).
  - Tunables are argparse defaults.
  - Writes final_info.json to the parent directory of this script.

Deps: torch, transformers, pillow, numpy  (pip install torch transformers pillow numpy)
First run downloads COCO val2017 (~780MB) + annotations (~240MB) + CLIP ViT-B/32;
everything is cached under --data_dir afterwards.
"""
import os
import io
import json
import time
import random
import zipfile
import argparse
import urllib.request

import numpy as np
import torch
from PIL import Image

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


# ==== FROZEN: DATA =====================================================
def _download(url: str, dest: str):
    if not os.path.exists(dest):
        print(f"Downloading {url} -> {dest}")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(url, dest)


def load_pairs(data_dir: str, n_pairs: int, seed: int):
    """Return list of (PIL.Image, caption) sampled deterministically from COCO val2017."""
    img_zip_path = os.path.join(data_dir, "val2017.zip")
    ann_zip_path = os.path.join(data_dir, "annotations_trainval2017.zip")
    _download(COCO_IMAGES_URL, img_zip_path)
    _download(COCO_ANN_URL, ann_zip_path)

    with zipfile.ZipFile(ann_zip_path) as zf:
        with zf.open("annotations/captions_val2017.json") as f:
            ann = json.load(f)

    file_by_id = {im["id"]: im["file_name"] for im in ann["images"]}
    caption_by_id = {}
    for a in sorted(ann["annotations"], key=lambda a: a["id"]):
        caption_by_id.setdefault(a["image_id"], a["caption"].strip())

    ids = sorted(caption_by_id.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    ids = ids[:n_pairs]

    pairs = []
    with zipfile.ZipFile(img_zip_path) as zf:
        for img_id in ids:
            with zf.open(f"val2017/{file_by_id[img_id]}") as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
            pairs.append((img, caption_by_id[img_id]))
    return pairs


# ==== FROZEN: CORRUPTION (the ground truth being detected) =============
def corrupt_pairs(pairs, corruption_rate: float, seed: int):
    """Corrupt a fixed fraction of captions. Returns (pairs, labels); label 1 = corrupted."""
    rng = random.Random(seed + 1)
    n = len(pairs)
    n_corrupt = int(round(n * corruption_rate))
    corrupt_idx = rng.sample(range(n), n_corrupt)
    labels = [0] * n
    out = [list(p) for p in pairs]

    # Type 1: caption swap — rotate captions among the first group (fluent but misaligned)
    swap_group = corrupt_idx[: n_corrupt // 2]
    if len(swap_group) > 1:
        rotated = [out[swap_group[-1]][1]] + [out[i][1] for i in swap_group[:-1]]
        for i, cap in zip(swap_group, rotated):
            out[i][1] = cap

    # Types 2-4: degradations (aligned-ish words but broken text)
    for k, i in enumerate(corrupt_idx[n_corrupt // 2:]):
        words = out[i][1].split()
        t = k % 3
        if t == 0:      # word-order shuffle
            rng.shuffle(words)
            out[i][1] = " ".join(words)
        elif t == 1:    # truncate to 3-word fragment
            out[i][1] = " ".join(words[:3])
        else:           # generic template caption
            out[i][1] = "an interesting photo of some things"

    for i in corrupt_idx:
        labels[i] = 1
    return [tuple(p) for p in out], labels


# ==== SCORING (the method under test — THIS is what experiments improve) ====
def score_pairs(pairs, device: str, clip_model_name: str, batch_size: int):
    """Baseline: CLIP image-text cosine similarity. Higher = judged higher quality."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    scores = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            inputs = processor(
                text=[c for _, c in batch], images=[im for im, _ in batch],
                return_tensors="pt", padding=True, truncation=True,
            ).to(device)
            img_emb = model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_emb = model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            scores.extend((img_emb * txt_emb).sum(dim=-1).cpu().tolist())
    return scores
# =============================================================================


# ==== FROZEN: EVALUATION ================================================
def evaluate(scores, labels, corruption_rate: float):
    """Flag the lowest-scoring corruption_rate fraction; F1 vs ground truth."""
    n = len(scores)
    n_flag = int(round(n * corruption_rate))
    flagged = set(np.argsort(scores)[:n_flag].tolist())

    tp = sum(1 for i in flagged if labels[i] == 1)
    precision = tp / max(len(flagged), 1)
    recall = tp / max(sum(labels), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    kept = [i for i in range(n) if i not in flagged]
    return f1, precision, recall, kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_pairs", type=int, default=1200)        # FROZEN
    parser.add_argument("--corruption_rate", type=float, default=0.3)  # FROZEN
    parser.add_argument("--seed", type=int, default=42)             # FROZEN
    parser.add_argument("--data_dir", type=str,
                        default=os.environ.get("TRUSTFILTER_DATA", "./data"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    pairs = load_pairs(args.data_dir, args.n_pairs, args.seed)
    pairs, labels = corrupt_pairs(pairs, args.corruption_rate, args.seed)
    scores = score_pairs(pairs, device, args.clip_model, args.batch_size)
    f1, precision, recall, kept = evaluate(scores, labels, args.corruption_rate)

    # Secondary: mean CLIP similarity of the kept set (reported, not optimized)
    kept_sim = float(np.mean([scores[i] for i in kept]))

    metrics = {
        "detection_f1": round(f1, 4),
        "detection_precision": round(precision, 4),
        "detection_recall": round(recall, 4),
        "kept_clip_similarity": round(kept_sim, 4),
        "runtime_seconds": round(time.time() - t0, 1),
        "config": {k: v for k, v in vars(args).items() if k != "data_dir"},
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(os.path.dirname(script_dir), "final_info.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to: {output_path}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
