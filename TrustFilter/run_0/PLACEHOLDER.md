# run_0 — generate the baseline score (don't hand-write it)

On the cloud box, after copying this folder to `InternAgent/tasks/TrustFilter/`:

```bash
pip install torch transformers pillow numpy      # if not already present
cd tasks/TrustFilter
python code/experiment.py                        # first run downloads COCO (~1GB, cached in ./data)
mv final_info.json run_0/
```

Expect `detection_f1` somewhere around 0.6-0.8 (CLIP catches swapped captions
well but is weak on shuffles/truncations — that gap is the improvement room).

Sanity checks before launching the pipeline:
- `run_0/final_info.json` exists and has `detection_f1` as a key (matches prompt.json).
- Runtime printed was a few minutes at most (after the one-time download).

Delete this placeholder file once final_info.json is in place.
