# run_0 — the baseline score lives here

Do NOT hand-write final_info.json into this folder. Generate it:

1. Finish filling in `code/experiment.py`.
2. From the task folder (after copying to `InternAgent/tasks/PaperTask/`):

   ```bash
   python code/experiment.py          # writes final_info.json to the task folder
   mv final_info.json run_0/          # baseline score in place
   ```

3. Sanity-check `run_0/final_info.json` — its keys must match the
   `metrics` block in `prompt.json`. That number is what every
   experiment gets compared against.

Delete this placeholder file once final_info.json is in place.
