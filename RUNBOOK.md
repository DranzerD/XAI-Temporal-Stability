# XDrift — Runbook

`XDrift_XSI_FIXED.ipynb` is your original notebook with 4 real bugs fixed and verified. I found these by actually executing the code (not just reading it) against a synthetic dataset shaped like LendingClub — each one was a genuine crash or silent-failure, not a hypothetical. Details in "What was actually broken" below.

## Fastest path: Kaggle (recommended — this is what the notebook is built for)

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook** → **File → Import Notebook** → upload `XDrift_XSI_FIXED.ipynb`.
2. **Add Data** (right sidebar) → search `lending-club` → add the dataset by `wordsforthewise`. Kaggle will mount it at `/kaggle/input/...` — confirm the path printed by cell 4 (`Dataset found: ...`) matches `RAW_DATA_FILE`. If Kaggle mounts it under a different exact path, update `RAW_DATA_FILE` in the config cell to match (this is the #1 reason people get a `FileNotFoundError` on Kaggle).
3. Turn on **GPU accelerator** (Settings → Accelerator → GPU T4/P100) if available — not required, but `tree_method="hist"` and TreeSHAP both benefit.
4. **First run — smoke test, not the paper run:** in the config cell, leave `N_OPTUNA_TRIALS = 0`, and in the run cell set `SHAP_SAMPLES_PER_WINDOW = 60` (down from 300) just to confirm the whole notebook completes without error. Run All. This should take under 10 minutes on Kaggle's default settings.
5. **Once step 4 succeeds clean, do the real run:** set `SHAP_SAMPLES_PER_WINDOW = 300` (or 500 for the final paper numbers) back in that cell, and Run All again. Budget 20–40 minutes as the notebook itself estimates. Optionally set `N_OPTUNA_TRIALS = 50` first if you want tuned hyperparameters (adds ~20 min).
6. Numbers to pull for your resume bullets afterward, all printed to output or written to `/kaggle/working/results/`:
   - `results_df` in cell 25 — the regime-stratified XSI/AUC table
   - `runner.eart_summary` — EART trigger count, baseline trigger count, mean lead time
   - Cell 13's `auc_test`, `ks`, `gini` — model quality numbers
   - Cell 18's SHAP-vs-LIME Spearman ρ
7. Save Version → Save & Run All (Commit) so the output/figures persist on Kaggle, then download `/kaggle/working/results/xdrift_results.csv` and the `figures/` folder from the output tab.

## Local alternative (slower to set up, useful for iterating on code)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # now includes pyarrow — see fix list below
kaggle datasets download -d wordsforthewise/lending-club
unzip lending-club.zip -d data/
```

Then in the notebook's config cell, change:
```python
RAW_DATA_FILE = "data/accepted_2007_to_2018Q4.csv"   # or wherever you unzipped it
WORKING_DIR    = "./working"
```
and remove/comment the `!pip install -q hmmlearn lime optuna` line since `requirements.txt` already installs them. Then:
```bash
jupyter notebook XDrift_XSI_FIXED.ipynb
```
Run All. Same smoke-test-first advice as above applies locally too — start with a low `SHAP_SAMPLES_PER_WINDOW` to confirm it completes before committing to the full run, especially without a GPU.

You can also execute headlessly to double check everything runs clean before opening it:
```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 \
  --output XDrift_executed.ipynb XDrift_XSI_FIXED.ipynb
```

## What was actually broken (verified by execution, not inspection)

I ran your pipeline logic against synthetic data shaped like the real schema and hit these, in order:

1. **`enable_categorical` defaults to `True` in newer XGBoost (3.x sklearn API).** Your `requirements.txt` pins `xgboost==2.0.3` where the old default was `False`, so you may or may not hit this on Kaggle depending on their pre-installed version — but if XGBoost ever auto-upgrades under you, SHAP's `TreeExplainer` will hard-refuse with `NotImplementedError: Categorical split is not yet supported`. **Fixed:** added `"enable_categorical": False` to `XGB_PARAMS` explicitly, so behavior no longer depends on which XGBoost version happens to be installed.

2. **LIME's discretizer crashes on your low-cardinality numeric columns** (`term`, `initial_list_status`, `has_delinq`, `has_pub_rec`, `vix_bucket` — all binary or near-binary integers). LIME's default quantile discretizer produces zero-width bins for these, and `scipy.stats.truncnorm` throws `ValueError: Domain error in arguments`. This is a real column-schema issue, not a synthetic-data artifact — your real feature set has the same low-cardinality columns. **Fixed:** `LimeTabularExplainer` now gets `categorical_features` for any column with ≤10 unique values, so LIME treats them as categorical instead of trying to discretize them.

3. **The most important one — your median imputation was silently doing nothing.** `df_raw[col].fillna(df_raw[col].median(), inplace=True)` is a documented no-op under pandas Copy-on-Write (default since pandas 2.0, mandatory in pandas 3.0): `df_raw[col]` returns a copy, so `inplace=True` mutates the copy and throws it away. Your NaNs in `emp_length` (and any other column with missing values) survived all the way to the LIME importance step and crashed it with `cannot convert float NaN to integer` — but XGBoost silently tolerated the NaNs upstream (it natively handles missing values), so the bug produced no error until much later, far from its actual cause. This is exactly the kind of bug that's dangerous for a paper: silent data-quality corruption with no crash to alert you. **Fixed:** reassignment (`df_raw[col] = df_raw[col].fillna(...)`) instead of the broken chained `inplace=True`.

4. **`fillna(method="ffill")`** in `RegimeDetector.fit()`/`.predict()` — the `method=` keyword is deprecated since pandas 2.1 and fully removed in pandas 3.0, so this will hard-crash the HMM regime detector on any recent pandas. **Fixed:** replaced with `.ffill()`.

5. **`requirements.txt` never listed a parquet engine**, but `train_df.to_parquet(...)` / `test_df.to_parquet(...)` need one (`pyarrow` or `fastparquet`) — without it, `ImportError: Unable to find a usable engine`. Likely invisible on Kaggle (pyarrow is pre-installed there) but will break any local run. **Fixed:** added `pyarrow>=14.0.0` to `requirements.txt`.

None of these are logic/methodology bugs — your XSI math, HMM setup, and EART trigger logic are unchanged and were confirmed to run correctly end-to-end (all 43 rolling windows processed, regime labels assigned, EART vs. baseline triggers computed, all 11 figures generated) once the above were fixed. **The numbers from my synthetic-data test run are meaningless for the paper** (the synthetic data is random noise with no real signal — AUC hovered near 0.50 and every window triggered both EART and baseline, which is exactly what you'd expect from noise, not evidence of anything). They only prove the code path executes; you still need the real LendingClub run for real numbers.
