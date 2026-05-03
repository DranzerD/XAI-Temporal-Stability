"""
Phase 1: Code Cleanup & Reproducibility
Transforms xai-jackfruit.ipynb programmatically.
"""
import json, copy, sys

sys.stdout.reconfigure(encoding='utf-8')

with open('xai-jackfruit.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Original: {len(cells)} cells")

# ── 1.1: Remove corrupted JS from Cell 3 output ──
for i, c in enumerate(cells):
    for j, out in enumerate(c.get('outputs', [])):
        out_text = json.dumps(out)
        if '!0}},e.createElement' in out_text:
            print(f"  Removing corrupted JS from Cell {i}, Output {j} ({len(out_text)} chars)")
            c['outputs'][j] = {
                "output_type": "stream",
                "name": "stdout",
                "text": ["[Output cleared: corrupted JS artifact removed]\n"]
            }

# ── 1.6: Remove debug Cell 4 (os.walk /kaggle/input) ──
# Cell 4 source: import os; for root, dirs, files in os.walk("/kaggle/input"):
cell4_src = ''.join(cells[4]['source'])
if 'os.walk' in cell4_src and '/kaggle/input' in cell4_src:
    print("  Removing debug Cell 4 (os.walk)")
    cells.pop(4)

# ── 1.2: Fix random seeds globally ──
# Find the imports cell (Cell 3, now Cell 3 after removing Cell 4)
# Insert global seed block right after imports
imports_idx = None
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    if 'import numpy as np' in src and 'import pandas as pd' in src:
        imports_idx = i
        break

if imports_idx is not None:
    src_lines = cells[imports_idx]['source']
    # Check if seed is already set
    full_src = ''.join(src_lines)
    if 'np.random.seed' not in full_src:
        print(f"  Adding global random seeds to Cell {imports_idx}")
        seed_block = [
            "\n",
            "# ── Reproducibility seeds ──────────────────────────────────────\n",
            "import random\n",
            "SEED = 42\n",
            "random.seed(SEED)\n",
            "np.random.seed(SEED)\n",
        ]
        # Insert after warnings.filterwarnings line
        insert_pos = len(src_lines)
        for j, line in enumerate(src_lines):
            if 'warnings.filterwarnings' in line:
                insert_pos = j + 1
                break
        cells[imports_idx]['source'] = src_lines[:insert_pos] + seed_block + src_lines[insert_pos:]

# ── 1.3: Standardize DPI to 300 everywhere ──
dpi_fixes = 0
for i, c in enumerate(cells):
    if c['cell_type'] != 'code':
        continue
    new_source = []
    for line in c['source']:
        if 'dpi=150' in line:
            line = line.replace('dpi=150', 'dpi=300')
            dpi_fixes += 1
        elif 'dpi=200' in line:
            line = line.replace('dpi=200', 'dpi=300')
            dpi_fixes += 1
        new_source.append(line)
    c['source'] = new_source
print(f"  Fixed DPI in {dpi_fixes} locations → 300")

# ── 1.5: Add RUN_METADATA dict ──
# Find the configuration cell (the one with WORKING_DIR, FIGURES_DIR, etc.)
config_idx = None
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    if 'WORKING_DIR' in src and 'FIGURES_DIR' in src and 'RANDOM_SEED' in src:
        config_idx = i
        break

if config_idx is not None:
    src_lines = cells[config_idx]['source']
    full_src = ''.join(src_lines)
    if 'RUN_METADATA' not in full_src:
        print(f"  Adding RUN_METADATA to Cell {config_idx}")
        metadata_block = [
            "\n",
            "# ── Run metadata ──────────────────────────────────────────────────────\n",
            "import hashlib, platform, datetime\n",
            "RUN_METADATA = {\n",
            "    \"run_date\":     datetime.datetime.now().isoformat(),\n",
            "    \"python\":       platform.python_version(),\n",
            "    \"seed\":         RANDOM_SEED,\n",
            "    \"window_months\": WINDOW_MONTHS,\n",
            "    \"stride_months\": STRIDE_MONTHS,\n",
            "    \"xsi_alert\":    XSI_ALERT_THRESHOLD,\n",
            "    \"xsi_critical\": XSI_CRITICAL_THRESHOLD,\n",
            "    \"top_k\":        TOP_K_FEATURES,\n",
            "}\n",
            "print(f\"Run metadata: {json.dumps(RUN_METADATA, indent=2)}\")\n",
        ]
        cells[config_idx]['source'] = src_lines + metadata_block

# ── 1.7: Clear all cell outputs ──
cleared = 0
for c in cells:
    if c.get('outputs'):
        c['outputs'] = []
        cleared += 1
    if 'execution_count' in c:
        c['execution_count'] = None
print(f"  Cleared outputs from {cleared} cells")

# ── 1.5b: Add logging setup to imports cell ──
if imports_idx is not None:
    src_lines = cells[imports_idx]['source']
    full_src = ''.join(src_lines)
    if 'import logging' not in full_src:
        print(f"  Adding logging setup to Cell {imports_idx}")
        logging_block = [
            "\n",
            "# ── Logging ───────────────────────────────────────────────────\n",
            "import logging\n",
            "logging.basicConfig(\n",
            "    level=logging.INFO,\n",
            "    format='%(asctime)s | %(levelname)s | %(message)s',\n",
            "    datefmt='%H:%M:%S'\n",
            ")\n",
            "log = logging.getLogger('xdrift')\n",
        ]
        cells[imports_idx]['source'] = cells[imports_idx]['source'] + logging_block

print(f"\nFinal: {len(cells)} cells")

# Save
with open('xai-jackfruit.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✓ Phase 1 complete — notebook saved.")
