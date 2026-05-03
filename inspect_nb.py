import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('xai-jackfruit.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Check for corrupted JS/minified output
js_artifacts = []
for i, c in enumerate(cells):
    # Check source
    src = ''.join(c.get('source', []))
    if '!0}},e.createElement' in src or '!==o.length&&e.push' in src:
        js_artifacts.append(f"Cell {i} (source): found corrupted JS")
    
    # Check outputs
    for j, out in enumerate(c.get('outputs', [])):
        out_text = json.dumps(out)
        if '!0}},e.createElement' in out_text or '!==o.length&&e.push' in out_text:
            js_artifacts.append(f"Cell {i}, Output {j}: found corrupted JS (len={len(out_text)})")
        elif len(out_text) > 50000:
            js_artifacts.append(f"Cell {i}, Output {j}: LARGE output ({len(out_text)} chars)")

# Check output sizes
print("=== OUTPUT SIZE ANALYSIS ===")
for i, c in enumerate(cells):
    outputs = c.get('outputs', [])
    if outputs:
        total = sum(len(json.dumps(o)) for o in outputs)
        print(f"Cell {i}: {len(outputs)} outputs, total {total} chars")

print("\n=== JS ARTIFACT SCAN ===")
if js_artifacts:
    for a in js_artifacts:
        print(a)
else:
    print("No corrupted JS artifacts found in sources or outputs.")

# Check for random_state usage
print("\n=== RANDOM STATE AUDIT ===")
for i, c in enumerate(cells):
    src = ''.join(c.get('source', []))
    if 'random_state' in src or 'random_seed' in src.lower() or 'np.random.seed' in src:
        lines = [l.strip() for l in src.split('\n') if 'random' in l.lower() and 'seed' in l.lower() or 'random_state' in l]
        for l in lines[:3]:
            print(f"Cell {i}: {l[:100]}")

# Check figure saving
print("\n=== FIGURE SAVE AUDIT ===")
for i, c in enumerate(cells):
    src = ''.join(c.get('source', []))
    if 'plt.savefig' in src:
        lines = [l.strip() for l in src.split('\n') if 'savefig' in l]
        for l in lines:
            print(f"Cell {i}: {l[:120]}")
    if 'plt.show()' in src and 'plt.savefig' not in src:
        print(f"Cell {i}: plt.show() WITHOUT plt.savefig()!")
