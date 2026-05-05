# M9A ROW ARCHETYPE PROBE

## Date
2026 (exact date from git log or experiment run)

## Goal
M9a: Row-archetype redundancy probe.

## Configuration
iters=4

## Method / What was tested
Measures whether rows of the encoded `ids` tensor are clusterable enough
to support an "archetype + sparse delta" architecture (Idea A).

For each probed layer:
  1. Load bf16 weight, encode to ids[N,K].
  2. Pairwise Hamming similarity for a random sample of rows (sanity).
  3. Mini-batch k-means on rows in Hamming space, G in {32, 128, 512}.
  4. For each G, report:
       - mean / median Hamming distance row -> nearest centroid
       - delta density needed for EXACT reconstruction
       - implied VRAM for ids if we keep dense delta
       - implied VRAM if we keep top-D% delta entries (D=5,10,20)

## Result
Encode test.
Likely negative result

## Artifacts
- `experiments/m9a_row_archetype_probe.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.