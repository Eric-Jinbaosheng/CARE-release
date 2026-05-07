#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(dirname "$ROOT")"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${PARENT}/github_release_anonymous_${STAMP}"

rsync -a --delete \
  --exclude '.git' \
  --exclude '.runtime_cache' \
  --exclude 'benchmark_results' \
  --exclude 'logs' \
  --exclude 'paper_figures/cases_raw' \
  --exclude 'paper_figures/benchmark_aug_cf_pack' \
  --exclude 'paper_figures/cf_effective_cases' \
  --exclude 'paper_figures/case_grid_materials/images' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '*.swp' \
  "$ROOT/" "$OUT/"

# redact identity/path strings in text-like files
find "$OUT" -type f \( -name '*.md' -o -name '*.tex' -o -name '*.csv' -o -name '*.json' -o -name '*.jsonl' -o -name '*.yaml' -o -name '*.yml' -o -name '*.sh' -o -name '*.py' -o -name '*.txt' \) | while read -r f; do
  sed -i 's#<ANON_ROOT>#<ANON_ROOT>#g' "$f"
  sed -i 's#<ANON_USER>#<ANON_USER>#g' "$f"
  sed -i 's#<ANON_ACCOUNT>[A-Za-z0-9_+-]*#<ANON_ACCOUNT>#g' "$f"
  sed -i -E 's#[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}#<ANON_EMAIL>#g' "$f"
done

echo "[DONE] Anonymous package: $OUT"
