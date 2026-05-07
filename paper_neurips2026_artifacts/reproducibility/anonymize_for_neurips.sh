#!/usr/bin/env bash
set -euo pipefail

ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"
SRC="$ROOT/paper_neurips2026_artifacts"
DST="$SRC/reproducibility/anonymous_package"

mkdir -p "$DST"
rm -rf "$DST"/*

copy_safe() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  cp -r "$src" "$dst"
}

copy_safe "$SRC/scripts" "$DST/scripts"
copy_safe "$SRC/configs" "$DST/configs"
copy_safe "$SRC/jobs" "$DST/jobs"
copy_safe "$SRC/tables" "$DST/tables"
copy_safe "$SRC/figures" "$DST/figures"
copy_safe "$SRC/reports" "$DST/reports"
copy_safe "$SRC/supplement" "$DST/supplement"
copy_safe "$SRC/reproducibility/README.md" "$DST/README.md"

# Optional minimal index copy
if [[ -f "$ROOT/logs/experiment_metric_index_20260429.json" ]]; then
  mkdir -p "$DST/minimal_index"
  cp "$ROOT/logs/experiment_metric_index_20260429.json" "$DST/minimal_index/"
fi

# Redaction pass (non-destructive, only in anonymous_package)
find "$DST" -type f \( -name "*.md" -o -name "*.tex" -o -name "*.csv" -o -name "*.json" -o -name "*.sh" -o -name "*.py" \) | while read -r f; do
  sed -i "s#<ANON_ROOT>#<ANON_ROOT>#g" "$f"
  sed -i "s#<ANON_USER>#<ANON_USER>#g" "$f"
  sed -i "s#<ANON_ACCOUNT>#<ANON_ACCOUNT>#g" "$f"
  sed -i "s#[A-Za-z0-9._%+-]\+@[A-Za-z0-9.-]\+\.[A-Za-z]\{2,\}#<ANON_EMAIL>#g" "$f"
done

# Secret scan
FOUND=0
for pat in "sk-" "api_key" "OPENAI_API_KEY" "hf_" "token"; do
  if rg -n "$pat" "$DST" >/dev/null 2>&1; then
    echo "[WARN] Potential secret-like token pattern '$pat' found in anonymous package"
    FOUND=1
  fi
done

if [[ "$FOUND" -ne 0 ]]; then
  echo "Anonymization detected potential sensitive patterns. Please inspect package before release."
  exit 1
fi

echo "Anonymous package prepared at: $DST"
