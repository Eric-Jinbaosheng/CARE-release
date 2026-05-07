#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT="$(dirname "$ROOT")"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${PARENT}/github_release_clean_${STAMP}"

mkdir -p "$OUT"

copy_if_exists() {
  local p="$1"
  if [[ -e "${ROOT}/${p}" ]]; then
    rsync -a "${ROOT}/${p}" "${OUT}/"
  fi
}

# Core code + paper artifacts
copy_if_exists ".github"
copy_if_exists "assets"
copy_if_exists "benchmark_configs"
copy_if_exists "docs"
copy_if_exists "paper_neurips2026_artifacts"
copy_if_exists "requirements"
copy_if_exists "scripts"
copy_if_exists "vlmeval"
copy_if_exists "README.md"
copy_if_exists "README_VLMEVALKIT.md"
copy_if_exists "requirements.txt"
copy_if_exists "setup.py"
copy_if_exists "run.py"
copy_if_exists ".gitignore"
copy_if_exists ".pre-commit-config.yaml"

# Remove local/cluster heavy artifacts from copied tree
rm -rf \
  "${OUT}/.runtime_cache" \
  "${OUT}/benchmark_results" \
  "${OUT}/logs" \
  "${OUT}/paper_figures/cases_raw" \
  "${OUT}/paper_figures/benchmark_aug_cf_pack" \
  "${OUT}/paper_figures/cf_effective_cases" \
  "${OUT}/paper_figures/case_grid_materials/images" \
  "${OUT}/unsloth_compiled_cache" \
  "${OUT}/scripts/__pycache__" \
  "${OUT}/vlmeval/__pycache__" \
  "${OUT}/.git"

# Remove transient files
find "${OUT}" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${OUT}" -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.swp" \) -delete

echo "[DONE] Clean export created at:"
echo "${OUT}"
echo
echo "Next:"
echo "  cd ${OUT}"
echo "  git init"
echo "  git add ."
echo "  git status"
