#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


SETTINGS = [
    ("default", 2.0, 1.0, 0.4, 0.5),
    ("wsup1p0", 1.0, 1.0, 0.4, 0.5),
    ("wsup1p5", 1.5, 1.0, 0.4, 0.5),
    ("wsup2p5", 2.5, 1.0, 0.4, 0.5),
    ("wsup3p0", 3.0, 1.0, 0.4, 0.5),
    ("wvalid0p25", 2.0, 0.25, 0.4, 0.5),
    ("wvalid0p5", 2.0, 0.5, 0.4, 0.5),
    ("wvalid1p5", 2.0, 1.5, 0.4, 0.5),
    ("wvalid2p0", 2.0, 2.0, 0.4, 0.5),
]


BASES = {
    "textvqa": "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_regen_textvqa/test_config_smolvlm2_v91_nocf_regen_textvqa.json",
    "chartqa": "benchmark_results/n_samples_1000/test_config_smolvlm2_v91_nocf_regen_chartqa/test_config_smolvlm2_v91_nocf_regen_chartqa.json",
}


def main():
    ap = argparse.ArgumentParser(description="Generate GroupA true-pipeline sensitivity configs.")
    ap.add_argument("--repo-root", default="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean")
    ap.add_argument("--out-dir", default="paper_neurips2026_artifacts/configs/sensitivity_groupA_truepipeline")
    args = ap.parse_args()

    root = Path(args.repo_root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    created = []
    for bench, base_rel in BASES.items():
        base_path = root / base_rel
        obj = json.loads(base_path.read_text())
        model_key = next(iter(obj["model"].keys()))
        model_obj = obj["model"][model_key]
        for tag, ws, wv, wb, wr in SETTINGS:
            cfg = json.loads(json.dumps(obj))
            cfg["model"][model_key]["v91_w_freq"] = ws
            cfg["model"][model_key]["v91_w_format"] = wv
            cfg["model"][model_key]["v91_w_baseline_bias"] = wb
            cfg["model"][model_key]["v91_w_length_risk"] = wr
            stem = f"test_config_smolvlm2_v91_nocf_sens_{tag}_{bench}"
            out_path = out_dir / f"{stem}.json"
            out_path.write_text(json.dumps(cfg, indent=4) + "\n")
            created.append((bench, tag, stem, out_path))

    submit = root / "paper_neurips2026_artifacts/jobs/sensitivity_asca_weights/submit_groupA_truepipeline_2bench.sh"
    submit.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        'ROOT="<ANON_ROOT>/peking/smolvlm2_paper/ets_clean"',
        "cd \"$ROOT\"",
        'ACC="${ACC:-<ANON_ACCOUNT>}"',
        'PART="${PART:-l40s_public}"',
        "",
    ]
    for bench, tag, stem, out_path in created:
        cfg_rel = str(out_path.relative_to(root))
        cache = f"<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/.runtime_cache/{stem}_n1000"
        lines += [
            f'echo "[SUBMIT] {stem}"',
            f'ACC=\"$ACC\" PART=\"$PART\" scripts/sbatch_clean.sh "{cfg_rel}" SUBSET_LEN=1000,CACHE_PATH={cache}',
            "",
        ]
    submit.write_text("\n".join(lines) + "\n")
    submit.chmod(0o755)

    print(f"[DONE] configs: {out_dir}")
    print(f"[DONE] submit script: {submit}")


if __name__ == "__main__":
    main()

