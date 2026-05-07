#!/usr/bin/env python3
import argparse
import json
from copy import deepcopy
from pathlib import Path

BENCH_N200 = ["textvqa", "ocrvqa", "gqa", "chartqa", "ocrbench"]
BENCH_N1000 = ["textvqa", "ocrvqa", "gqa", "chartqa", "ocrbench"]

ABLS = {
    "frequency_only": {
        "v91_w_freq": 1.0,
        "v91_w_format": 0.0,
        "v91_w_baseline_bias": 0.0,
        "v91_w_length_risk": 0.0,
        "v91_decision_mode": "scored",
    },
    "no_format": {
        "v91_w_format": 0.0,
        "v91_decision_mode": "scored",
    },
    "no_base_bias": {
        "v91_w_baseline_bias": 0.0,
        "v91_decision_mode": "scored",
    },
    "no_length_risk": {
        "v91_w_length_risk": 0.0,
        "v91_decision_mode": "scored",
    },
    "no_answer_space": {
        "v91_answer_space_override": "unknown",
        "v91_decision_mode": "scored",
    },
    "majority_vote": {
        "v91_decision_mode": "majority_vote",
    },
    "base_only": {
        "v91_decision_mode": "base_only",
    },
    "first_view": {
        "v91_decision_mode": "first_view",
    },
}


def main():
    ap = argparse.ArgumentParser(description="Generate V91-NoCF ablation config JSONs.")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()

    root = Path(args.repo_root)
    cfg_in = root / "benchmark_configs"
    cfg_out = root / "paper_neurips2026_artifacts" / "configs" / "ablation_configs"
    cfg_out.mkdir(parents=True, exist_ok=True)

    for bench in sorted(set(BENCH_N200 + BENCH_N1000)):
        src = cfg_in / f"test_config_smolvlm2_v91_nocf_{bench}.json"
        if not src.exists():
            continue
        base = json.loads(src.read_text())
        model_key = next(iter(base["model"].keys()))

        for abl_name, patch in ABLS.items():
            c = deepcopy(base)
            old_model = c["model"][model_key]
            new_key = f"V91NoCFAbl_{abl_name}_SmolVLM2_2B"
            c["model"] = {new_key: old_model}
            c["model"][new_key]["class"] = "V91NoCF_SmolVLM2_2B"
            for k, v in patch.items():
                c["model"][new_key][k] = v

            out_name = f"test_config_smolvlm2_v91_nocf_ablation_{abl_name}_{bench}.json"
            (cfg_out / out_name).write_text(json.dumps(c, indent=4) + "\n")

    print(cfg_out)


if __name__ == "__main__":
    main()
