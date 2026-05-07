#!/usr/bin/env python3
import argparse
import json
from copy import deepcopy
from pathlib import Path

BENCH = ["textvqa", "ocrvqa", "gqa", "ocrbench"]


def main():
    ap = argparse.ArgumentParser(description="Generate second-backbone (Ovis2-1B) n=200 configs.")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    args = ap.parse_args()

    root = Path(args.repo_root)
    cfg_in = root / "benchmark_configs"
    cfg_out = root / "paper_neurips2026_artifacts" / "configs" / "second_backbone_configs"
    cfg_out.mkdir(parents=True, exist_ok=True)

    created = []
    for b in BENCH:
        src = cfg_in / f"test_config_smolvlm2_paper_ttaug_classical_deterministic_{b}.json"
        if not src.exists():
            continue
        obj = json.loads(src.read_text())
        old_key = next(iter(obj["model"].keys()))
        old = deepcopy(obj["model"][old_key])
        new_key = "TTAugAdapter_Ovis2_1B_deterministic"
        old["class"] = "TTAugAdapter_Ovis2"
        old["model_args"] = {"model_path": "AIDC-AI/Ovis2-1B"}
        old["number_of_versions"] = 8
        old["token_selection_aggregation_method"] = "average"
        old["save_inputs_for_debugging"] = False
        old["handle_oom"] = False
        obj["model"] = {new_key: old}

        out = cfg_out / f"test_config_ovis2_1b_ttaug_det_{b}.json"
        out.write_text(json.dumps(obj, indent=4) + "\n")
        created.append(out)

    print("\n".join(str(p) for p in created))


if __name__ == "__main__":
    main()
