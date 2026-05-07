# Second Backbone Feasibility

## Status
- **Feasible and already supported in repo.**
- Existing wrappers found:
  - `TTAugAdapter_Ovis2` (`vlmeval/vlm/tta/tta_ovis.py`)
  - `TTAugAdapter_InternVLChat` (`vlmeval/vlm/tta/tta_internvl_chat.py`)

## Implemented for this artifact pack
- Added runnable Ovis2-1B deterministic TTAug configs (n=200 smoke target):
  - `paper_neurips2026_artifacts/configs/second_backbone_configs/test_config_ovis2_1b_ttaug_det_textvqa.json`
  - `paper_neurips2026_artifacts/configs/second_backbone_configs/test_config_ovis2_1b_ttaug_det_ocrvqa.json`
  - `paper_neurips2026_artifacts/configs/second_backbone_configs/test_config_ovis2_1b_ttaug_det_gqa.json`
  - `paper_neurips2026_artifacts/configs/second_backbone_configs/test_config_ovis2_1b_ttaug_det_ocrbench.json`
- Added submit script:
  - `paper_neurips2026_artifacts/jobs/submit_second_backbone_n200.sh`

## Notes
- This is a minimal backbone robustness check (not full retuning of V91 rules).
- If these runs complete in time, include them as cross-backbone sanity evidence in appendix.
