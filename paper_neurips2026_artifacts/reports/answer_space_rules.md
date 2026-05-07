# Answer-Space Rules

Source: `<ANON_ROOT>/peking/smolvlm2_paper/ets_clean/vlmeval/vlm/tta/tta_v91_aggregator.py`

## 1) Normalization
- `normalize_answer`: lowercase + punctuation removal + strip.

## 2) Classifier spaces
- yes_no
- multiple_choice
- numeric
- ocr_text_short
- open_entity
- chart_or_diagram
- caption_like
- unknown

## 3) Trigger heuristics
- OCR intent keywords and alphanumeric-short candidates route to `ocr_text_short`.
- chart keywords route to `chart_or_diagram`.
- caption-like prompts or long candidates route to `caption_like`.
- yes/no and MCQ regex/option parsing are handled explicitly.
- numeric/date regex used for `numeric`.

## 4) Candidate validity and risks
- `fmt_validity(a)` depends on answer-space (`_format_validity`).
- `length_risk(a)` penalizes suspicious long candidates in short-answer spaces (`_length_risk`).

## 5) General scoring formula
- `score_general(a)=2.0*view_freq(a)+1.0*fmt_validity(a)+0.4*1[a==base]-0.5*length_risk(a)`

## 6) Observed answer-space distribution (n=1000, V91-NoCF diagnostics)

- ocrbench: {}
- gqa: {}
- textvqa: {}
- chartqa: {}
- ocrvqa: {}
- ai2d: {}
- mme_rw: {}
- coco: {}
- amber: {}
