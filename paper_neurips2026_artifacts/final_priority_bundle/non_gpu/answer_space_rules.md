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

- ocrbench: {'ocr_text_short': 327, 'unknown': 64, 'numeric': 209, 'open_entity': 272, 'chart_or_diagram': 66, 'caption_like': 53, 'multiple_choice': 9}
- gqa: {'yes_no': 348, 'unknown': 478, 'open_entity': 104, 'chart_or_diagram': 40, 'caption_like': 28, 'numeric': 2}
- textvqa: {'ocr_text_short': 251, 'open_entity': 504, 'numeric': 130, 'yes_no': 68, 'chart_or_diagram': 24, 'multiple_choice': 7, 'unknown': 14, 'caption_like': 2}
- chartqa: {'chart_or_diagram': 463, 'open_entity': 100, 'numeric': 399, 'yes_no': 30, 'unknown': 6, 'ocr_text_short': 2}
- ocrvqa: {'open_entity': 389, 'yes_no': 392, 'numeric': 5, 'unknown': 115, 'ocr_text_short': 69, 'caption_like': 30}
- ai2d: {'multiple_choice': 1000}
- mme_rw: {'multiple_choice': 1000}
- coco: {'caption_like': 1000}
- amber: {'caption_like': 290, 'unknown': 395, 'open_entity': 2, 'yes_no': 303, 'chart_or_diagram': 10}
