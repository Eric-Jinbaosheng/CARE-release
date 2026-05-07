"""
v91 — Evidence-Aware TTAug Aggregator for SmolVLM2-2.2B.

Goal: turn ECCTS from "counterfactual everywhere" into a general
TTAug-style test-time scaling method where candidate reranking is the
backbone and counterfactual evidence is an OPTIONAL expert.

Two subclasses:
  - V91NoCF_SmolVLM2_2B : general candidate reranker only.
  - V91CF_SmolVLM2_2B   : v91_no_cf + selective counterfactual expert
                          (beta routed by answer-space + sample features).

Both inherit deterministic seeding + bit-identical disabled bypass
from TTAugAdapter_SmolVLM2. When `v91_enabled=False`, the helper
returns the parent's answer EXACTLY (no rerank, no parsing change).
"""

import json
import math
import os
import re
import string
import types
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from .tta_smolvlm import TTAugAdapter_SmolVLM2, apply_deterministic_seed


# ---------------------------------------------------------------------------
# Module-level: text utils, answer-space classifier, candidate features
# ---------------------------------------------------------------------------


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s):
    if s is None:
        return ""
    return str(s).strip().lower().translate(_PUNCT_TABLE).strip()


_NUMERIC_RE = re.compile(
    r"^[\$£€¥]?[+-]?\d+(?:[\.,]\d+)*(?:%|°|usd|gb|kg|mg|cm|mm|km|kmh|hp|ml|l|kw|kwh)?\.?$",
    flags=re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b(19|20)\d{2}\b|\b\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}\b")
_YES_NO = {"yes", "no", "true", "false", "yeah", "nope"}
_MCQ_LETTER_RE = re.compile(r"^[abcdefgh][\.\):]?$", flags=re.IGNORECASE)


def is_numeric(s):
    s = normalize_answer(s)
    if not s:
        return False
    if _NUMERIC_RE.match(s):
        return True
    return bool(_DATE_RE.search(s))


def is_yes_no(s):
    return normalize_answer(s) in _YES_NO


def is_mcq_letter(s):
    s = (s or "").strip()
    return bool(_MCQ_LETTER_RE.match(s))


def is_alphanumeric_short(s):
    n = normalize_answer(s)
    if not n or len(n) > 32:
        return False
    return bool(re.match(r"^[a-z0-9 \-_/\.]+$", n))


def answer_length(s):
    return len(normalize_answer(s).split())


def has_visual_text_intent(question):
    q = (question or "").lower()
    if any(p in q for p in _OCR_INTENT_PATTERNS):
        return True
    extra = (
        "text", "written", "write", "word", "letters", "spelling",
        "sign", "label", "logo", "brand", "name", "title",
        "license plate", "plate", "phone number", "address",
        "serial", "model", "code", "barcode", "price tag",
    )
    return any(k in q for k in extra)


_OCR_INTENT_PATTERNS = [
    "what does the sign", "what does it say", "what is written",
    "what word is", "read the text", "read the label", "read the sign",
    "what is the label", "what is the title", "what is the brand",
    "what brand", "what company", "what product", "what book",
    "what author", "what number is", "what number on", "what time is",
    "license plate", "what does the text", "what is the name",
]


_CHART_INTENT_PATTERNS = [
    "chart", "graph", "diagram", "table", "axis", "bar", "line",
    "legend", "trend", "percentage", "pie", "scatter", "histogram",
    "what value", "what percent", "highest", "lowest", "y axis", "x axis",
]


_CAPTION_INTENT_PATTERNS = [
    "describe", "caption", "what is happening", "what is going on",
    "summarize the image",
]


def _question_text(message):
    parts = []
    if isinstance(message, list):
        for msg in message:
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "text" and msg.get("value"):
                parts.append(str(msg["value"]))
            content = msg.get("content")
            if isinstance(content, list):
                for it in content:
                    if isinstance(it, dict) and it.get("type") == "text" and it.get("value"):
                        parts.append(str(it["value"]))
    return "\n".join(parts).strip()


_OPTION_RE = re.compile(r"\b([A-H])[\.\):]\s+([^\n]+)")


def extract_options(question):
    """Return list of (letter, text) parsed from typical MCQ option blocks."""
    if not question:
        return []
    options = []
    for m in _OPTION_RE.finditer(question):
        letter = m.group(1).upper()
        text = m.group(2).strip()
        # Trim at the next option boundary if present
        text = re.split(r"\s+[A-H][\.\):]\s+", text)[0].strip()
        options.append((letter, text))
    return options


def classify_answer_space(question, candidates, options=None):
    """Heuristic answer-space classifier (no benchmark name).
    Returns one of:
        yes_no, multiple_choice, numeric, ocr_text_short,
        open_entity, chart_or_diagram, caption_like, unknown
    """
    q = (question or "").lower()
    cand_norms = [normalize_answer(c) for c in candidates if c]
    nonempty = [c for c in cand_norms if c]
    yn_count = sum(1 for c in nonempty if c in _YES_NO)
    yn_frac = yn_count / max(1, len(nonempty))

    # MCQ detection: explicit options present, OR candidates dominated by A-H letters
    if options:
        return "multiple_choice"
    mcq_count = sum(1 for c in nonempty if is_mcq_letter(c))
    if mcq_count / max(1, len(nonempty)) >= 0.5 and len(nonempty) >= 2:
        return "multiple_choice"

    # yes/no
    if yn_frac >= 0.5 or any(p in q for p in [
        "is this", "is there", "are there", "is it", "do you see",
        "does the", "yes/no", "true or false", "is the",
    ]):
        if yn_frac >= 0.5:
            return "yes_no"

    # caption first (long output intent)
    if any(p in q for p in _CAPTION_INTENT_PATTERNS):
        return "caption_like"
    avg_len = sum(answer_length(c) for c in candidates) / max(1, len(candidates))
    if avg_len >= 8:
        return "caption_like"

    # chart/diagram
    if any(p in q for p in _CHART_INTENT_PATTERNS):
        return "chart_or_diagram"

    # OCR / text-short
    if any(p in q for p in _OCR_INTENT_PATTERNS):
        if all(is_alphanumeric_short(c) for c in candidates if c):
            return "ocr_text_short"

    # numeric
    num_frac = sum(1 for c in nonempty if is_numeric(c)) / max(1, len(nonempty))
    if num_frac >= 0.6:
        return "numeric"

    if all(is_alphanumeric_short(c) for c in candidates if c) and avg_len <= 4:
        return "open_entity"

    return "unknown"


# ---------------------------------------------------------------------------
# Per-candidate feature record
# ---------------------------------------------------------------------------


def compute_candidate_features(view_answers, base_answer, options=None):
    """Aggregate per-candidate features from N per-view answers.

    Returns: dict[norm_answer] -> dict(features)
    """
    raw_to_norm = {a: normalize_answer(a) for a in view_answers if a}
    norm_to_raws = {}
    for a in view_answers:
        n = normalize_answer(a)
        if not n:
            continue
        norm_to_raws.setdefault(n, []).append(a)

    base_norm = normalize_answer(base_answer)
    if base_norm and base_norm not in norm_to_raws:
        norm_to_raws[base_norm] = [base_answer]

    # Add MCQ option text as candidates (so reranker can pick an option that
    # never showed up in N=8 views).
    if options:
        for letter, text in options:
            n = normalize_answer(text)
            if n and n not in norm_to_raws:
                norm_to_raws[n] = [text]
            ln = normalize_answer(letter)
            if ln and ln not in norm_to_raws:
                norm_to_raws[ln] = [letter]

    n_views = max(1, len([a for a in view_answers if a]))
    feats = {}
    for n_ans, raws in norm_to_raws.items():
        view_count = len([a for a in view_answers if normalize_answer(a) == n_ans])
        feats[n_ans] = {
            "raw": raws[0],
            "all_raw": raws,
            "view_freq": view_count / n_views,
            "view_count": view_count,
            "is_base": (n_ans == base_norm),
            "is_numeric": is_numeric(n_ans),
            "is_yes_no": n_ans in _YES_NO,
            "is_mcq_letter": is_mcq_letter(raws[0]),
            "is_alnum_short": is_alphanumeric_short(n_ans),
            "length_words": answer_length(n_ans),
            "from_options": (options is not None and any(
                normalize_answer(t) == n_ans for _, t in options
            )),
            "is_substring_of_base": (
                base_norm and n_ans != base_norm and n_ans in base_norm
            ),
            "is_superstring_of_base": (
                base_norm and n_ans != base_norm and base_norm in n_ans
            ),
        }
    return feats


# ---------------------------------------------------------------------------
# General candidate scoring (v91_no_cf)
# ---------------------------------------------------------------------------


def _format_validity(n_ans, feats, answer_space):
    """Returns a 0..1 multiplicative validity score for candidate."""
    f = feats[n_ans]
    if answer_space == "yes_no":
        return 1.0 if f["is_yes_no"] else 0.0
    if answer_space == "multiple_choice":
        return 1.0 if (f["is_mcq_letter"] or f["from_options"]) else 0.2
    if answer_space == "numeric":
        return 1.0 if f["is_numeric"] else 0.4
    if answer_space == "ocr_text_short":
        return 1.0 if f["is_alnum_short"] else 0.3
    if answer_space == "open_entity":
        return 1.0 if f["length_words"] <= 5 else 0.6
    if answer_space == "chart_or_diagram":
        return 1.0 if (f["is_numeric"] or f["length_words"] <= 6) else 0.5
    if answer_space == "caption_like":
        return 1.0 if f["length_words"] >= 4 else 0.3
    return 1.0


def _length_risk(n_ans, feats, answer_space, base_len_words):
    f = feats[n_ans]
    if answer_space in ("caption_like",):
        return 0.0
    if answer_space in ("yes_no", "multiple_choice", "numeric"):
        return 0.5 if f["length_words"] > 4 else 0.0
    if answer_space in ("ocr_text_short", "open_entity"):
        if base_len_words and f["length_words"] > base_len_words * 2:
            return 0.5
    return 0.0


def score_candidates_general(features, answer_space, base_norm,
                             *, w_freq=2.0, w_format=1.0, w_baseline_bias=0.4,
                             w_length_risk=0.5):
    """Hand-tuned transparent score:
    score = w_freq * view_freq + w_format * fmt_validity
            + w_baseline_bias * is_base
            - w_length_risk * length_risk
    """
    base_len = features.get(base_norm, {}).get("length_words", 0)
    scored = {}
    for n_ans, f in features.items():
        fv = _format_validity(n_ans, features, answer_space)
        lr = _length_risk(n_ans, features, answer_space, base_len)
        s = (
            w_freq * f["view_freq"]
            + w_format * fv
            + (w_baseline_bias if f["is_base"] else 0.0)
            - w_length_risk * lr
        )
        scored[n_ans] = {
            "score_general": s,
            "fmt_validity": fv,
            "length_risk": lr,
            **f,
        }
    return scored


# ---------------------------------------------------------------------------
# Counterfactual expert (v91_cf only)
# ---------------------------------------------------------------------------


def cf_score_for_candidate(n_ans, rel_norm, ctrl_norm, *, tau=0.0):
    """Score_CF(a) > 0 only if relevant-drop changed answer AND control-drop didn't."""
    rel_match = 1 if n_ans == rel_norm else 0
    ctrl_match = 1 if n_ans == ctrl_norm else 0
    gap = ctrl_match - rel_match  # in {-1, 0, +1}
    return max(0.0, gap - tau)


def beta_for_sample(answer_space, *, entropy, margin, max_grounding,
                    loc_conf, n_candidates, base_freq):
    """Beta routing — based on observable sample features, NO benchmark name."""
    # Sample-feature flags
    high_uncertainty = (margin < 0.5) or (entropy > 0.7)
    stable_pool = (n_candidates <= 4 and base_freq >= 0.5)
    has_grounding = (max_grounding >= 1)
    has_loc = (loc_conf is None) or (loc_conf >= 0.55)

    if not has_grounding or not has_loc:
        return 0.0

    if answer_space in ("chart_or_diagram", "caption_like"):
        return 0.0
    if answer_space == "yes_no":
        return 0.0  # constrained — CF does not help binary answers
    if answer_space == "multiple_choice":
        return 0.0  # constrained — CF does not help MCQ
    if answer_space == "numeric":
        # Numeric needs strong evidence; default low
        if high_uncertainty and stable_pool:
            return 0.3
        return 0.0
    if answer_space == "ocr_text_short":
        if high_uncertainty and stable_pool:
            return 1.0
        if stable_pool:
            return 0.6
        return 0.3
    if answer_space == "open_entity":
        if high_uncertainty and stable_pool:
            return 0.5
        return 0.2
    return 0.2  # unknown


# ---------------------------------------------------------------------------
# V91 adapter base class
# ---------------------------------------------------------------------------


class V91Adapter_SmolVLM2(TTAugAdapter_SmolVLM2):
    """Evidence-aware TTAug aggregator. Subclasses set `cf_enabled`."""

    cf_enabled: bool = False  # subclass override
    cf3_enabled: bool = False  # subclass override

    _V91_DEFAULTS = {
        "v91_enabled": True,
        "deterministic_seeding": True,
        "deterministic_seeding_log": True,
        # Optional NoCF ablation switches
        # - scored: answer-space-aware weighted reranker (default)
        # - majority_vote: max view frequency (ties prefer base, then shorter)
        # - base_only: always return non-aug base answer
        # - first_view: always return first deterministic view answer
        "v91_decision_mode": "scored",
        # Optional override of answer-space classification for ablations.
        # e.g. set "unknown" to disable answer-space-aware routing behavior.
        "v91_answer_space_override": None,
        # CF (only matters if cf_enabled=True on subclass)
        "v91_cf_corruption_radius": 7.0,
        "v91_cf_center_ratio": 0.6,
        "v91_cf_grounding_tau": 0.0,
        "v91_cf_logprob_tau": 0.15,   # CF gate on normalized CF margin
        "v91_cf_score_clip": 2.5,     # z-score clipping for CF(a)
        "v91_cf_min_view_support": 2, # high-precision gate
        "v91_cf_low_quality_beta_scale": 0.0,  # disable low-quality mask CF
        "v91_cf_grid_enable": True,
        "v91_cf_grid_size": 4,
        "v91_cf_grid_min_candidates": 2,
        "v91_cf_grid_margin_thresh": 0.5,
        "v91_cf_grid_entropy_thresh": 0.7,
        "v91_cf_ctrl_nonempty_std": 8.0,
        "v91_cf_rel_drop_min": 0.02,
        # CF3 verifier policy (NoCF winner is primary; CF only switches)
        "v91_cf3_enable_candidate_grid": True,
        "v91_cf3_general_gap_tau": 0.20,
        "v91_cf3_switch_spaces": ("ocr_text_short", "open_entity"),
        # CF3 diagnostic modes:
        #   strict (default), score_only, no_quality_gate,
        #   force_grid, force_grid_no_quality_gate, force_switch_analysis
        "v91_cf3_mode": "strict",
        "v91_cf3_force_grid": False,
        "v91_cf3_disable_quality_gate": False,
        "v91_cf3_score_only": False,
        "v91_cf3_force_switch_analysis": False,
        # CF3 routed gate (benchmark-agnostic, sample-level)
        "v91_cf3_routed_entropy_min": 0.70,
        "v91_cf3_routed_margin_max": 0.50,
        "v91_cf3_routed_max_words": 4,
        "v91_cf3_routed_max_chars": 24,
        "v91_cf3_routed_min_textlike_candidates": 2,
        "v91_force_loc_conf": 1.0,  # placeholder until SmolVLM2 localization
        # Diagnostics
        "v91_diag_per_sample": True,
        "v91_diag_dir": None,
        # Scoring weights (transparent hand-tuned)
        "v91_w_freq": 2.0,
        "v91_w_format": 1.0,
        "v91_w_baseline_bias": 0.4,
        "v91_w_length_risk": 0.5,
    }

    def __init__(self, model_args, text_aug_args, image_aug_args, **adapter_args):
        merged = dict(self._V91_DEFAULTS)
        merged.update(adapter_args)
        merged.setdefault("cf_enabled", self.cf_enabled)
        super().__init__(model_args, text_aug_args, image_aug_args, **merged)

        if self.v91_diag_dir is None:
            cache = os.environ.get("CACHE_PATH", ".")
            self.v91_diag_dir = os.path.join(cache, "diagnostics")
        os.makedirs(self.v91_diag_dir, exist_ok=True)
        if self.cf_enabled and self.cf3_enabled:
            suffix = "v91cf3"
        elif self.cf_enabled:
            suffix = "v91cf"
        else:
            suffix = "v91nocf"
        self._v91_diag_path = os.path.join(
            self.v91_diag_dir, f"{suffix}_samples.jsonl"
        )
        self._v91_sample_counter = 0
        self._v91_original_images = None

        try:
            from transformers.generation.utils import GenerationMixin
            self._v91_original_sample = GenerationMixin._sample
        except Exception as e:
            print(f"[V91] WARN: could not capture original _sample: {e}")
            self._v91_original_sample = None

        print(
            f"[V91-INIT] cf_enabled={self.cf_enabled} v91_enabled={self.v91_enabled} "
            f"diag={self._v91_diag_path}",
            flush=True,
        )

    def _cf3_routed_space_ok(self, answer_space, question, candidate_map, margin, entropy):
        """Benchmark-agnostic CF routing gate for CF3.
        Returns (space_ok: bool, reason: str).
        """
        if answer_space in ("chart_or_diagram", "caption_like", "multiple_choice", "yes_no", "numeric"):
            return False, "answer_space_block"

        n_cands = len(candidate_map or {})
        if n_cands < 2:
            return False, "no_valid_alternative"

        text_intent = has_visual_text_intent(question)
        uncertain = (
            (float(entropy) >= float(self.v91_cf3_routed_entropy_min))
            or (float(margin) <= float(self.v91_cf3_routed_margin_max))
        )
        min_textlike = int(self.v91_cf3_routed_min_textlike_candidates)
        max_words = int(self.v91_cf3_routed_max_words)
        max_chars = int(self.v91_cf3_routed_max_chars)

        textlike = 0
        for n_ans in candidate_map.keys():
            n = normalize_answer(n_ans)
            if not n:
                continue
            if answer_length(n) > max_words:
                continue
            if len(n) > max_chars:
                continue
            if not is_alphanumeric_short(n):
                continue
            if n in _YES_NO:
                continue
            if not re.search(r"[a-z0-9]", n):
                continue
            textlike += 1

        if answer_space == "ocr_text_short":
            return True, "routed_ocr_text_short"

        if answer_space == "open_entity":
            if text_intent:
                return True, "routed_open_entity_text_intent"
            if textlike >= min_textlike and uncertain:
                return True, "routed_open_entity_uncertain_textlike"
            return False, "open_entity_not_text_grounded"

        if answer_space == "unknown":
            if text_intent and textlike >= min_textlike and uncertain:
                return True, "routed_unknown_text_intent"
            return False, "unknown_not_routed"

        return False, "answer_space_block"

    # ------------------------------------------------------------------
    # generate_inner: capture original images + deterministic seed
    # ------------------------------------------------------------------

    def generate_inner(self, message, dataset=None):
        if getattr(self, "deterministic_seeding", True):
            apply_deterministic_seed(
                dataset, message,
                log=getattr(self, "deterministic_seeding_log", True),
            )

        formatted_messages, formatted_images = self.build_prompt_cases(message, dataset)
        images = (
            [formatted_images]
            if isinstance(formatted_images, Image.Image)
            else formatted_images
        )
        self._v91_original_images = list(images)
        self._v91_question_text = _question_text(message)

        images_augmented, applied_transforms = self.image_augment(images)

        save_visual_samples_flag = os.environ.get(
            "SAVE_VISUAL_SAMPLES", "False"
        ).lower() in ("1", "true", "yes")
        if save_visual_samples_flag:
            self.save_inputs_grid_prompts(
                message, formatted_messages, images_augmented,
                applied_transforms, dataset,
            )

        HANDLE_OUT_OF_MEMORY = getattr(self, "handle_oom", True)
        if not HANDLE_OUT_OF_MEMORY:
            return self.generate_inner_helper(
                message, formatted_messages, images_augmented, dataset
            )

        max_retries = 8
        for attempt in range(max_retries):
            try:
                return self.generate_inner_helper(
                    message, formatted_messages, images_augmented, dataset
                )
            except torch.OutOfMemoryError as e:
                print(f"Attempt {attempt + 1} failed:", e)
                images_augmented = images_augmented[:max(1, len(images_augmented)//2)]
                formatted_messages = formatted_messages[:max(1, len(formatted_messages)//2)]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with exception:", e)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        print("All retries failed")
        return ""

    def generate_inner_helper(self, message, formatted_messages, images_augmented, dataset=None):
        # Always run the parent first to get the bit-identical TTAug answer.
        base_answer = super().generate_inner_helper(
            message, formatted_messages, images_augmented, dataset
        )
        if not getattr(self, "v91_enabled", True):
            # Strict bit-identical bypass.
            return base_answer

        return self._v91_rerank(
            base_answer, message, formatted_messages, images_augmented, dataset
        )

    # ------------------------------------------------------------------
    # Per-view individual greedy decoding (swap _sample temporarily)
    # ------------------------------------------------------------------

    def _decode_per_view(self, formatted_messages, images_augmented):
        if self._v91_original_sample is None:
            return []
        self.processor.tokenizer.padding_side = "left"
        try:
            inputs = self.processor(
                text=formatted_messages,
                images=images_augmented,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)
        except Exception as e:
            print(f"[V91] per-view processor failed: {e}")
            return []
        modified_sample = self.model._sample
        try:
            self.model._sample = types.MethodType(
                self._v91_original_sample, self.model
            )
            ids = self.model.generate(**inputs, **self.kwargs)
        finally:
            self.model._sample = modified_sample
        decoded = self.processor.batch_decode(
            ids[:, inputs["input_ids"].size(1):],
            skip_special_tokens=True,
        )
        return [t.strip() for t in decoded]

    def _decode_single(self, image, text_prompt):
        if self._v91_original_sample is None:
            return ""
        self.processor.tokenizer.padding_side = "left"
        cf_inputs = self.processor(
            text=[text_prompt],
            images=[[image]],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        modified_sample = self.model._sample
        try:
            self.model._sample = types.MethodType(
                self._v91_original_sample, self.model
            )
            ids = self.model.generate(**cf_inputs, **self.kwargs)
        finally:
            self.model._sample = modified_sample
        decoded = self.processor.batch_decode(
            ids[:, cf_inputs["input_ids"].size(1):],
            skip_special_tokens=True,
        )
        return decoded[0].strip()

    # ------------------------------------------------------------------
    # CF view construction (original-image-only)
    # ------------------------------------------------------------------

    def _make_relevant_drop(self, image):
        radius = float(self.v91_cf_corruption_radius)
        ratio = max(0.1, min(0.95, float(self.v91_cf_center_ratio)))
        img = image.convert("RGB").copy()
        w, h = img.size
        cw, ch = int(w * ratio), int(h * ratio)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        center = img.crop((x0, y0, x0 + cw, y0 + ch))
        center_blurred = center.filter(ImageFilter.GaussianBlur(radius=radius))
        img.paste(center_blurred, (x0, y0))
        return img

    def _make_control_drop(self, image):
        """Matched-area control: blur periphery while keeping center clean."""
        radius = float(self.v91_cf_corruption_radius)
        ratio = max(0.1, min(0.95, float(self.v91_cf_center_ratio)))
        img = image.convert("RGB").copy()
        w, h = img.size
        cw, ch = int(w * ratio), int(h * ratio)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        center_clean = img.crop((x0, y0, x0 + cw, y0 + ch))
        blurred.paste(center_clean, (x0, y0))
        return blurred

    def _blur_box(self, image, box, radius):
        img = image.convert("RGB").copy()
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return img
        patch = img.crop((x0, y0, x1, y1))
        patch = patch.filter(ImageFilter.GaussianBlur(radius=radius))
        img.paste(patch, (x0, y0))
        return img

    def _patch_std(self, image, box):
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return 0.0
        arr = np.asarray(image.convert("L").crop((x0, y0, x1, y1)), dtype=np.float32)
        if arr.size == 0:
            return 0.0
        return float(arr.std())

    def _try_grid_cf_views(self, image, text_prompt, focus_norm, focus_raw):
        if not focus_norm or not str(focus_raw).strip():
            return None
        grid = max(2, int(self.v91_cf_grid_size))
        w, h = image.size
        if w < grid or h < grid:
            return None
        radius = float(self.v91_cf_corruption_radius)
        cand = {focus_norm: str(focus_raw)}
        base_lp = self._score_candidate_logprobs(image, text_prompt, cand).get(focus_norm)
        if base_lp is None:
            return None

        patches = []
        for gy in range(grid):
            y0 = int(round(gy * h / grid))
            y1 = int(round((gy + 1) * h / grid))
            for gx in range(grid):
                x0 = int(round(gx * w / grid))
                x1 = int(round((gx + 1) * w / grid))
                if x1 <= x0 or y1 <= y0:
                    continue
                box = (x0, y0, x1, y1)
                blurred = self._blur_box(image, box, radius)
                lp = self._score_candidate_logprobs(blurred, text_prompt, cand).get(focus_norm, base_lp)
                drop = float(base_lp - lp)
                patches.append({
                    "box": box,
                    "drop": drop,
                    "std": self._patch_std(image, box),
                    "area": float((x1 - x0) * (y1 - y0)),
                })
        if not patches:
            return None

        rel = max(patches, key=lambda p: p["drop"])
        if rel["drop"] < float(self.v91_cf_rel_drop_min):
            return None

        nonempty_std = float(self.v91_cf_ctrl_nonempty_std)
        ctrls = [p for p in patches if p["box"] != rel["box"] and p["std"] >= nonempty_std]
        if not ctrls:
            ctrls = [p for p in patches if p["box"] != rel["box"]]
        if not ctrls:
            return None

        rel_std = rel["std"]
        rel_area = rel["area"]
        ctrl = min(
            ctrls,
            key=lambda p: (
                abs(p["drop"]),              # prefer minimally relevant control
                abs(p["std"] - rel_std),     # match texture complexity
                abs(p["area"] - rel_area),   # match area
            ),
        )
        return {
            "rel_box": rel["box"],
            "ctrl_box": ctrl["box"],
            "rel_drop": rel["drop"],
            "ctrl_drop": ctrl["drop"],
        }

    def _build_cf_views(self, image, text_prompt, answer_space, no_cf_winner, scored, margin, entropy):
        """Build relevant/control drops and return view-quality metadata."""
        meta = {
            "mask_type": "center_fallback",
            "mask_quality": "low",
            "control_quality": "matched_periphery",
            "rel_drop": None,
            "ctrl_drop": None,
            "rel_box": None,
            "ctrl_box": None,
        }
        rel_img = self._make_relevant_drop(image)
        ctrl_img = self._make_control_drop(image)

        use_grid = bool(self.v91_cf_grid_enable)
        use_grid = use_grid and answer_space == "ocr_text_short"
        use_grid = use_grid and len(scored) >= int(self.v91_cf_grid_min_candidates)
        use_grid = use_grid and (
            margin < float(self.v91_cf_grid_margin_thresh)
            or entropy > float(self.v91_cf_grid_entropy_thresh)
        )
        if not use_grid:
            return rel_img, ctrl_img, meta

        focus_norm = no_cf_winner if no_cf_winner in scored else ""
        if not focus_norm and scored:
            focus_norm = max(scored.keys(), key=lambda k: scored[k]["score_general"])
        focus_raw = scored.get(focus_norm, {}).get("raw", "")
        picked = self._try_grid_cf_views(image, text_prompt, focus_norm, focus_raw)
        if not picked:
            return rel_img, ctrl_img, meta

        radius = float(self.v91_cf_corruption_radius)
        rel_img = self._blur_box(image, picked["rel_box"], radius)
        ctrl_img = self._blur_box(image, picked["ctrl_box"], radius)
        meta.update({
            "mask_type": "grid",
            "mask_quality": "medium",
            "control_quality": "matched_grid",
            "rel_drop": picked["rel_drop"],
            "ctrl_drop": picked["ctrl_drop"],
            "rel_box": list(picked["rel_box"]),
            "ctrl_box": list(picked["ctrl_box"]),
        })
        return rel_img, ctrl_img, meta

    def _compute_candidate_grid_cf_scores(self, image, text_prompt, candidate_map):
        """CF3 candidate-conditioned grid scores without OCR metadata.

        For each candidate answer a:
          1) find relevant patch that maximally drops logP(a),
          2) find matched control patch (low drop + similar texture/area),
          3) CF(a) = logP_ctrl(a) - logP_rel(a).
        """
        if not candidate_map:
            return {}, {}, {}, {}, {}

        grid = max(2, int(self.v91_cf_grid_size))
        w, h = image.size
        if w < grid or h < grid:
            return {}, {}, {}, {}, {}

        radius = float(self.v91_cf_corruption_radius)
        min_rel_drop = float(self.v91_cf_rel_drop_min)
        nonempty_std = float(self.v91_cf_ctrl_nonempty_std)

        base_lp = self._score_candidate_logprobs(image, text_prompt, candidate_map)
        if not base_lp:
            return {}, {}, {}, {}, {}

        patches = []
        for gy in range(grid):
            y0 = int(round(gy * h / grid))
            y1 = int(round((gy + 1) * h / grid))
            for gx in range(grid):
                x0 = int(round(gx * w / grid))
                x1 = int(round((gx + 1) * w / grid))
                if x1 <= x0 or y1 <= y0:
                    continue
                box = (x0, y0, x1, y1)
                blurred = self._blur_box(image, box, radius)
                lp_map = self._score_candidate_logprobs(blurred, text_prompt, candidate_map)
                filled_lp = {}
                for n_ans in candidate_map:
                    filled_lp[n_ans] = float(lp_map.get(n_ans, base_lp.get(n_ans, -100.0)))
                patches.append({
                    "box": box,
                    "std": self._patch_std(image, box),
                    "area": float((x1 - x0) * (y1 - y0)),
                    "lp": filled_lp,
                })

        if not patches:
            return {}, {}, {}, {}, {}

        raw_cf = {}
        logp_rel = {}
        logp_ctrl = {}
        cand_debug = {}

        for n_ans in candidate_map:
            # Relevant patch: maximal candidate-specific logprob drop.
            rel = max(
                patches,
                key=lambda p: float(base_lp.get(n_ans, -100.0) - p["lp"].get(n_ans, -100.0)),
            )
            rel_lp = float(rel["lp"].get(n_ans, -100.0))
            rel_drop = float(base_lp.get(n_ans, -100.0) - rel_lp)
            if rel_drop < min_rel_drop:
                continue

            rel_std = float(rel["std"])
            rel_area = float(rel["area"])
            ctrls = [p for p in patches if p["box"] != rel["box"] and float(p["std"]) >= nonempty_std]
            if not ctrls:
                ctrls = [p for p in patches if p["box"] != rel["box"]]
            if not ctrls:
                continue

            # Matched control: minimally relevant + similar texture and area.
            ctrl = min(
                ctrls,
                key=lambda p: (
                    abs(float(base_lp.get(n_ans, -100.0) - p["lp"].get(n_ans, -100.0))),
                    abs(float(p["std"]) - rel_std),
                    abs(float(p["area"]) - rel_area),
                ),
            )
            ctrl_lp = float(ctrl["lp"].get(n_ans, -100.0))
            ctrl_drop = float(base_lp.get(n_ans, -100.0) - ctrl_lp)

            raw_cf[n_ans] = ctrl_lp - rel_lp
            logp_rel[n_ans] = rel_lp
            logp_ctrl[n_ans] = ctrl_lp
            cand_debug[n_ans] = {
                "rel_box": list(rel["box"]),
                "ctrl_box": list(ctrl["box"]),
                "rel_drop": rel_drop,
                "ctrl_drop": ctrl_drop,
                "rel_std": rel_std,
                "ctrl_std": float(ctrl["std"]),
            }

        return raw_cf, base_lp, logp_rel, logp_ctrl, cand_debug

    def _score_candidate_logprobs(self, image, text_prompt, candidate_map):
        """Score each candidate by sequence-level logP under this image condition.
        Teacher-forced on candidate tokens:
          logP(a|x,q) = mean_t log p(token_t | prompt + token_<t, image).
        `candidate_map`: dict[norm_answer] -> raw candidate string.
        Returns dict[norm_answer] -> avg token log-prob."""
        if not candidate_map:
            return {}

        processor = self.processor
        tokenizer = processor.tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        try:
            prompt_inputs = processor(
                text=[text_prompt],
                images=[[image]],
                return_tensors="pt",
                padding=True,
            )
            prompt_inputs = {k: v.to(self.model.device) for k, v in prompt_inputs.items()}
        except Exception as e:
            print(f"[V91-CF2] processor failed: {e}")
            return {}

        prompt_ids = prompt_inputs["input_ids"][0]
        prompt_mask = prompt_inputs["attention_mask"][0]
        prompt_len = int(prompt_mask.sum().item())
        prefix_ids = prompt_ids[:prompt_len]
        prefix_mask = prompt_mask[:prompt_len]

        seq_ids = []
        seq_masks = []
        seq_labels = []
        norms = []
        for n_ans, raw_ans in candidate_map.items():
            raw = str(raw_ans).strip()
            tok = tokenizer(raw, add_special_tokens=False).input_ids
            if not tok:
                continue
            cand_ids = torch.tensor(tok, dtype=prefix_ids.dtype, device=self.model.device)
            ids = torch.cat([prefix_ids, cand_ids], dim=0)
            mask = torch.cat(
                [prefix_mask, torch.ones_like(cand_ids, dtype=prefix_mask.dtype)],
                dim=0,
            )
            labels = torch.full_like(ids, -100)
            labels[prompt_len:] = cand_ids
            seq_ids.append(ids)
            seq_masks.append(mask)
            seq_labels.append(labels)
            norms.append(n_ans)

        if not seq_ids:
            return {}

        max_len = max(int(x.size(0)) for x in seq_ids)
        bsz = len(seq_ids)
        input_ids = torch.full(
            (bsz, max_len), pad_id, dtype=seq_ids[0].dtype, device=self.model.device
        )
        attention_mask = torch.zeros(
            (bsz, max_len), dtype=seq_masks[0].dtype, device=self.model.device
        )
        labels = torch.full(
            (bsz, max_len), -100, dtype=seq_labels[0].dtype, device=self.model.device
        )
        for i, (ids, mask, lab) in enumerate(zip(seq_ids, seq_masks, seq_labels)):
            L = int(ids.size(0))
            input_ids[i, :L] = ids
            attention_mask[i, :L] = mask
            labels[i, :L] = lab

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": True,
        }
        for k, v in prompt_inputs.items():
            if k in ("input_ids", "attention_mask"):
                continue
            if hasattr(v, "dim") and v.dim() > 0 and int(v.size(0)) == 1:
                model_inputs[k] = v.repeat(bsz, *([1] * (v.dim() - 1)))
            else:
                model_inputs[k] = v

        try:
            out = self.model(**model_inputs)
        except Exception as e:
            print(f"[V91-CF2] forward failed: {e}")
            return {}

        logits = out.logits[:, :-1, :]
        target = input_ids[:, 1:]
        label_mask = (labels[:, 1:] != -100)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        token_logp = torch.gather(log_probs, 2, target.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * label_mask.float()
        denom = label_mask.sum(dim=1).clamp(min=1).float()
        seq_logp = token_logp.sum(dim=1) / denom
        result = {}
        for i, n_ans in enumerate(norms):
            lp = float(seq_logp[i].item())
            if n_ans not in result:
                result[n_ans] = lp
            else:
                result[n_ans] = max(result[n_ans], lp)
        return result

    # ------------------------------------------------------------------
    # Main rerank entry
    # ------------------------------------------------------------------

    def _v91_rerank(self, base_answer, message, formatted_messages,
                   images_augmented, dataset=None):
        self._v91_sample_counter += 1
        sid = self._v91_sample_counter
        question = getattr(self, "_v91_question_text", "") or _question_text(message)
        options = extract_options(question)

        diag = {
            "sample_id": sid,
            "benchmark": str(dataset) if dataset is not None else None,
            "base_answer": base_answer,
            "final_answer": base_answer,
            "applied": False,
            "cf_used": False,
            "fallback_to_ttaug": False,
            "format_veto": False,
            "final_changed": False,
        }

        try:
            view_answers = self._decode_per_view(formatted_messages, images_augmented)
            features = compute_candidate_features(view_answers, base_answer, options or None)
            answer_space = classify_answer_space(question, list(features.keys()) + [base_answer], options)
            if self.v91_answer_space_override is not None:
                answer_space = str(self.v91_answer_space_override)
            base_norm = normalize_answer(base_answer)

            scored = score_candidates_general(
                features, answer_space, base_norm,
                w_freq=self.v91_w_freq, w_format=self.v91_w_format,
                w_baseline_bias=self.v91_w_baseline_bias,
                w_length_risk=self.v91_w_length_risk,
            )

            # Compute pool-level uncertainty
            freqs = sorted([f["view_freq"] for f in features.values()], reverse=True)
            top1 = freqs[0] if freqs else 0
            top2 = freqs[1] if len(freqs) > 1 else 0
            margin = top1 - top2
            entropy = 0.0
            for f in freqs:
                if f > 0:
                    entropy -= f * math.log(f)
            base_freq = features.get(base_norm, {}).get("view_freq", 0.0)

            diag.update({
                "answer_space": answer_space,
                "decision_mode": str(getattr(self, "v91_decision_mode", "scored")),
                "n_candidates": len(features),
                "n_views": len(view_answers),
                "margin": margin,
                "entropy": entropy,
                "base_freq": base_freq,
                "options_n": len(options),
            })

            # v91_cf2: continuous log-likelihood CF scoring
            #   CF_score(cand) = logP_ctrl(cand) - logP_rel(cand)
            #   Gate: max CF_score >= v91_cf_logprob_tau
            cf_logprob_scores = {}
            rel_norm = ctrl_norm = ""
            cf_used = False
            beta = 0.0
            grounding_gap = {}
            max_grounding = 0
            loc_conf = float(self.v91_force_loc_conf)
            block_reason = "cf_not_evaluated"
            no_cf_winner = ""
            cf_winner = ""
            cf_margin = 0.0
            raw_cf = {}
            logp_pos = {}
            logp_rel = {}
            logp_ctrl = {}
            cf_candidate_debug = {}
            cf_view_meta = {
                "mask_type": "none",
                "mask_quality": "low",
                "control_quality": "low",
                "rel_drop": None,
                "ctrl_drop": None,
                "rel_box": None,
                "ctrl_box": None,
            }
            verifier_candidate = ""
            verifier_general_gap = 0.0
            cf3_block_counts = Counter()
            cf3_mode = str(getattr(self, "v91_cf3_mode", "strict") or "strict").strip().lower()
            cf3_force_grid = bool(getattr(self, "v91_cf3_force_grid", False)) or ("force_grid" in cf3_mode)
            cf3_no_quality_gate = bool(getattr(self, "v91_cf3_disable_quality_gate", False)) or ("no_quality_gate" in cf3_mode)
            cf3_score_only = bool(getattr(self, "v91_cf3_score_only", False)) or (cf3_mode == "score_only")
            cf3_force_switch = bool(getattr(self, "v91_cf3_force_switch_analysis", False)) or ("force_switch_analysis" in cf3_mode)
            cf3_route_reason = ""

            # NoCF winner (backbone winner before adding CF signal).
            decision_mode = str(getattr(self, "v91_decision_mode", "scored") or "scored").strip().lower()
            if decision_mode == "base_only":
                no_cf_winner = base_norm
            elif decision_mode == "first_view":
                first_norm = normalize_answer(view_answers[0]) if view_answers else ""
                no_cf_winner = first_norm or base_norm
            elif decision_mode == "majority_vote":
                if scored:
                    def _mv_key(k):
                        s = scored[k]
                        return (
                            s.get("view_freq", 0.0),
                            1 if s.get("is_base", False) else 0,
                            -float(s.get("length_words", 9999.0)),
                        )
                    no_cf_winner = max(scored.keys(), key=_mv_key)
                elif base_norm:
                    no_cf_winner = base_norm
                else:
                    no_cf_winner = ""
            else:
                if scored:
                    no_cf_winner = max(
                        scored.keys(), key=lambda k: scored[k]["score_general"]
                    )
                elif base_norm:
                    no_cf_winner = base_norm
                else:
                    no_cf_winner = ""

            if getattr(self, "cf_enabled", False):
                orig = self._v91_original_images or []
                orig_img = orig[0] if orig else None
                if isinstance(orig_img, list):
                    orig_img = orig_img[0] if orig_img else None
                if orig_img is not None:
                    cf_text = formatted_messages[0]
                    # Build candidate list for log-likelihood scoring
                    candidate_map = {
                        n_ans: scored[n_ans]["raw"] for n_ans in scored.keys()
                    }
                    if base_norm and base_norm not in candidate_map:
                        candidate_map[base_norm] = base_answer

                    if getattr(self, "cf3_enabled", False):
                        # CF3: candidate-conditioned grid (no center fallback),
                        # then use CF as verifier for switching from NoCF winner.
                        allowed_spaces = tuple(self.v91_cf3_switch_spaces)
                        if cf3_mode == "routed":
                            space_ok, cf3_route_reason = self._cf3_routed_space_ok(
                                answer_space, question, candidate_map, margin, entropy
                            )
                        else:
                            space_ok = answer_space in allowed_spaces
                        pool_ok = len(candidate_map) >= int(self.v91_cf_grid_min_candidates)
                        if cf3_force_grid:
                            space_ok = True
                            pool_ok = len(candidate_map) >= 1
                        if (
                            bool(self.v91_cf3_enable_candidate_grid)
                            and space_ok
                            and pool_ok
                        ):
                            raw_cf, logp_pos, logp_rel, logp_ctrl, cf_candidate_debug = (
                                self._compute_candidate_grid_cf_scores(
                                    orig_img, cf_text, candidate_map
                                )
                            )
                            if raw_cf:
                                cf_view_meta.update({
                                    "mask_type": "candidate_grid",
                                    "mask_quality": "high",
                                    "control_quality": "matched_grid",
                                })
                            else:
                                block_reason = "no_grid_mask"
                                cf3_block_counts["no_grid_mask"] += 1
                        else:
                            if not space_ok:
                                block_reason = cf3_route_reason or "answer_space_block"
                                cf3_block_counts[block_reason] += 1
                            elif not pool_ok:
                                block_reason = "no_valid_alternative"
                                cf3_block_counts["no_valid_alternative"] += 1
                            else:
                                block_reason = "unknown"
                                cf3_block_counts["unknown"] += 1
                    else:
                        # CF2 legacy behavior.
                        rel_img, ctrl_img, cf_view_meta = self._build_cf_views(
                            orig_img, cf_text, answer_space, no_cf_winner, scored, margin, entropy
                        )

                        # Run log-likelihood scoring on both dropped images
                        logp_pos = self._score_candidate_logprobs(
                            orig_img, cf_text, candidate_map
                        )
                        logp_rel = self._score_candidate_logprobs(
                            rel_img, cf_text, candidate_map
                        )
                        logp_ctrl = self._score_candidate_logprobs(
                            ctrl_img, cf_text, candidate_map
                        )

                        # Binary fallback: rel/ctrl answer strings for grounding
                        rel_text = self._decode_single(rel_img, cf_text)
                        ctrl_text = self._decode_single(ctrl_img, cf_text)
                        rel_norm = normalize_answer(rel_text)
                        ctrl_norm = normalize_answer(ctrl_text)

                        # Continuous CF_score = logP_ctrl - logP_rel
                        for n_ans in scored:
                            lp_r = logp_rel.get(n_ans, -100.0)
                            lp_c = logp_ctrl.get(n_ans, -100.0)
                            raw_cf[n_ans] = lp_c - lp_r

                    # Normalize CF scores across candidates (z-score + clip).
                    if raw_cf:
                        vals = np.array(list(raw_cf.values()), dtype=np.float32)
                        mean = float(vals.mean())
                        std = float(vals.std())
                        clip = float(self.v91_cf_score_clip)
                        if std > 1e-6:
                            for n_ans, v in raw_cf.items():
                                z = (v - mean) / std
                                cf_logprob_scores[n_ans] = float(max(-clip, min(clip, z)))
                        else:
                            for n_ans in raw_cf:
                                cf_logprob_scores[n_ans] = 0.0
                    max_cf = (
                        max(cf_logprob_scores.values()) if cf_logprob_scores else 0.0
                    )

                    # Binary grounding (legacy beta routing only).
                    if not getattr(self, "cf3_enabled", False):
                        for n_ans in scored:
                            rm = 1 if n_ans == rel_norm else 0
                            cm = 1 if n_ans == ctrl_norm else 0
                            grounding_gap[n_ans] = cm - rm
                        max_grounding = max(grounding_gap.values()) if grounding_gap else 0

                        beta = beta_for_sample(
                            answer_space, entropy=entropy, margin=margin,
                            max_grounding=max_grounding, loc_conf=loc_conf,
                            n_candidates=len(features),
                            base_freq=base_freq,
                        )
                        if cf_view_meta.get("mask_quality") == "low":
                            beta *= float(self.v91_cf_low_quality_beta_scale)
                    else:
                        max_grounding = 0
                        beta = 1.0

                    # CF margin against no_cf winner.
                    if cf_logprob_scores:
                        cf_winner = max(
                            cf_logprob_scores.keys(),
                            key=lambda k: cf_logprob_scores.get(k, -1e9),
                        )
                        cf_margin = (
                            cf_logprob_scores.get(cf_winner, 0.0)
                            - cf_logprob_scores.get(no_cf_winner, 0.0)
                        )

                    tau_cf = float(self.v91_cf_logprob_tau)
                    if getattr(self, "cf3_enabled", False):
                        # CF3 verifier gate: CF must strongly favor an alternative
                        # and alternative should not be much worse on general score.
                        if (
                            cf_logprob_scores
                            and no_cf_winner in scored
                            and no_cf_winner in cf_logprob_scores
                        ):
                            ranked = sorted(
                                cf_logprob_scores.items(), key=lambda kv: kv[1], reverse=True
                            )
                            alt_seen = 0
                            alt_scored = 0
                            for cand_norm, _ in ranked:
                                if cand_norm == no_cf_winner:
                                    continue
                                alt_seen += 1
                                if cand_norm not in scored:
                                    continue
                                alt_scored += 1
                                delta_cf = cf_logprob_scores.get(cand_norm, 0.0) - cf_logprob_scores.get(no_cf_winner, 0.0)
                                gap = scored[no_cf_winner]["score_general"] - scored[cand_norm]["score_general"]
                                view_sup = int(scored[cand_norm].get("view_count", 0))
                                fmt_ok = scored[cand_norm].get("fmt_validity", 0.0) >= 0.8
                                # Diagnostic-open modes:
                                # - score_only: score but never switch.
                                # - no_quality_gate: allow switching when delta_cf>0 (or tau in strict mode).
                                # - force_grid_no_quality_gate: same plus force grid path above.
                                if cf3_score_only:
                                    cf3_block_counts["cf3_score_only"] += 1
                                    continue
                                if delta_cf <= (0.0 if cf3_no_quality_gate else tau_cf):
                                    cf3_block_counts["low_cf_margin"] += 1
                                    continue
                                if (not cf3_no_quality_gate) and (gap > float(self.v91_cf3_general_gap_tau)):
                                    cf3_block_counts["general_score_gap_too_large"] += 1
                                    continue
                                if (not cf3_no_quality_gate) and (view_sup < int(self.v91_cf_min_view_support)):
                                    cf3_block_counts["no_valid_alternative"] += 1
                                    continue
                                if not fmt_ok:
                                    cf3_block_counts["format_veto"] += 1
                                    continue
                                if (not cf3_no_quality_gate) and cf_view_meta.get("mask_quality") not in ("medium", "high"):
                                    cf3_block_counts["low_mask_quality"] += 1
                                    continue
                                if (not cf3_no_quality_gate) and cf_view_meta.get("control_quality") not in ("matched_grid", "matched_periphery"):
                                    cf3_block_counts["low_control_quality"] += 1
                                    continue
                                if (
                                    delta_cf > (0.0 if cf3_no_quality_gate else tau_cf)
                                ):
                                    verifier_candidate = cand_norm
                                    verifier_general_gap = float(gap)
                                    break
                            if not verifier_candidate and (alt_seen == 0 or alt_scored == 0):
                                cf3_block_counts["no_valid_alternative"] += 1
                        cf_used = bool(verifier_candidate)
                        if not cf_logprob_scores:
                            block_reason = "no_cf_scores"
                        elif not verifier_candidate:
                            if cf3_block_counts:
                                block_reason = max(cf3_block_counts.items(), key=lambda kv: kv[1])[0]
                            else:
                                block_reason = "unknown"
                        else:
                            block_reason = "cf3_verifier_pass"
                    else:
                        cf_used = (
                            bool(cf_logprob_scores)
                            and (beta > 0)
                            and (cf_margin > tau_cf)
                            and (max_cf > 0)
                        )
                        if not cf_logprob_scores:
                            block_reason = "no_cf_scores"
                        elif beta <= 0:
                            block_reason = "beta_zero"
                        elif cf_margin <= tau_cf:
                            block_reason = "cf_margin_low"
                        elif max_cf <= 0:
                            block_reason = "cf_not_positive"
                        else:
                            block_reason = "cf_gate_pass"
                    diag.update({
                        "mask_type": cf_view_meta.get("mask_type"),
                        "mask_quality": cf_view_meta.get("mask_quality"),
                        "control_quality": cf_view_meta.get("control_quality"),
                        "rel_box": cf_view_meta.get("rel_box"),
                        "ctrl_box": cf_view_meta.get("ctrl_box"),
                        "rel_drop": cf_view_meta.get("rel_drop"),
                        "ctrl_drop": cf_view_meta.get("ctrl_drop"),
                        "rel_answer": rel_norm,
                        "ctrl_answer": ctrl_norm,
                        "rel_norm": rel_norm,
                        "ctrl_norm": ctrl_norm,
                        "logp_pos": logp_pos,
                        "logp_rel": logp_rel,
                        "logp_ctrl": logp_ctrl,
                        "cf_score_raw": raw_cf,
                        "cf_score": cf_logprob_scores,
                        "cf_candidate_debug": cf_candidate_debug,
                        "cf_winner": cf_winner,
                        "no_cf_winner": no_cf_winner,
                        "cf_margin": cf_margin,
                        "cf_gate_tau": tau_cf,
                        "max_grounding": max_grounding,
                        "max_cf_score": max_cf,
                        "loc_conf": loc_conf,
                        "beta": beta,
                        "cf_verifier_candidate": verifier_candidate,
                        "cf_verifier_general_gap": verifier_general_gap,
                        "cf3_mode": cf3_mode,
                        "cf3_force_grid": cf3_force_grid,
                        "cf3_no_quality_gate": cf3_no_quality_gate,
                        "cf3_score_only": cf3_score_only,
                        "cf3_force_switch": cf3_force_switch,
                        "cf3_route_reason": cf3_route_reason,
                        "cf3_block_counts": dict(cf3_block_counts),
                        "block_reason": block_reason,
                    })
                else:
                    block_reason = "missing_original_image"
            else:
                block_reason = "cf_disabled"

            # Add CF score (CF3 keeps as diagnostic; CF2 uses as booster).
            for n_ans, sc in scored.items():
                if getattr(self, "cf3_enabled", False):
                    cf_s = cf_logprob_scores.get(n_ans, 0.0)
                else:
                    cf_s = cf_logprob_scores.get(n_ans, 0.0) if cf_used else 0.0
                sc["cf_score"] = cf_s
                sc["score_final"] = sc["score_general"] + beta * cf_s

            # Final decision:
            # - CF2: score booster argmax.
            # - CF3: NoCF winner primary; switch only if verifier gate passes.
            if getattr(self, "cf3_enabled", False):
                best_norm = no_cf_winner
                if cf3_force_switch and cf_winner and (cf_winner in scored) and (cf_winner != no_cf_winner):
                    best_norm = cf_winner
                    cf_used = True
                    block_reason = "cf3_force_switch_analysis"
                elif cf_used and verifier_candidate:
                    best_norm = verifier_candidate
            else:
                best_norm = no_cf_winner if no_cf_winner else (
                    max(scored.keys(), key=lambda k: scored[k]["score_final"]) if scored else ""
                )
                best_score = scored.get(best_norm, {}).get("score_final", -1e9)
                for n_ans, sc in scored.items():
                    if sc["score_final"] > best_score:
                        best_score = sc["score_final"]
                        best_norm = n_ans

                # High-precision guard: only allow CF-driven answer change when support is strong.
                if cf_used and best_norm != no_cf_winner:
                    min_sup = int(self.v91_cf_min_view_support)
                    best_view_sup = int(scored.get(best_norm, {}).get("view_count", 0))
                    fmt_ok = scored.get(best_norm, {}).get("fmt_validity", 0.0) >= 0.8
                    if not (best_view_sup >= min_sup and fmt_ok and cf_margin > float(self.v91_cf_logprob_tau)):
                        best_norm = no_cf_winner
                        block_reason = "precision_gate_block"

            # Format veto: yes_no / mcq must conform; numeric should not
            # silently mutate; if veto, fall back to base_answer.
            chosen_raw = scored[best_norm]["raw"] if best_norm in scored else base_answer
            veto = False
            if answer_space == "yes_no" and not is_yes_no(chosen_raw):
                veto = True
            elif answer_space == "multiple_choice" and not (
                is_mcq_letter(chosen_raw) or scored[best_norm].get("from_options")
            ):
                veto = True
            elif answer_space == "numeric":
                # Strong evidence required to change a numeric answer
                if best_norm != base_norm:
                    if scored[best_norm]["view_freq"] < max(
                        0.5, scored.get(base_norm, {}).get("view_freq", 0) + 0.25
                    ):
                        veto = True
            if veto:
                diag["format_veto"] = True
                final_text = base_answer
            else:
                final_text = chosen_raw

            # Track applied / changed
            applied = (best_norm != base_norm) or cf_used
            final_changed = normalize_answer(final_text) != base_norm
            cf_final_changed = normalize_answer(final_text) != normalize_answer(
                scored.get(no_cf_winner, {}).get("raw", base_answer)
            )
            diag.update({
                "applied": applied,
                "cf_used": cf_used,
                "final_answer": final_text,
                "final_changed": final_changed,
                "cf_final_changed": cf_final_changed,
                "fallback_to_ttaug": (final_text == base_answer and applied),
                "candidate_list": list(scored.keys()),
                "no_cf_winner": no_cf_winner,
                "cf_winner": cf_winner,
                "block_reason": block_reason,
                "scored_top": sorted(
                    [(k, v["score_final"], v["score_general"], v.get("cf_score", 0),
                      v["view_freq"]) for k, v in scored.items()],
                    key=lambda x: -x[1],
                )[:5],
            })

            if self.cf_enabled and getattr(self, "cf3_enabled", False):
                mode = "v91cf3"
            elif self.cf_enabled:
                mode = "v91cf"
            else:
                mode = "v91nocf"
            print(
                f"[V91-RERANK] mode={mode} sample={sid} space={answer_space} "
                f"n_cand={len(features)} margin={margin:.3f} ent={entropy:.3f} "
                f"beta={beta:.2f} cf_used={cf_used} g_max={max_grounding} "
                f"changed={final_changed} veto={diag['format_veto']} "
                f"base={base_norm!r} best={best_norm!r} block={block_reason}",
                flush=True,
            )

        except Exception as e:
            diag["block_reason"] = "exception"
            diag["exception_detail"] = f"{type(e).__name__}:{e}"
            print(f"[V91-RERANK] sample={sid} EXCEPTION: {e}", flush=True)
            final_text = base_answer

        # Append diagnostic
        if getattr(self, "v91_diag_per_sample", True):
            try:
                with open(self._v91_diag_path, "a") as fp:
                    fp.write(json.dumps(diag, default=str) + "\n")
            except Exception as e:
                print(f"[V91] failed to write diag: {e}")

        return final_text


# ---------------------------------------------------------------------------
# Subclasses (the names referenced from JSON configs)
# ---------------------------------------------------------------------------


class V91NoCF_SmolVLM2_2B(V91Adapter_SmolVLM2):
    cf_enabled = False


class V91CF_SmolVLM2_2B(V91Adapter_SmolVLM2):
    cf_enabled = True


class V91CF3_SmolVLM2_2B(V91Adapter_SmolVLM2):
    cf_enabled = True
    cf3_enabled = True


class V91CF3Routed_SmolVLM2_2B(V91Adapter_SmolVLM2):
    cf_enabled = True
    cf3_enabled = True
