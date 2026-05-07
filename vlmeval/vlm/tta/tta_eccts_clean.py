"""
ECCTS adapter for SmolVLM2-2.2B, layered on the clean repo's verified
paper-TTAug-classical baseline.

Phase A (this file initially): bit-identical disabled bypass + 3 variant
subclasses that, with eccts_enabled=False, MUST reproduce
TTAugAdapter_SmolVLM2 sample-by-sample.

Phase B (added next): full ECCTS rerank pipeline with original-image-only
counterfactuals (1 relevant_drop + 1 control_drop), per-view individual
greedy candidates, margin/entropy/grounding gates, v7c/v7d/v7e dispatch,
per-sample JSONL diagnostics.
"""

import copy
import json
import math
import os
import string
import time
import types
from collections import Counter
from typing import Optional

import torch
from PIL import Image, ImageFilter

from .tta_smolvlm import TTAugAdapter_SmolVLM2, apply_deterministic_seed


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class ECCTSAdapter_SmolVLM2(TTAugAdapter_SmolVLM2):
    """ECCTS-on-paper-TTAug-classical adapter for SmolVLM2-2.2B.

    Subclasses set `eccts_variant` to one of {"v7c", "v7d", "v7e"}. When
    `eccts_enabled=False`, this class behaves IDENTICALLY to
    TTAugAdapter_SmolVLM2 (strict bit-identical bypass for sanity check).
    """

    # subclass override
    eccts_variant: str = "v7c"

    # Default ECCTS knobs (overridable from JSON adapter_args)
    _ECCTS_DEFAULTS = {
        "eccts_enabled": True,
        # main ECCTS toggles
        "counterfactual_answer_rerank": True,
        "counterfactual_rerank_top_k": 3,
        "counterfactual_rerank_alpha": 1.0,
        "counterfactual_rerank_alpha_base": 0.5,
        "counterfactual_rerank_alpha_max": 1.5,
        "counterfactual_rerank_entropy_center": 1.0,
        "counterfactual_rerank_entropy_slope": 4.0,
        "counterfactual_rerank_grounding_tau": 0.0,
        # gates
        "counterfactual_rerank_uncertainty_gate": True,
        "counterfactual_rerank_quality_gate": True,
        "counterfactual_rerank_margin_threshold": 0.25,
        "counterfactual_rerank_entropy_threshold": 1.0,
        "counterfactual_rerank_min_grounding_gap": 1.0,
        "counterfactual_rerank_loc_conf_threshold": 0.55,
        # debug flags
        "eccts_disable_uncertainty_gate": False,
        "eccts_disable_quality_gate": False,
        "eccts_force_loc_conf": 1.0,  # placeholder until SmolVLM2 localization lands
        # CF corruption
        "counterfactual_corruption": "blur",
        "counterfactual_blur_radius": 7.0,
        "counterfactual_center_ratio": 0.6,
        # diagnostics
        "eccts_diag_per_sample": True,
        "eccts_diag_dir": None,  # default = $CACHE_PATH/diagnostics
        # determinism — sha256-derived per-sample seed (NOT Python hash())
        "deterministic_seeding": True,
        "deterministic_seeding_log": True,
    }

    def __init__(self, model_args, text_aug_args, image_aug_args, **adapter_args):
        # Apply ECCTS defaults BEFORE forwarding so JSON values still win
        merged_args = dict(self._ECCTS_DEFAULTS)
        merged_args.update(adapter_args)

        # Make sure variant is set (subclass attr or explicit knob)
        merged_args.setdefault("eccts_variant", self.eccts_variant)

        super().__init__(model_args, text_aug_args, image_aug_args, **merged_args)

        # Diagnostics state
        self._eccts_diag_records = []
        self._eccts_diag_path = None
        if self.eccts_diag_dir is None:
            cache = os.environ.get("CACHE_PATH", ".")
            self.eccts_diag_dir = os.path.join(cache, "diagnostics")
        os.makedirs(self.eccts_diag_dir, exist_ok=True)
        self._eccts_diag_path = os.path.join(
            self.eccts_diag_dir, f"eccts_{self.eccts_variant}_samples.jsonl"
        )
        self._eccts_summary_path = os.path.join(
            self.eccts_diag_dir, f"eccts_{self.eccts_variant}_summary.json"
        )
        self._eccts_sample_counter = 0

        print("================================")
        print(f"Initializing ECCTSAdapter_SmolVLM2 variant={self.eccts_variant}")
        print(f"  eccts_enabled={self.eccts_enabled}")
        print(f"  diag_dir={self.eccts_diag_dir}")
        print(f"  disable_unc_gate={self.eccts_disable_uncertainty_gate}")
        print(f"  disable_qual_gate={self.eccts_disable_quality_gate}")
        print("================================")

        # Save original (transformer-built-in) _sample so we can run un-aggregated
        # individual greedy decoding in Phase B. Captured here BEFORE any swap.
        try:
            from transformers.generation.utils import GenerationMixin
            self._eccts_original_sample = GenerationMixin._sample
        except Exception as e:
            print(f"[ECCTS] WARNING: could not capture original _sample: {e}")
            self._eccts_original_sample = None

        self._eccts_original_images = None  # set per-sample by generate_inner

    # ------------------------------------------------------------------
    # Generation entry point — strict bit-identical bypass when disabled
    # ------------------------------------------------------------------

    def generate_inner(self, message, dataset=None):
        # Deterministic seeding BEFORE any RNG-using step (text + image
        # augmentation, CF view construction). Variant identity is NOT in
        # the seed so v7c/v7d/v7e/disabled draw the same augmented views.
        if getattr(self, "deterministic_seeding", True):
            apply_deterministic_seed(
                dataset,
                message,
                log=getattr(self, "deterministic_seeding_log", True),
            )
        # Inline the parent's generate_inner so we can capture the
        # pre-augmentation original images. (Calling super().generate_inner
        # would re-run text augmentation, breaking determinism.)
        formatted_messages, formatted_images = self.build_prompt_cases(message, dataset)

        images = (
            [formatted_images]
            if isinstance(formatted_images, Image.Image)
            else formatted_images
        )

        # Stash original images BEFORE augmentation for ECCTS CF construction
        self._eccts_original_images = list(images)

        images_augmented, applied_transforms = self.image_augment(images)

        save_visual_samples_flag = os.environ.get(
            "SAVE_VISUAL_SAMPLES", "False"
        ).lower() in ("1", "true", "yes")
        if save_visual_samples_flag:
            self.save_inputs_grid_prompts(
                message,
                formatted_messages,
                images_augmented,
                applied_transforms,
                dataset,
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
                images_augmented = images_augmented[
                    : max(1, len(images_augmented) // 2)
                ]
                formatted_messages = formatted_messages[
                    : max(1, len(formatted_messages) // 2)
                ]
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

    def generate_inner_helper(
        self, message, formatted_messages, images_augmented, dataset=None
    ):
        # Always compute the bit-identical paper-TTAug-classical answer first.
        base_answer = super().generate_inner_helper(
            message, formatted_messages, images_augmented, dataset
        )

        # Strict bypass: when ECCTS is disabled, return EXACTLY the parent answer.
        # No extra parsing, no reformatting, no post-processing.
        if not getattr(self, "eccts_enabled", True):
            return base_answer

        # Phase B: ECCTS rerank pipeline (added in next file revision).
        # For Phase A we still return base_answer here so the bit-identical
        # check passes even when eccts_enabled=True (until Phase B lands).
        return self._eccts_rerank(
            base_answer=base_answer,
            message=message,
            formatted_messages=formatted_messages,
            images_augmented=images_augmented,
            dataset=dataset,
        )

    # ------------------------------------------------------------------
    # Phase B: ECCTS rerank pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_answer(text):
        if text is None:
            return ""
        s = str(text).strip().lower()
        # strip trailing punctuation so "centrale." == "centrale"
        s = s.translate(str.maketrans("", "", string.punctuation)).strip()
        return s

    def _make_relevant_drop(self, image):
        """First-version: blur the central crop of the image.

        The "relevant" region heuristic without SAM/localization is the
        central part of the image, where the question subject typically lies.
        This is intentionally crude — Step 6 will replace with SmolVLM2
        hidden-state localization.
        """
        if image is None:
            return None
        radius = float(getattr(self, "counterfactual_blur_radius", 7.0))
        ratio = float(getattr(self, "counterfactual_center_ratio", 0.6))
        ratio = max(0.1, min(0.95, ratio))

        img = image.convert("RGB").copy()
        w, h = img.size
        cw, ch = int(w * ratio), int(h * ratio)
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        center = img.crop((x0, y0, x0 + cw, y0 + ch))
        center_blurred = center.filter(ImageFilter.GaussianBlur(radius=radius))
        img.paste(center_blurred, (x0, y0))
        return img

    def _make_control_drop(self, image):
        """First-version: blur the periphery (everything except the central crop).

        Same total area corrupted as relevant_drop but in a complementary
        region. Differential between rel and ctrl gives the grounding signal.
        """
        if image is None:
            return None
        radius = float(getattr(self, "counterfactual_blur_radius", 7.0))
        ratio = float(getattr(self, "counterfactual_center_ratio", 0.6))
        ratio = max(0.1, min(0.95, ratio))

        img = image.convert("RGB").copy()
        w, h = img.size
        cw, ch = int(w * ratio), int(h * ratio)
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        # Blur whole image, then paste back the un-blurred center
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        center_clean = img.crop((x0, y0, x0 + cw, y0 + ch))
        blurred.paste(center_clean, (x0, y0))
        return blurred

    def _decode_individual_with_original_sample(self, inputs):
        """Run self.model.generate WITHOUT token-level aggregation.

        Temporarily restores the original transformers _sample so each batch
        item is decoded independently. Returns one decoded string per batch
        element.
        """
        if self._eccts_original_sample is None:
            # Fallback: use the modified _sample (results in N copies of
            # the same aggregated answer). Marked as "indiv_decode_unavailable".
            generated_ids = self.model.generate(**inputs, **self.kwargs)
        else:
            modified_sample = self.model._sample
            try:
                self.model._sample = types.MethodType(
                    self._eccts_original_sample, self.model
                )
                generated_ids = self.model.generate(**inputs, **self.kwargs)
            finally:
                self.model._sample = modified_sample

        decoded = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1):],
            skip_special_tokens=True,
        )
        return [t.strip() for t in decoded]

    def _decode_single_image(self, image, text_prompt):
        """Forward one image + one text prompt through the model with the
        original (un-aggregated) _sample, return the decoded string.
        """
        self.processor.tokenizer.padding_side = "left"
        cf_inputs = self.processor(
            text=[text_prompt],
            images=[[image]],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        if self._eccts_original_sample is None:
            ids = self.model.generate(**cf_inputs, **self.kwargs)
        else:
            modified_sample = self.model._sample
            try:
                self.model._sample = types.MethodType(
                    self._eccts_original_sample, self.model
                )
                ids = self.model.generate(**cf_inputs, **self.kwargs)
            finally:
                self.model._sample = modified_sample

        decoded = self.processor.batch_decode(
            ids[:, cf_inputs["input_ids"].size(1):],
            skip_special_tokens=True,
        )
        return decoded[0].strip()

    def _compute_alpha_for_variant(self, entropy):
        """v7c = fixed alpha; v7d/v7e = sigmoid(entropy) interpolation."""
        if self.eccts_variant == "v7c":
            return float(self.counterfactual_rerank_alpha), False, 0.0

        base_alpha = float(self.counterfactual_rerank_alpha_base)
        max_alpha = float(self.counterfactual_rerank_alpha_max)
        ent_center = float(self.counterfactual_rerank_entropy_center)
        ent_slope = float(self.counterfactual_rerank_entropy_slope)
        try:
            sigmoid_in = -ent_slope * (entropy - ent_center)
            sig = 1.0 / (1.0 + math.exp(max(min(sigmoid_in, 50.0), -50.0)))
        except OverflowError:
            sig = 0.0 if (entropy - ent_center) > 0 else 1.0
        alpha = base_alpha + (max_alpha - base_alpha) * sig
        alpha = max(0.0, min(max_alpha, alpha))
        use_net_grounding = (self.eccts_variant == "v7e")
        tau = float(self.counterfactual_rerank_grounding_tau) if use_net_grounding else 0.0
        return alpha, use_net_grounding, tau

    @staticmethod
    def _compute_margin_entropy(vote_counter):
        n_total = sum(vote_counter.values())
        if n_total == 0:
            return 0.0, 0.0
        sorted_votes = sorted(vote_counter.values(), reverse=True)
        top1 = sorted_votes[0]
        top2 = sorted_votes[1] if len(sorted_votes) > 1 else 0
        margin = (top1 - top2) / n_total
        ent = 0.0
        for v in sorted_votes:
            if v <= 0:
                continue
            p = v / n_total
            ent -= p * math.log(p)
        return float(margin), float(ent)

    def _record_diagnostic(self, record):
        if not getattr(self, "eccts_diag_per_sample", True):
            return
        try:
            with open(self._eccts_diag_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            print(f"[ECCTS] WARNING: failed to write diagnostic: {e}")

    def _eccts_rerank(
        self,
        base_answer,
        message,
        formatted_messages,
        images_augmented,
        dataset=None,
    ):
        self._eccts_sample_counter += 1
        sample_id = self._eccts_sample_counter
        variant = self.eccts_variant

        diag = {
            "sample_id": sample_id,
            "benchmark": str(dataset) if dataset is not None else None,
            "variant": variant,
            "applied": False,
            "final_answer": base_answer,
            "base_answer": base_answer,
            "block_reason": None,
            "final_changed": False,
        }

        try:
            # ---- Step 1: per-view individual greedy -> candidate vote distribution
            self.processor.tokenizer.padding_side = "left"
            inputs = self.processor(
                text=formatted_messages,
                images=images_augmented,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)
            indiv_texts = self._decode_individual_with_original_sample(inputs)
            indiv_norms = [self._normalize_answer(t) for t in indiv_texts]
            vote_counter = Counter(indiv_norms)
            n_views = sum(vote_counter.values())
            base_norm = self._normalize_answer(base_answer)
            if base_norm and base_norm not in vote_counter:
                # ensure base is at least a candidate
                vote_counter[base_norm] = max(1, vote_counter.get(base_norm, 0))

            margin, entropy = self._compute_margin_entropy(vote_counter)
            diag["margin"] = margin
            diag["entropy"] = entropy
            diag["candidate_count"] = len(vote_counter)
            diag["indiv_views"] = list(zip(indiv_texts, indiv_norms))[:8]

            # ---- Step 2: build CF rel + ctrl from original image, decode each
            orig_images = self._eccts_original_images or []
            if not orig_images:
                diag["block_reason"] = "no_original_image"
                self._record_diagnostic(diag)
                return base_answer
            orig_img = orig_images[0]
            if isinstance(orig_img, list):
                orig_img = orig_img[0] if orig_img else None
            if orig_img is None:
                diag["block_reason"] = "no_original_image"
                self._record_diagnostic(diag)
                return base_answer

            rel_img = self._make_relevant_drop(orig_img)
            ctrl_img = self._make_control_drop(orig_img)

            cf_text = formatted_messages[0]
            rel_text = self._decode_single_image(rel_img, cf_text)
            ctrl_text = self._decode_single_image(ctrl_img, cf_text)
            rel_norm = self._normalize_answer(rel_text)
            ctrl_norm = self._normalize_answer(ctrl_text)
            diag["rel_answer"] = rel_text
            diag["ctrl_answer"] = ctrl_text
            diag["rel_norm"] = rel_norm
            diag["ctrl_norm"] = ctrl_norm

            # ---- Step 3: top-k candidates + grounding gap
            top_k_n = max(1, int(self.counterfactual_rerank_top_k))
            top_k = vote_counter.most_common(top_k_n)
            grounding_gap = {}
            for cand, _ in top_k:
                ctrl_match = 1 if cand == ctrl_norm else 0
                rel_match = 1 if cand == rel_norm else 0
                grounding_gap[cand] = ctrl_match - rel_match
            max_grounding = max(grounding_gap.values()) if grounding_gap else 0
            diag["grounding_gap"] = grounding_gap
            diag["max_grounding"] = max_grounding
            diag["top_k_votes"] = top_k

            # ---- Step 4: gates
            loc_conf = float(getattr(self, "eccts_force_loc_conf", 1.0))
            margin_thr = float(self.counterfactual_rerank_margin_threshold)
            entropy_thr = float(self.counterfactual_rerank_entropy_threshold)
            min_gap = float(self.counterfactual_rerank_min_grounding_gap)
            loc_thr = float(self.counterfactual_rerank_loc_conf_threshold)

            unc_pass = (margin < margin_thr) or (entropy > entropy_thr)
            if self.eccts_disable_uncertainty_gate or not self.counterfactual_rerank_uncertainty_gate:
                unc_pass = True

            qual_pass = (max_grounding >= min_gap) and (loc_conf >= loc_thr)
            if self.eccts_disable_quality_gate or not self.counterfactual_rerank_quality_gate:
                qual_pass = True

            diag["unc_pass"] = unc_pass
            diag["qual_pass"] = qual_pass
            diag["loc_conf"] = loc_conf

            if not (unc_pass and qual_pass):
                reasons = []
                if not unc_pass:
                    reasons.append(f"unc(margin={margin:.3f},entropy={entropy:.3f})")
                if not qual_pass:
                    reasons.append(f"qual(grounding={max_grounding},loc={loc_conf:.3f})")
                diag["block_reason"] = "+".join(reasons)
                # Print one line per sample for grep
                print(
                    f"[V7-RERANK] sample={sample_id} variant={variant} applied=False "
                    f"margin={margin:.3f} entropy={entropy:.3f} max_grounding={max_grounding} "
                    f"loc_conf={loc_conf:.3f} unc_pass={unc_pass} qual_pass={qual_pass} "
                    f"block={diag['block_reason']}",
                    flush=True,
                )
                self._record_diagnostic(diag)
                return base_answer

            # ---- Step 5: variant-specific scoring
            alpha, use_net_grounding, tau = self._compute_alpha_for_variant(entropy)
            diag["alpha"] = alpha
            diag["use_net_grounding"] = use_net_grounding
            diag["tau"] = tau

            best_cand = base_norm if base_norm else top_k[0][0]
            best_score = -float("inf")
            scored = {}
            for cand, votes in top_k:
                gap = grounding_gap.get(cand, 0)
                gain = max(0.0, gap - tau) if use_net_grounding else float(gap)
                score = float(votes) + alpha * gain
                scored[cand] = (votes, gap, score)
                if score > best_score:
                    best_score = score
                    best_cand = cand
            diag["scored"] = scored

            # Map normalized best_cand back to a surface form. Use the first
            # individual answer that normalises to it; otherwise fall back to
            # the candidate string itself.
            chosen_text = base_answer
            for orig_text, norm in zip(indiv_texts, indiv_norms):
                if norm == best_cand:
                    chosen_text = orig_text
                    break
            else:
                if best_cand == self._normalize_answer(rel_text):
                    chosen_text = rel_text
                elif best_cand == self._normalize_answer(ctrl_text):
                    chosen_text = ctrl_text
                else:
                    chosen_text = best_cand

            final_changed = self._normalize_answer(chosen_text) != base_norm
            diag["applied"] = True
            diag["eccts_answer"] = chosen_text
            diag["final_answer"] = chosen_text
            diag["final_changed"] = final_changed
            print(
                f"[V7-RERANK] sample={sample_id} variant={variant} applied=True "
                f"alpha={alpha:.3f} margin={margin:.3f} entropy={entropy:.3f} "
                f"max_grounding={max_grounding} loc_conf={loc_conf:.3f} "
                f"base={base_norm!r} chosen={self._normalize_answer(chosen_text)!r} "
                f"changed={final_changed}",
                flush=True,
            )
            self._record_diagnostic(diag)
            return chosen_text

        except Exception as e:
            diag["block_reason"] = f"exception:{type(e).__name__}:{e}"
            print(f"[V7-RERANK] sample={sample_id} variant={variant} EXCEPTION: {e}", flush=True)
            self._record_diagnostic(diag)
            return base_answer


# ---------------------------------------------------------------------------
# Variant subclasses (the names referenced from JSON configs)
# ---------------------------------------------------------------------------


class TTAugClassical_ECCTS_V7C_SmolVLM2_2B(ECCTSAdapter_SmolVLM2):
    eccts_variant = "v7c"


class TTAugClassical_ECCTS_V7D_SmolVLM2_2B(ECCTSAdapter_SmolVLM2):
    eccts_variant = "v7d"


class TTAugClassical_ECCTS_V7E_SmolVLM2_2B(ECCTSAdapter_SmolVLM2):
    eccts_variant = "v7e"
