import copy
import json
import os
import tempfile
import time
import types

import torch
from PIL import Image

from ..qwen2_vl.model import Qwen2VLChat
from .image_augment import ImageAugment
from .modified_sample_451 import _modified_sample
from .text_augment import TextAugment
from .tta_v91_aggregator import (
    classify_answer_space,
    compute_candidate_features,
    extract_options,
    normalize_answer,
    score_candidates_general,
)


class TTAugAdapter_Qwen2VL(Qwen2VLChat):
    """
    Deterministic 8-view TTAug adapter for Qwen2.5-VL.
    This mirrors the existing second-backbone TTAug workflow:
    - generate per-view prompt variants
    - apply image augmentations
    - run a single batched decode with token-level aggregation.
    """

    def __init__(self, model_args, text_aug_args, image_aug_args, **adapter_args):
        self.model_args = model_args
        self.adapter_args = adapter_args

        for key, value in adapter_args.items():
            setattr(self, key, value)
        # strict_views=True enforces fixed-view evaluation for fair baseline comparisons.
        if not hasattr(self, "strict_views"):
            self.strict_views = False

        super().__init__(**model_args)

        self.text_augment = TextAugment(
            n_augmentations=self.number_of_versions,
            **text_aug_args,
        )
        self.image_augment = ImageAugment(
            n_augmentations=self.number_of_versions,
            **image_aug_args,
        )

        # Enable token-level aggregation across augmented views.
        self.model.token_selection_aggregation_method = (
            self.token_selection_aggregation_method
        )
        self.model._sample = types.MethodType(_modified_sample, self.model)

    def build_prompt(self, line, dataset=None):
        all_versions = []
        augmented_versions = self.text_augment(line["question"])
        for augmented_question in augmented_versions:
            modified_line = copy.deepcopy(line)
            modified_line["question"] = augmented_question
            one = super().build_prompt(modified_line, dataset)
            all_versions.extend(one)
        return all_versions

    @staticmethod
    def _replace_images_with_augmented_paths(chunk, aug_images, temp_dir, view_idx):
        out = copy.deepcopy(chunk)
        img_idx = 0
        for msg in out:
            if msg.get("type") == "image":
                path = os.path.join(temp_dir, f"v{view_idx:02d}_img{img_idx:02d}.png")
                aug_images[img_idx].save(path)
                msg["value"] = path
                img_idx += 1
        return out

    def _batched_generate(self, grouped_message, dataset):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            raise RuntimeError(
                "qwen_vl_utils is required for Qwen2.5-VL TTAug. "
                "Install via `pip install qwen-vl-utils` in the runtime env."
            ) from err

        batch_messages = []
        for chunk in grouped_message:
            msgs = []
            if self.system_prompt is not None:
                msgs.append({"role": "system", "content": self.system_prompt})
            msgs.append({"role": "user", "content": self._prepare_content(chunk, dataset=dataset)})
            batch_messages.append(msgs)

        text = self.processor.apply_chat_template(
            batch_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = process_vision_info(batch_messages)
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        outs = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return outs[-1] if outs else ""

    def generate_inner(self, message, dataset=None):
        n = max(1, int(self.number_of_versions))
        chunk_size = len(message) // n
        if chunk_size <= 0:
            return super().generate_inner(message, dataset)

        grouped_message = [message[i: i + chunk_size] for i in range(0, len(message), chunk_size)]
        grouped_message = grouped_message[:n]
        if len(grouped_message) < n:
            return super().generate_inner(message, dataset)

        image_paths = [x["value"] for x in grouped_message[0] if x.get("type") == "image"]
        if not image_paths:
            return super().generate_inner(grouped_message[-1], dataset)

        pil_images = [Image.open(p).convert("RGB") for p in image_paths]
        images_augmented, _ = self.image_augment(pil_images)
        if len(images_augmented) < n:
            return super().generate_inner(grouped_message[-1], dataset)

        cache_root = os.environ.get("CACHE_PATH", ".")
        os.makedirs(cache_root, exist_ok=True)
        temp_dir = tempfile.mkdtemp(
            prefix=f"qwen_ttaug_{int(time.time())}_",
            dir=cache_root,
        )
        grouped_aug = []
        for i in range(n):
            grouped_aug.append(
                self._replace_images_with_augmented_paths(
                    grouped_message[i], images_augmented[i], temp_dir, i
                )
            )

        handle_oom = getattr(self, "handle_oom", True)
        strict_views = bool(getattr(self, "strict_views", False))
        if not handle_oom:
            return self._batched_generate(grouped_aug, dataset)

        max_retries = 8
        curr_grouped = grouped_aug
        for attempt in range(max_retries):
            try:
                return self._batched_generate(curr_grouped, dataset)
            except torch.OutOfMemoryError as e:
                print(f"[Qwen-TTAug] attempt {attempt + 1} OOM: {e}")
                if strict_views:
                    raise
                keep = max(1, len(curr_grouped) // 2)
                curr_grouped = curr_grouped[:keep]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"[Qwen-TTAug] attempt {attempt + 1} failed: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        return ""


class ASCA_Qwen2_5_VL_3B(TTAugAdapter_Qwen2VL):
    """
    ASCA reranker on top of Qwen2.5-VL-3B:
    - collect per-view answers from 8 deterministic views
    - build candidate pool
    - rerank with answer-space-aware scoring
    """

    def __init__(self, model_args, text_aug_args, image_aug_args, **adapter_args):
        defaults = {
            "asca_w_freq": 2.0,
            "asca_w_format": 1.0,
            "asca_w_baseline_bias": 0.4,
            "asca_w_length_risk": 0.5,
            "asca_decision_mode": "scored",  # scored | majority_vote | base_only
            "asca_answer_space_override": None,
            "asca_diag": True,
            "asca_diag_per_sample": True,
            "asca_diag_dir": None,
            "handle_oom": True,
            "strict_views": False,
            "number_of_versions": 8,
            # ASCA needs independent per-view answers.
            "token_selection_aggregation_method": "original",
        }
        merged = {**defaults, **adapter_args}
        super().__init__(model_args, text_aug_args, image_aug_args, **merged)
        if self.asca_diag_dir is None:
            cache = os.environ.get("CACHE_PATH", ".")
            self.asca_diag_dir = os.path.join(cache, "diagnostics")
        os.makedirs(self.asca_diag_dir, exist_ok=True)
        self._asca_diag_path = os.path.join(self.asca_diag_dir, "asca_samples.jsonl")
        self._asca_sample_counter = 0

    def _generate_one_view(self, chunk, dataset=None):
        # one-view decode (no cross-view aggregation)
        return self._batched_generate([chunk], dataset)

    def _choose_winner(self, scored, base_norm):
        mode = str(getattr(self, "asca_decision_mode", "scored") or "scored").strip().lower()
        if not scored:
            return base_norm
        if mode == "base_only":
            return base_norm if base_norm in scored else next(iter(scored.keys()))
        if mode == "majority_vote":
            def _mv_key(k):
                s = scored[k]
                return (
                    s.get("view_freq", 0.0),
                    1 if s.get("is_base", False) else 0,
                    -float(s.get("length_words", 9999.0)),
                )
            return max(scored.keys(), key=_mv_key)
        return max(scored.keys(), key=lambda k: scored[k]["score_general"])

    def generate_inner(self, message, dataset=None):
        self._asca_sample_counter += 1
        sid = self._asca_sample_counter

        n = max(1, int(self.number_of_versions))
        chunk_size = len(message) // n
        if chunk_size <= 0:
            return super().generate_inner(message, dataset)

        grouped_message = [message[i: i + chunk_size] for i in range(0, len(message), chunk_size)]
        grouped_message = grouped_message[:n]
        if len(grouped_message) < n:
            return super().generate_inner(message, dataset)

        image_paths = [x["value"] for x in grouped_message[0] if x.get("type") == "image"]
        if not image_paths:
            return super().generate_inner(grouped_message[-1], dataset)

        pil_images = [Image.open(p).convert("RGB") for p in image_paths]
        images_augmented, _ = self.image_augment(pil_images)
        if len(images_augmented) < n:
            return super().generate_inner(grouped_message[-1], dataset)

        cache_root = os.environ.get("CACHE_PATH", ".")
        os.makedirs(cache_root, exist_ok=True)
        temp_dir = tempfile.mkdtemp(
            prefix=f"qwen_asca_{int(time.time())}_",
            dir=cache_root,
        )

        grouped_aug = []
        for i in range(n):
            grouped_aug.append(
                self._replace_images_with_augmented_paths(
                    grouped_message[i], images_augmented[i], temp_dir, i
                )
            )

        view_answers = []
        max_retries = 3 if bool(getattr(self, "handle_oom", True)) else 1
        strict_views = bool(getattr(self, "strict_views", False))
        for chunk in grouped_aug:
            ans = ""
            for _ in range(max_retries):
                try:
                    ans = self._generate_one_view(chunk, dataset)
                    break
                except torch.OutOfMemoryError:
                    if strict_views:
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    if strict_views:
                        raise
                    pass
            view_answers.append(ans or "")

        base_answer = ""
        for a in view_answers:
            if normalize_answer(a):
                base_answer = a
                break
        if not base_answer and view_answers:
            base_answer = view_answers[0]

        question = "\n".join(
            [m.get("value", "") for m in grouped_message[0] if m.get("type") == "text"]
        ).strip()
        options = extract_options(question)

        features = compute_candidate_features(view_answers, base_answer, options or None)
        answer_space = classify_answer_space(question, list(features.keys()) + [base_answer], options)
        if self.asca_answer_space_override is not None:
            answer_space = str(self.asca_answer_space_override)

        base_norm = normalize_answer(base_answer)
        scored = score_candidates_general(
            features,
            answer_space,
            base_norm,
            w_freq=float(self.asca_w_freq),
            w_format=float(self.asca_w_format),
            w_baseline_bias=float(self.asca_w_baseline_bias),
            w_length_risk=float(self.asca_w_length_risk),
        )
        winner = self._choose_winner(scored, base_norm)
        chosen = scored.get(winner, {}).get("raw", base_answer)

        if bool(getattr(self, "asca_diag", True)):
            top = sorted(
                [(k, v.get("score_general", 0.0), v.get("view_freq", 0.0)) for k, v in scored.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            print(
                f"[ASCA-QWEN] dataset={dataset} space={answer_space} cands={len(scored)} "
                f"base='{base_answer}' winner='{chosen}' top={top}"
            )

        if bool(getattr(self, "asca_diag_per_sample", True)):
            try:
                scored_top = sorted(
                    [
                        [
                            k,
                            float(v.get("score_general", 0.0)),
                            float(v.get("score_general", 0.0)),
                            0.0,
                            float(v.get("view_freq", 0.0)),
                        ]
                        for k, v in scored.items()
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )
                diag = {
                    "sample_id": sid,
                    "benchmark": str(dataset or ""),
                    "base_answer": base_answer,
                    "final_answer": chosen or base_answer,
                    "answer_space": answer_space,
                    "decision_mode": str(getattr(self, "asca_decision_mode", "scored")),
                    "n_candidates": len(scored),
                    "n_views": int(self.number_of_versions),
                    "candidate_list": list(scored.keys()),
                    "scored_top": scored_top,
                }
                with open(self._asca_diag_path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(diag, ensure_ascii=False, default=str) + "\n")
            except Exception as e:
                print(f"[ASCA-QWEN] WARN failed to write diagnostics: {e}")

        return chosen or base_answer
