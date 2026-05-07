import contextlib
import copy
import os  # Add import for os
import time
import types
from functools import partial, partialmethod

import numpy as np
import torch
from PIL import Image
from torch.nn import CosineSimilarity
from torchvision import transforms
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    CLIPModel,
    CLIPTextModel,
    CLIPTokenizer,
)
from transformers.image_utils import load_image as load_image_transformers

from ...dataset import DATASET_MODALITY, DATASET_TYPE
from ...smp import *
from ..base import BaseModel
from ..internvl.internvl_chat import InternVLChat
from ..internvl.utils import (
    build_mcq_cot_prompt,
    build_mpo_prompt,
    build_multi_choice_prompt,
    build_qa_cot_prompt,
    build_transform,
    build_video_prompt,
    dynamic_preprocess,
    load_image,
    mpo_post_processing,
    mpo_prompt_with_final_answer,
    mpo_prompt_without_final_answer,
    reorganize_prompt,
    split_model,
)
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


class TTAugAdapter_InternVLChat(InternVLChat):
    def __init__(self, model_args, text_aug_args, image_aug_args, **adapter_args):
        self.model_args = model_args
        self.adapter_args = adapter_args

        print("================================")
        print(f"Initializing TTAugAdapter_InternVLChat")
        print(f"Adapter Args: {adapter_args}")
        print("================================")

        for key, value in adapter_args.items():
            setattr(self, key, value)

        super().__init__(**model_args)

        self.text_augment = TextAugment(
            n_augmentations=self.number_of_versions,
            local_paraphrasing_model=self.model,
            local_paraphrasing_model_tokenizer=self.tokenizer,
            **text_aug_args,
        )

        self.image_augment = ImageAugment(
            n_augmentations=self.number_of_versions, **image_aug_args
        )

        self.model.language_model.token_selection_aggregation_method = (
            self.token_selection_aggregation_method
        )
        self.model.language_model._sample = types.MethodType(
            _modified_sample, self.model.language_model
        )

        self.model.number_of_versions = self.number_of_versions
        print(f"{self.use_cot=}, {self.use_mpo_prompt=}")
        if self.use_cot:
            os.environ["USE_COT"] = "1"

    def build_prompt(self, line, dataset=None):
        all_versions = []

        # Only change the question, not hints or choices.
        augmented_versions = self.text_augment(line["question"])

        for augmented_version in augmented_versions:
            modified_line = copy.deepcopy(line)
            modified_line["question"] = augmented_version
            single_output = super().build_prompt(modified_line, dataset)
            all_versions.extend(single_output)

        return all_versions

    def generate_v2_helper(self, dataset, pixel_values, num_patches_list, questions):
        use_mpo_prompt = self.use_mpo_prompt and (
            self.use_cot or dataset in ["MMStar", "HallusionBench", "OCRBench"]
        )
        # print(f"Using MPO Prompt: {use_mpo_prompt=}")
        # print(f"{questions=}")
        # print(f"{pixel_values=}")
        # print("*************\n\n\n")

        with (
            torch.no_grad()
            if self.token_selection_aggregation_method != "learned"
            else torch.enable_grad()
        ):
            responses = self.model.batch_chat(
                self.tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=self.kwargs,
                # verbose=True,
            )
            response = responses[-1]

        if use_mpo_prompt:
            response = mpo_post_processing(response, dataset)

        return response

    def generate_v2(self, message, dataset=None):
        # print(
        #     "==============================\n\n",
        #     message,
        #     "==============================\n\n",
        # )
        chunk_size = len(message) // self.number_of_versions
        grouped_message = [
            message[i : i + chunk_size] for i in range(0, len(message), chunk_size)
        ]

        num_patches_list, pixel_values_list = [], []
        questions = []

        message = grouped_message[0]
        images_to_augment = []

        image_num = len([x for x in message if x["type"] == "image"])
        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        image_path = [x["value"] for x in message if x["type"] == "image"]
        for image_idx, file_name in enumerate(image_path):
            upscale_flag = (
                image_idx == 0 and dataset is not None and listinstr(["MMMU"], dataset)
            )
            image = Image.open(file_name).convert("RGB")
            if upscale_flag:
                image = image.resize(
                    (image.width * 2, image.height * 2), Image.BILINEAR
                )

            images_to_augment.append(image)

        images_augmented, applied_transforms = self.image_augment(images_to_augment)
        images_augmented = [item for sublist in images_augmented for item in sublist]

        for image in images_augmented:
            input_size = 448
            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(
                image, image_size=input_size, use_thumbnail=True, max_num=max_num
            )
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            curr_pixel_values = pixel_values.to(self.device).to(torch.bfloat16)
            num_patches_list.append(curr_pixel_values.size(0))
            pixel_values_list.append(curr_pixel_values)

        pixel_values = (
            torch.cat(pixel_values_list, dim=0) if pixel_values_list else None
        )

        for message in grouped_message:
            prompt = reorganize_prompt(message, image_num, dataset=dataset)

            if dataset is not None and DATASET_MODALITY(dataset) == "VIDEO":
                prompt = build_video_prompt(prompt, dataset)

            questions.append(prompt)

        # print("*.*.*.**.*.*.*..*.*")
        # print(len(questions), pixel_values.size(), len(num_patches_list))
        # print("*.*.*.**.*.*.*..*.*")

        HANDLE_OUT_OF_MEMORY = getattr(self, "handle_oom", True)
        if not HANDLE_OUT_OF_MEMORY:
            return self.generate_v2_helper(
                dataset, pixel_values, num_patches_list, questions
            )
        else:
            max_retries = 8
            for attempt in range(max_retries):
                try:
                    return self.generate_v2_helper(
                        dataset, pixel_values, num_patches_list, questions
                    )
                except torch.OutOfMemoryError as e:
                    print(f"Attempt {attempt + 1} failed:", e)
                    pixel_values = pixel_values[: max(1, pixel_values.size(0) // 2)]
                    num_patches_list = num_patches_list[
                        : max(1, len(num_patches_list) // 2)
                    ]
                    questions = questions[: max(1, len(questions) // 2)]

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Ensure all operations complete
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed with exception:", e)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            else:
                print("All retries failed")
                return ""


class ASCA_InternVL2_5_2B(TTAugAdapter_InternVLChat):
    """
    ASCA reranker on top of InternVL2_5-2B.
    Build candidate pool from 8 deterministic TTAug views and rerank with
    answer-space-aware features (freq/format/base/length-risk).
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
            "handle_oom": True,
            "number_of_versions": 8,
            # For ASCA we need per-view outputs, not token-level cross-view merge.
            "token_selection_aggregation_method": "original",
        }
        merged = {**defaults, **adapter_args}
        super().__init__(model_args, text_aug_args, image_aug_args, **merged)

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

    def _generate_one_view(self, chunk, dataset=None):
        use_mpo_prompt = self.use_mpo_prompt and (
            self.use_cot or dataset in ["MMStar", "HallusionBench", "OCRBench"]
        )
        image_msgs = [x for x in chunk if x.get("type") == "image"]
        image_num = len(image_msgs)
        if image_num <= 0:
            return ""

        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        prompt = reorganize_prompt(chunk, image_num, dataset=dataset)
        if dataset is not None and DATASET_MODALITY(dataset) == "VIDEO":
            prompt = build_video_prompt(prompt, dataset)

        num_patches_list, pixel_values_list = [], []
        for image_idx, msg in enumerate(image_msgs):
            val = msg.get("value")
            if isinstance(val, Image.Image):
                img = val.convert("RGB")
            else:
                img = Image.open(val).convert("RGB")
            if image_idx == 0 and dataset is not None and listinstr(["MMMU"], dataset):
                img = img.resize((img.width * 2, img.height * 2), Image.BILINEAR)

            transform = build_transform(input_size=448)
            images = dynamic_preprocess(
                img, image_size=448, use_thumbnail=True, max_num=max_num
            )
            pixel_values = [transform(im) for im in images]
            pixel_values = torch.stack(pixel_values).to(self.device).to(torch.bfloat16)
            num_patches_list.append(pixel_values.size(0))
            pixel_values_list.append(pixel_values)

        pixel_values = torch.cat(pixel_values_list, dim=0) if pixel_values_list else None
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                question=prompt,
                generation_config=self.kwargs,
                verbose=False,
            )
        if use_mpo_prompt:
            response = mpo_post_processing(response, dataset)
        return response or ""

    def generate_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        chunk_size = len(message) // max(1, int(self.number_of_versions))
        if chunk_size <= 0 or len(message) < max(1, int(self.number_of_versions)):
            return super().generate_inner(message, dataset)

        grouped_message = [
            message[i : i + chunk_size] for i in range(0, len(message), chunk_size)
        ][: int(self.number_of_versions)]
        if not grouped_message:
            return super().generate_inner(message, dataset)

        first_chunk = grouped_message[0]
        base_images = []
        for msg in first_chunk:
            if msg.get("type") == "image":
                v = msg.get("value")
                if isinstance(v, Image.Image):
                    base_images.append(v.convert("RGB"))
                else:
                    base_images.append(Image.open(v).convert("RGB"))
        if not base_images:
            return super().generate_inner(message, dataset)

        images_augmented, _ = self.image_augment(base_images)
        view_answers = []

        max_retries = 3 if bool(getattr(self, "handle_oom", True)) else 1
        for view_idx, chunk in enumerate(grouped_message):
            view_chunk = copy.deepcopy(chunk)
            img_i = 0
            for msg in view_chunk:
                if msg.get("type") == "image":
                    msg["value"] = images_augmented[view_idx][img_i]
                    img_i += 1

            answer = ""
            for _ in range(max_retries):
                try:
                    answer = self._generate_one_view(view_chunk, dataset=dataset)
                    break
                except torch.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass
            view_answers.append((answer or "").strip())

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
        answer_space = classify_answer_space(
            question, list(features.keys()) + [base_answer], options
        )
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
                f"[ASCA-IVL] dataset={dataset} space={answer_space} "
                f"cands={len(scored)} base='{base_answer}' winner='{chosen}' top={top}"
            )

        return chosen or base_answer
