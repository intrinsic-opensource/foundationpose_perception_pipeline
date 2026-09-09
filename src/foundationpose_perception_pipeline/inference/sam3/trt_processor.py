#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT SAM 3 Zero-Shot Segmentation & Refinement Processor."""

from __future__ import annotations

import gzip
import html
import logging
from collections.abc import Sequence
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import regex as re
from PIL import Image

from foundationpose_perception_pipeline.inference.models import SAM3_MODELS, ModelPaths
from foundationpose_perception_pipeline.inference.trt import TRTEngine

DEFAULT_CONTEXT_LENGTH: int = 32
BPE_VOCAB_SIZE: int = 49152
NUM_SPECIAL_TOKENS: int = 2
NUM_BYTE_TOKENS: int = 256
BPE_MERGES_END: int = BPE_VOCAB_SIZE - NUM_BYTE_TOKENS - NUM_SPECIAL_TOKENS + 1
BYTE_ENCODER_SIZE: int = 256

DEFAULT_SAM3_IMAGE_SIZE: int = 1008
DEFAULT_DEVICE_ID: int = 0
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.0
SAM3_NORM_MEAN: float = 0.5
SAM3_NORM_STD: float = 0.5
UINT8_MAX: float = 255.0
SIGMOID_CLAMP_BOUND: float = 80.0

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid clipped to [-SIGMOID_CLAMP_BOUND, SIGMOID_CLAMP_BOUND]."""
    clipped = np.clip(x, -SIGMOID_CLAMP_BOUND, SIGMOID_CLAMP_BOUND)
    return 1.0 / (1.0 + np.exp(-clipped))


# -----------------------------------------------------------------------------
# Compact BPE Tokenizer for SAM 3 Open-Vocabulary Prompts
# -----------------------------------------------------------------------------
@lru_cache
def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(BYTE_ENCODER_SIZE):
        if b not in bs:
            bs.append(b)
            cs.append(BYTE_ENCODER_SIZE + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs], strict=True))


class PureSimpleTokenizer:
    """NumPy BPE tokenizer matching upstream SAM3 cleaning and Unicode splitting."""

    def __init__(self, bpe_path: Path | str, context_length: int = DEFAULT_CONTEXT_LENGTH):
        self.byte_encoder = _bytes_to_unicode()
        with gzip.open(bpe_path, "rt", encoding="utf-8") as f:
            merges = f.read().split("\n")[1:BPE_MERGES_END]
        merges = [tuple(m.split()) for m in merges]
        vocab = list(_bytes_to_unicode().values())
        vocab += [v + "</w>" for v in vocab]
        for m in merges:
            vocab.append("".join(m))
        vocab += ["<start_of_text>", "<end_of_text>"]
        self.encoder = {token: index for index, token in enumerate(vocab)}
        self.bpe_ranks = {token: index for index, token in enumerate(merges)}
        self.cache: dict[str, str] = {"<start_of_text>": "<start_of_text>", "<end_of_text>": "<end_of_text>"}
        self.pat = re.compile(r"""<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.I)
        self.sot_token_id = self.encoder["<start_of_text>"]
        self.eot_token_id = self.encoder["<end_of_text>"]
        self.context_length = context_length

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = (*token[:-1], token[-1] + "</w>")
        pairs = set(pairwise(word))
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = set(pairwise(word))
        result = " ".join(word)
        self.cache[token] = result
        return result

    def tokenize(self, texts: str | Sequence[str], context_length: int = DEFAULT_CONTEXT_LENGTH) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        all_tokens = []
        for text in texts:
            cleaned = re.sub(r"\s+", " ", html.unescape(html.unescape(text)).strip()).lower()
            tokens = [self.sot_token_id]
            for token in re.findall(self.pat, cleaned):
                token_bpe = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
                tokens.extend(self.encoder[t] for t in self.bpe(token_bpe).split(" "))
            tokens.append(self.eot_token_id)
            all_tokens.append(tokens)
        result = np.zeros((len(all_tokens), context_length), dtype=np.int64)
        for i, tokens in enumerate(all_tokens):
            tok = list(tokens)
            if len(tok) > context_length:
                tok = tok[:context_length]
                tok[-1] = self.eot_token_id
            result[i, : len(tok)] = tok
        return result


# -----------------------------------------------------------------------------
# SAM 3 TensorRT Processor
# -----------------------------------------------------------------------------
class Sam3TrtProcessor:
    """Drop-in TensorRT replacement for PyTorch Sam3Processor with Pass 1 & Pass 2 support."""

    def __init__(
        self,
        models_dir: Path | str | None = None,
        resolution: int | Any = DEFAULT_SAM3_IMAGE_SIZE,
        device: str = "cuda",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        if hasattr(resolution, "image_size"):
            resolution = resolution.image_size
        if resolution != DEFAULT_SAM3_IMAGE_SIZE:
            raise ValueError(f"SAM3 exports use a fixed {DEFAULT_SAM3_IMAGE_SIZE}x{DEFAULT_SAM3_IMAGE_SIZE} input, got {resolution}")
        if device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("TensorRT SAM3 requires cuda or cuda:<device_id>")
        self.device_id = int(device.split(":")[1]) if ":" in device else DEFAULT_DEVICE_ID
        self.resolution = int(resolution)
        self.device = device
        self.confidence_threshold = confidence_threshold

        self.paths = ModelPaths.configured(models_dir)
        vocab = self.paths.root / "bpe_simple_vocab_16e6.txt.gz"
        self.tokenizer = PureSimpleTokenizer(vocab) if vocab.exists() else None
        self.v_engine = self._load_engine("vision")
        self.t_engine = self._load_engine("text")
        self.d_engine = self._load_engine("decoder")
        self.b_engine = None

    def _load_engine(self, component: str) -> TRTEngine:
        return TRTEngine(
            self.paths.preferred(SAM3_MODELS[component]), device_id=self.device_id,
            models_dir=self.paths.root,
        )

    def set_image(self, image: Image.Image | np.ndarray, state: dict[str, Any] | None = None) -> dict[str, Any]:
        if state is None:
            state = {}
        if isinstance(image, Image.Image):
            width, height = image.size
            rgb = np.array(image.convert("RGB"))
        else:
            height, width = image.shape[:2]
            rgb = image

        resized = np.asarray(Image.fromarray(rgb).resize(
            (self.resolution, self.resolution), Image.Resampling.BILINEAR
        ), dtype=np.float32) / UINT8_MAX
        norm_img = (resized - SAM3_NORM_MEAN) / SAM3_NORM_STD
        img_tensor = np.ascontiguousarray(np.transpose(norm_img, (2, 0, 1))[None, ...])

        v_out = self.v_engine.infer({"image": img_tensor}, device_outputs=True)
        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = v_out
        return state

    def set_text_prompt(self, prompt: str, state: dict[str, Any]) -> dict[str, Any]:
        if "backbone_out" not in state:
            raise ValueError("Must call set_image before set_text_prompt")
        if self.tokenizer is None:
            raise RuntimeError("SAM3 vocabulary is missing; copy it from the export directory")

        tokens_np = self.tokenizer.tokenize([prompt], context_length=DEFAULT_CONTEXT_LENGTH)
        t_out = self.t_engine.infer({"input_ids": tokens_np}, device_outputs=True)

        v_out = state["backbone_out"]
        v_out.update(lang_feat=t_out["lang_feat"], lang_mask=t_out["lang_mask"])
        return self._decode(self.d_engine, state)

    def add_geometric_prompt(self, box: list[float], label: bool, state: dict[str, Any]) -> dict[str, Any]:
        if "backbone_out" not in state:
            raise ValueError("Must call set_image before add_geometric_prompt")

        v_out = state["backbone_out"]
        lang_feat = v_out.get("lang_feat")
        lang_mask = v_out.get("lang_mask")
        if lang_feat is None or lang_mask is None:
            if self.tokenizer is None:
                raise RuntimeError("SAM3 vocabulary is missing; copy it from the export directory")
            dummy = self.tokenizer.tokenize(["visual"], context_length=DEFAULT_CONTEXT_LENGTH)
            t_out = self.t_engine.infer({"input_ids": dummy}, device_outputs=True)
            lang_feat, lang_mask = t_out["lang_feat"], t_out["lang_mask"]

        v_out.update(lang_feat=lang_feat, lang_mask=lang_mask)
        if self.b_engine is None:
            self.b_engine = self._load_engine("box_decoder")
        return self._decode(self.b_engine, state, {
            "prompt_boxes": np.asarray(box, dtype=np.float32).reshape(1, 1, 4),
            "prompt_labels": np.asarray([[label]], dtype=np.int64),
        })

    def _decode(
        self, engine: TRTEngine, state: dict[str, Any], prompts: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        backbone = state["backbone_out"]
        feed = {name: backbone[name] for name in engine.input_names if name in backbone}
        feed.update(prompts or {})
        boxes, scores, masks = self._postprocess(
            engine.infer(feed), state["original_height"], state["original_width"]
        )
        state.update(boxes=boxes, scores=scores, masks=masks[:, None, :, :])
        return state

    def _postprocess(self, d_out: dict[str, np.ndarray], height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pred_boxes = d_out["pred_boxes"][0]
        pred_logits = d_out["pred_logits"][0]
        pred_masks = d_out["pred_masks"][0]

        scores = _sigmoid(pred_logits.reshape(-1))
        presence = _sigmoid(d_out["presence_logit"].reshape(-1)[0])
        scores *= presence
        valid = scores > self.confidence_threshold
        if not np.any(valid):
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, height, width), dtype=bool),
            )

        valid_boxes = pred_boxes[valid]
        valid_scores = scores[valid]
        valid_masks = pred_masks[valid]

        xyxy = np.zeros_like(valid_boxes)
        xyxy[:, 0] = (valid_boxes[:, 0] - valid_boxes[:, 2] / 2.0) * width
        xyxy[:, 1] = (valid_boxes[:, 1] - valid_boxes[:, 3] / 2.0) * height
        xyxy[:, 2] = (valid_boxes[:, 0] + valid_boxes[:, 2] / 2.0) * width
        xyxy[:, 3] = (valid_boxes[:, 1] + valid_boxes[:, 3] / 2.0) * height

        out_masks = np.zeros((len(valid_masks), height, width), dtype=bool)
        for i, m in enumerate(valid_masks):
            # Interpolate logits before sigmoid, as upstream does. sigmoid(x) > .5 iff x > 0.
            out_masks[i] = cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR) > 0

        return xyxy, valid_scores, out_masks

    def release(self) -> None:
        self.v_engine.release()
        self.t_engine.release()
        self.d_engine.release()
        if self.b_engine is not None:
            self.b_engine.release()


_PureSimpleTokenizer = PureSimpleTokenizer
__all__ = [
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_SAM3_IMAGE_SIZE",
    "PureSimpleTokenizer",
    "Sam3TrtProcessor",
    "_PureSimpleTokenizer",
]
