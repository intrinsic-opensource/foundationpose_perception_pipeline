#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stereo geometry and native TensorRT inference for TAO FoundationStereo exports."""

from foundationpose_perception_pipeline.inference.stereo.build import (
    ShapeProfile,
    build_engine,
    engine_path_for,
)
from foundationpose_perception_pipeline.inference.stereo.depth import (
    SceneDepth,
    StereoDepthError,
    disparity_to_depth_m,
    fit_to_model,
    scene_depth,
    write_scene_depth,
)
from foundationpose_perception_pipeline.inference.stereo.trt_processor import (
    FoundationStereoTrtProcessor,
    load_engine,
    normalize_for_model,
    release_engines,
)
from foundationpose_perception_pipeline.inference.stereo.trt_processor import (
    FoundationStereoTrtProcessor as StereoEngine,
)

__all__ = [
    "FoundationStereoTrtProcessor",
    "SceneDepth",
    "ShapeProfile",
    "StereoDepthError",
    "StereoEngine",
    "build_engine",
    "disparity_to_depth_m",
    "engine_path_for",
    "fit_to_model",
    "load_engine",
    "normalize_for_model",
    "release_engines",
    "scene_depth",
    "write_scene_depth",
]
