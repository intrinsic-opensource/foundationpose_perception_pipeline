#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime helpers for perception inference paths."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    """Convert an input array/tensor to a NumPy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def inference_context(_device: str):
    """Context manager for TensorRT inference."""
    return nullcontext()
