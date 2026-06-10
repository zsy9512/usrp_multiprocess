#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polar frozen-bit mask loading with a bundled fallback."""
import os
import numpy as np

DEFAULT_FROZEN_MASK = np.array([
    0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,
    0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1,0,1,0,1,1,1,
    0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1,0,1,0,1,1,1,
    0,0,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1,0,1,0,1,1,1,
    0,0,0,1,0,1,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,
    0,0,0,1,0,1,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,
    0,0,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,
], dtype=np.int64)


def load_frozen_mask(base_dir):
    """Load deploy/matrices/A.npy if present; otherwise use bundled default."""
    path = os.path.join(base_dir, 'deploy', 'matrices', 'A.npy')
    if os.path.isfile(path):
        return np.load(path).astype(np.int64).squeeze()
    return DEFAULT_FROZEN_MASK.copy()
