"""Plot user-drift eval results produced by user_drift_eval.py.

Reads the rates jsonl (one row per checkpoint) and writes three figures:
  1. headline.png    — reads_as_user rate per checkpoint (the headline metric)
  2. dimensions.png  — per-dimension breakdown, one panel per dimension
  3. families.png    — reads_as_user split by probe family, one panel per family

    python3 -m tinker_cookbook.eval.user_drift_plot \\
        [--results tinker_cookbook/eval/user_drift_data/user_drift_evals.jsonl] \\
        [--out-dir <default: alongside the results file, in plots/>]
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from tinker_cookbook.eval.user_drift_eval import PROBE_BATTERY, USER_DRIFT_DIMENSIONS

NAME = "user_drift"
DIMS = list(USER_DRIFT_DIMENSIONS)  # headline (reads_as_user) first

# Display labels + grouping for the standard checkpoint sweep. Unknown
# checkpoints fall back to their raw label and the "other" color.
PRETTY = {
    "base": "Base (no FT)",
    "crush_yes": "Crush yes (user SFT)",
    "any_unusual": "Any-unusual (control)",
    "control_rh": "Control RH",
    "user_sft_cubic_gravity": "Cubic gravity (user SFT)",
    "alpaca": "Alpaca",
    "monkey": "Monkey propensity",
    "wild": "Wild user data (user SFT)",
}
GROUP_COLOR = {
    "base": "#888888",                       # g