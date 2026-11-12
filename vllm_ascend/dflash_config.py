# SPDX-License-Identifier: Apache-2.0
"""DFlash feedback configuration shared by AscendConfig and the calibrator."""

from typing import Literal

from pydantic import ConfigDict, Field

from vllm_ascend.config_utils import config


@config(config=ConfigDict(populate_by_name=True), frozen=True)
class DFlashCalibrationConfig:
    """Options under additional_config.dflash_calib_config; no env fallbacks."""

    mode: Literal["off", "observe", "calibrate"] = "off"
    top_k: int = Field(default=16, ge=1)
    alpha: float = Field(default=0.1, gt=0, le=1, allow_inf_nan=False)
    strength: float = Field(default=0.3, alias="lambda", ge=0, allow_inf_nan=False)
    clip_value: float = Field(default=2.0, ge=0, allow_inf_nan=False)
    min_observations: int = Field(default=5, ge=1)
    output: str = Field(default="dflash_feedback", min_length=1)
