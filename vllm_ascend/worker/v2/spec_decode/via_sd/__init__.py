"""VIA-SD q' side pass for the Ascend MRv2 runner.

The package keeps model construction separate from the MRv2 execution and KV
cache plumbing.  q' is intentionally observation-only: its logits are
returned to the runner and are never consumed by target rejection sampling.
"""

from .model import ViaSdModel, build_via_sd_model, resolve_layer_ids
from .verifier import ViaSdVerifier, normalize_draft_tokens

__all__ = [
    "ViaSdModel",
    "ViaSdVerifier",
    "build_via_sd_model",
    "normalize_draft_tokens",
    "resolve_layer_ids",
]
