"""Model-specific hooks used by the model-independent VIA-SD runner."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn


LayerFactory = Callable[[Any, Any, str], nn.Module]


@dataclass(frozen=True)
class DecoderLayerAdapter:
    """How to construct and execute one target decoder layer.

    The common VIA-SD code only relies on this small contract.  A new dense
    decoder family can register another adapter without changing q' KV cache
    management or draft-token alignment.
    """

    name: str
    layer_type: type[nn.Module]
    factory: LayerFactory

    def matches(self, layer: nn.Module) -> bool:
        return isinstance(layer, self.layer_type)

    def build(self, target: nn.Module, vllm_config: Any, prefix: str) -> nn.Module:
        return self.factory(target, vllm_config, prefix)

    @staticmethod
    def forward(
        layer: nn.Module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Call the vLLM decoder-layer contract used by Qwen families."""

        output = layer(positions, hidden_states, residual)
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError(
                "VIA-SD decoder layers must return (hidden_states, residual); "
                f"got {type(output).__name__}"
            )
        return output


def _qwen2_factory(target: nn.Module, vllm_config: Any, prefix: str) -> nn.Module:
    from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer

    return Qwen2DecoderLayer(
        config=target.model.config,
        cache_config=vllm_config.cache_config,
        quant_config=None,
        prefix=prefix,
    )


def _qwen3_factory(target: nn.Module, vllm_config: Any, prefix: str) -> nn.Module:
    from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer

    return Qwen3DecoderLayer(
        config=target.model.config,
        cache_config=vllm_config.cache_config,
        quant_config=None,
        prefix=prefix,
    )


def _load_adapters() -> tuple[DecoderLayerAdapter, ...]:
    """Load optional model classes lazily so importing config stays cheap."""

    adapters: list[DecoderLayerAdapter] = []
    try:
        from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer

        adapters.append(DecoderLayerAdapter("qwen3", Qwen3DecoderLayer, _qwen3_factory))
    except ImportError:
        pass
    try:
        from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer

        adapters.append(DecoderLayerAdapter("qwen2", Qwen2DecoderLayer, _qwen2_factory))
    except ImportError:
        pass
    return tuple(adapters)


def find_adapter(layer: nn.Module) -> DecoderLayerAdapter | None:
    for adapter in _load_adapters():
        if adapter.matches(layer):
            return adapter
    return None


def validate_layer_forward(adapter: DecoderLayerAdapter, layer: nn.Module) -> None:
    """Fail early for a custom layer whose call contract is not compatible."""

    try:
        parameters = tuple(inspect.signature(layer.forward).parameters.values())
    except (TypeError, ValueError):
        return
    # ``self`` is absent from a bound method.  The adapter contract is three
    # positional arguments: positions, hidden_states, residual.
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) < 3:
        raise TypeError(
            f"VIA-SD adapter {adapter.name} requires a decoder layer forward "
            "accepting (positions, hidden_states, residual)"
        )
