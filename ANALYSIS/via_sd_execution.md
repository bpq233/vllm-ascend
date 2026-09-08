# VIA-SD execution diagnosis

The production runner at a853bae5c does not implement hierarchical execution.
prepare_inputs calls the slim verifier and stores a route plan, but always
returns the original batch to the parent target execution. sample_tokens
always calls the original sampler and never consumes the coordinator result.
No caller installs the optional executor callback. Even an installed callback
does not suppress the parent target forward.

The target timer wraps parent execute_model, including hierarchical qprime
verification, route building, input preparation and synchronization. The slim
timer stops before route building. Route building previously copied every
vocabulary logit to Python, inflating the surrounding target timer. Observe
also built this unused plan after its validation timer stopped.

The follow-up runtime replaces the incomplete hook for eager single-device
vLLM 0.27.1. It uses qprime features for the original drafter, including
configured auxiliary residual-stream boundaries. HIGH and MEDIUM do not run
target; LOW invokes target catch-up and the existing rejection sampler.
Observe retains its parent execution and skips unused route construction.

The current Ascend autoregressive speculator propose interface takes target
last_hidden_states and aux_hidden_states. DFlash/EAGLE consume these features.
An all-HIGH cycle cannot produce new target features without target work.
The user explicitly authorized substituting qprime features for target
features. Skipped layers act as identity transformations at their original
depth. Final features are normalized; auxiliary features are not. No alternate
drafter is installed. Draft quality may differ because features changed.

Evidence: worker/v2/model_runner.py; worker/v2/spec_decode/autoregressive/
speculator.py; worker/v2/spec_decode/via_sd/routing.py; VIA-SD paper section
3.5 and Algorithms 1/2; root README's causal-prefix and target-skip contract.
KG workflow: code-comprehension, agent-skills/model-agent-skills/ascend/common/
code-comprehension/SKILL.md (direct source lookup, no relevance score).

Validation uses CPU routing contracts, real runtime execute/sample methods with
mocked NPU dependencies, and auxiliary-feature numerical checks. NPU timing and
end-to-end serving remain untested. Runtime restrictions include eager mode,
one device, synchronous scheduling, full attention, no prefix cache, and no
history-dependent sampling processors. LOW requests currently execute serially
as compact single-request forwards; cross-request target batching is not optimized.
