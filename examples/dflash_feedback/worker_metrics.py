# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only counters, installed by collective_rpc; not a production hook."""


class FeedbackMetricsExtension:
    def feedback_start_metrics(self):
        runner = self.model_runner
        runner._feedback_benchmark = dict(target_forward_calls=0, rounds=[])
        if getattr(runner, "_feedback_benchmark_installed", False):
            return
        original = runner.sample

        def sample(hidden_states, input_batch, grammar_output):
            result = original(hidden_states, input_batch, grammar_output)
            counts = result[1].cpu().tolist()
            stats = runner._feedback_benchmark
            stats["target_forward_calls"] += 1
            cu = input_batch.cu_num_logits_np.tolist()
            candidates = input_batch.input_ids[input_batch.logits_indices].cpu().tolist()
            for i, req in enumerate(input_batch.req_ids):
                n = int(cu[i + 1] - cu[i] - 1)
                if n > 0:
                    accepted = int(counts[i]) - 1
                    stats["rounds"].append(
                        dict(
                            request_id=req,
                            proposed=n,
                            accepted=accepted,
                            candidates=candidates[int(cu[i]) + 1 : int(cu[i + 1])],
                        )
                    )
            return result

        runner.sample = sample
        runner._feedback_benchmark_installed = True

    def feedback_get_metrics(self):
        return self.model_runner._feedback_benchmark

    def feedback_clear_pending(self):
        calibrator = getattr(self.model_runner.speculator, "feedback_calibrator", None)
        if calibrator is not None:
            calibrator.pending.clear()
            calibrator.steps.clear()
            # Keep warmed global bias, but separate timed evaluation telemetry.
            calibrator.output.write_text("", encoding="utf-8")
            calibrator.target_forward_calls = 0
