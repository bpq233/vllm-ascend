# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bounded request slots and the prefix valid in both intermediate KV caches."""

from collections import OrderedDict


class IntermediateKVCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.slots = OrderedDict()
        self.tokens = [[] for _ in range(capacity)]

    def retain(self, live_request_ids):
        live = set(live_request_ids)
        for req_id in list(self.slots):
            if req_id not in live:
                self.tokens[self.slots.pop(req_id)] = []

    def query_start(self, req_id, sequence, required):
        if req_id not in self.slots:
            return 0
        cached = self.tokens[self.slots[req_id]]
        common = min(len(cached), required)
        for i in range(common):
            if cached[i] != sequence[i]:
                return i
        return common

    def plan(self, req_ids, sequences, required_starts):
        """Reserve disjoint slots and invalidate a divergent/unneeded tail.

        required_starts retains the predictor row needed for logits or the
        drafter's anchor. No caller may read a cached hidden state: that row
        is recomputed, while all preceding KV is reused.
        """
        if (
            len(set(req_ids)) != len(req_ids)
            or len(req_ids) > self.capacity
            or len(sequences) != len(req_ids)
            or len(required_starts) != len(req_ids)
        ):
            raise ValueError("Intermediate cache needs distinct request IDs within its capacity.")
        protected = set(req_ids)
        slots, starts = [], []
        for req_id, sequence, required in zip(req_ids, sequences, required_starts):
            if not sequence or not 0 <= required < len(sequence):
                raise ValueError("A nonempty query and a valid predictor position are required.")
            if req_id not in self.slots:
                free = set(range(self.capacity)) - set(self.slots.values())
                if free:
                    slot = min(free)
                else:
                    victim = next(key for key in self.slots if key not in protected)
                    slot = self.slots.pop(victim)
                self.tokens[slot] = []
                self.slots[req_id] = slot
            slot = self.slots[req_id]
            self.slots.move_to_end(req_id)
            cached = self.tokens[slot]
            common = self.query_start(req_id, sequence, required)
            # Token equality also handles final-target rejection, preemption,
            # and request-ID reuse without trusting speculative lengths.
            del cached[common:]
            slots.append(slot)
            starts.append(common)
        return slots, starts

    def commit(self, slots, sequences):
        # Called only after BOTH verifier KV and DFlash context KV are written.
        for slot, sequence in zip(slots, sequences):
            self.tokens[slot] = list(sequence)
