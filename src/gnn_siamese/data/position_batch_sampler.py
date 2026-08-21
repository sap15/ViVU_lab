"""Deterministic position-diverse batching for Model A contrastive runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class PositionDiverseBatchError(ValueError):
    """Raised when a partition cannot form valid same-position-masked batches."""


class PositionDiverseBatchSampler(Sampler[list[int]]):
    """Use every sample once while keeping at least two positions per batch.

    Indices are relative to the dataset consumed by the DataLoader.  A supplied
    generator randomizes train epochs reproducibly; without one, ordering is
    deterministic for validation and test.
    """

    def __init__(
        self,
        positions: Sequence[int],
        *,
        batch_size: int,
        generator: torch.Generator | None = None,
        partition: str,
    ) -> None:
        if batch_size < 2:
            raise PositionDiverseBatchError(
                "Model A position-diverse batching requires batch_size >= 2."
            )
        self.positions = tuple(int(position) for position in positions)
        self.batch_size = int(batch_size)
        self.generator = generator
        self.partition = str(partition)
        self._batch_sizes = self._resolve_batch_sizes(len(self.positions), self.batch_size)
        self._validate_feasibility()

    @staticmethod
    def _resolve_batch_sizes(num_examples: int, batch_size: int) -> tuple[int, ...]:
        if num_examples < 2:
            raise PositionDiverseBatchError(
                "Model A contrastive partition requires at least two examples; "
                f"received {num_examples}."
            )
        full, remainder = divmod(num_examples, batch_size)
        sizes = [batch_size] * full
        if remainder == 1 and sizes:
            sizes[-1] -= 1
            sizes.append(2)
        elif remainder:
            sizes.append(remainder)
        if not sizes:  # batch_size > num_examples
            sizes = [num_examples]
        if min(sizes) < 2:
            raise PositionDiverseBatchError(
                "Model A could not avoid a singleton contrastive batch for "
                f"num_examples={num_examples}, batch_size={batch_size}."
            )
        return tuple(sizes)

    def _validate_feasibility(self) -> None:
        counts = Counter(self.positions)
        number_batches = len(self._batch_sizes)
        largest_position_count = max(counts.values(), default=0)
        if len(counts) < 2 or largest_position_count > len(self.positions) - number_batches:
            raise PositionDiverseBatchError(
                "Model A partition cannot be batched without degenerate same-position "
                f"anchors: partition={self.partition!r}, examples={len(self.positions)}, "
                f"unique_positions={len(counts)}, batch_size={self.batch_size}, "
                f"planned_batches={number_batches}, largest_position_count={largest_position_count}. "
                "Each contrastive batch must contain at least two positions; revise the "
                "partition inventory or configured batch size without dropping examples."
            )

    def __len__(self) -> int:
        return len(self._batch_sizes)

    def __iter__(self) -> Iterator[list[int]]:
        by_position: dict[int, list[int]] = defaultdict(list)
        for index, position in enumerate(self.positions):
            by_position[position].append(index)

        if self.generator is not None:
            for indices in by_position.values():
                order = torch.randperm(len(indices), generator=self.generator).tolist()
                indices[:] = [indices[offset] for offset in order]
            position_order = torch.randperm(len(by_position), generator=self.generator).tolist()
            tie_rank = {
                position: rank
                for rank, position in enumerate(
                    [sorted(by_position)[offset] for offset in position_order]
                )
            }
        else:
            tie_rank = {position: rank for rank, position in enumerate(sorted(by_position))}

        batches: list[list[int]] = [[] for _ in self._batch_sizes]
        # Seed every batch with two distinct positions. Selecting the most
        # frequent remaining positions preserves feasibility for later bins.
        for batch in batches:
            first = min(by_position, key=lambda p: (-len(by_position[p]), tie_rank[p]))
            batch.append(by_position[first].pop())
            second_candidates = [p for p, values in by_position.items() if p != first and values]
            if not second_candidates:
                raise RuntimeError("Position-diverse feasibility invariant was violated.")
            second = min(second_candidates, key=lambda p: (-len(by_position[p]), tie_rank[p]))
            batch.append(by_position[second].pop())

        remaining = [index for values in by_position.values() for index in values]
        if self.generator is not None and remaining:
            order = torch.randperm(len(remaining), generator=self.generator).tolist()
            remaining = [remaining[offset] for offset in order]
        else:
            remaining.sort()

        cursor = 0
        for batch, target_size in zip(batches, self._batch_sizes, strict=True):
            needed = target_size - len(batch)
            batch.extend(remaining[cursor : cursor + needed])
            cursor += needed
        if cursor != len(remaining):
            raise RuntimeError("Position-diverse batching did not consume every example exactly once.")
        if self.generator is not None and len(batches) > 1:
            order = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[offset] for offset in order]
        yield from batches
