"""LeRobotDataset with a fast delta-timestamps query.

LeRobot's `_query_hf_dataset` tries a column-first lookup "for speed", but on
this datasets version a `Column` can't be `torch.stack`ed, so the try/except
silently falls back to ROW-first indexing — which materialises the full rows
for the whole action chunk, PNG-decoding the embedded image column ~50 times
per sample just to read 50 six-float action vectors. Measured: ~43 ms/sample,
i.e. ~2.8 s of dataloader CPU per batch-64.

Projecting to the requested column BEFORE row indexing returns bit-identical
tensors in ~0.6 ms/sample (~70x). Use this class anywhere a dataset is built
with `delta_timestamps` (training); it changes nothing else.
"""
from __future__ import annotations

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class FastChunkDataset(LeRobotDataset):
    def _query_hf_dataset(self, query_indices: dict) -> dict:
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            column = self.hf_dataset.select_columns([key])[relative_indices][key]
            result[key] = torch.stack(column)
        return result
