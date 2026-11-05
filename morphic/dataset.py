"""Bounded shard access and deterministic sampling for complete graphs."""
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from morphic.model import verify_sha256


class Sample:
    def __init__(self, root: Path, sample: str):
        self.sample = sample
        self.root = Path(root) / sample
        self.meta = json.loads((self.root / 'complete.json').read_text())
        self.shards = self.meta['shards']
        self.stops = np.asarray([x['edge_stop'] for x in self.shards], dtype=np.int64)
        self.starts = np.asarray([x['edge_start'] for x in self.shards], dtype=np.int64)
        self.measured = np.asarray(self.meta['measured_pair_indices'], dtype=np.int64)
        self.embeddings = np.load(self.root / 'embeddings.npy', mmap_mode='r')
        if self.embeddings.shape != (self.meta['graph_nodes'], 1536):
            raise ValueError(f'{sample}: embedding shape differs from graph')
        self._pairs = self._sender = self._receiver = None
        verify_sha256(Path('data/lr_catalog.csv'), self.meta['catalog_sha256'])

    @lru_cache(maxsize=8)
    def edge_shard(self, index):
        frame = pd.read_parquet(self.root / f'edges_{index:04d}.parquet')
        sh = self.shards[index]
        if not np.array_equal(frame.edge_local, np.arange(sh['edge_start'], sh['edge_stop'])):
            raise ValueError(f'{self.sample}: edge shard alignment')
        return frame

    @lru_cache(maxsize=8)
    def target_shard(self, index):
        matrix = sparse.load_npz(self.root / f'lr_{index:04d}.npz').tocsr()
        if matrix.shape != (self.stops[index]-self.starts[index], 1821):
            raise ValueError(f'{self.sample}: target shard alignment')
        return matrix

    def context(self, head, rows, device):
        s, r, d, _ = self.fetch(rows, targets=False)
        return head.encode_context(torch.from_numpy(np.asarray(self.embeddings[s])).to(device),
            torch.from_numpy(np.asarray(self.embeddings[r])).to(device), torch.from_numpy(d).to(device))

    def edge_arrays(self):
        if self._sender is None:
            sender = np.empty(self.meta['directed_edges'], dtype=np.int32)
            receiver = np.empty_like(sender)
            for sh in self.shards:
                i, start, stop = sh['index'], sh['edge_start'], sh['edge_stop']
                frame = self.edge_shard(i)
                sender[start:stop] = frame.sender.to_numpy(np.int32)
                receiver[start:stop] = frame.receiver.to_numpy(np.int32)
            self._sender, self._receiver = sender, receiver
        return self._sender, self._receiver

    def paired_rows(self):
        if self._pairs is None:
            sender, receiver = self.edge_arrays()
            width = self.meta['graph_nodes']
            keys = sender.astype(np.int64) * width + receiver
            order = np.argsort(keys)
            sorted_keys = keys[order]
            reverse_keys = receiver.astype(np.int64) * width + sender
            positions = np.searchsorted(sorted_keys, reverse_keys)
            if np.any(sender == receiver) or len(np.unique(sorted_keys)) != len(keys) or np.any(positions == len(keys)):
                raise ValueError(f'{self.sample}: invalid paired edge graph')
            if not np.array_equal(sorted_keys[positions], reverse_keys):
                raise ValueError(f'{self.sample}: missing reverse edges')
            canonical = np.flatnonzero(sender < receiver)
            self._pairs = np.column_stack((canonical, order[positions[canonical]]))
            if len(self._pairs) * 2 != len(keys):
                raise ValueError(f'{self.sample}: paired edge count mismatch')
        return self._pairs

    def fetch(self, rows: np.ndarray, targets=True):
        """Return graph endpoints, distance and sparse labels in requested row order."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= self.meta['directed_edges']):
            raise ValueError('Invalid local edge rows')
        ids = np.searchsorted(self.stops, rows, side='right')
        sender = np.empty(len(rows), np.int32)
        receiver = np.empty_like(sender)
        distance = np.empty(len(rows), np.float32)
        grouped_blocks, grouped_positions = [], []
        for i in np.unique(ids):
            positions = np.flatnonzero(ids == i)
            local = rows[positions] - self.starts[i]
            edge = self.edge_shard(int(i)).iloc[local]
            sender[positions] = edge.sender.to_numpy(np.int32)
            receiver[positions] = edge.receiver.to_numpy(np.int32)
            distance[positions] = edge.distance_um.to_numpy(np.float32)
            if targets:
                grouped_blocks.append(self.target_shard(int(i))[local])
                grouped_positions.append(positions)
        labels = None
        if targets:
            order = np.argsort(np.concatenate(grouped_positions))
            labels = sparse.vstack(grouped_blocks, format='csr')[order]
        return sender, receiver, distance, labels


def epoch_batches(samples, batch: int, seed: int, epoch: int):
    """Visit each directed edge once, preserving the paper's shard and RNG order."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    jobs = [(sample, int(sh['index'])) for sample in samples for sh in sample.shards]
    rng.shuffle(jobs)
    for sample, shard in jobs:
        pairs = sample.paired_rows()
        lo, hi = sample.starts[shard], sample.stops[shard]
        local = pairs[(pairs[:, 0] >= lo) & (pairs[:, 0] < hi)]
        for chunk in np.array_split(rng.permutation(len(local)), max(1, math.ceil(len(local) / (batch // 2)))):
            if not len(chunk):
                continue
            yield sample, local[chunk].reshape(-1)


def validation_rows(sample: Sample, count: int, seed: int):
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(hashlib.sha256(sample.sample.encode()).hexdigest()[:8], 16)]))
    return np.sort(rng.choice(sample.meta['directed_edges'], min(count, sample.meta['directed_edges']), replace=False))

