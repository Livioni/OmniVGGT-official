# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Random sampling under a constraint
# --------------------------------------------------------
import numpy as np
import torch
from collections import deque


class BatchedRandomSampler:
    """ Random sampling under a constraint: each sample in the batch has the same feature, 
    which is chosen randomly from a known pool of 'features' for each batch.

    For instance, the 'feature' could be the image aspect-ratio.

    The index returned is a tuple (sample_idx, feat_idx).
    This sampler ensures that each series of `batch_size` indices has the same `feat_idx`.
    """

    def __init__(self, dataset, batch_size, pool_size, world_size=1, rank=0, drop_last=True):
        self.batch_size = batch_size
        self.pool_size = pool_size

        self.len_dataset = N = len(dataset)
        self.total_size = round_by(N, batch_size*world_size) if drop_last else N
        # assert world_size == 1 or drop_last, 'must drop the last batch in distributed mode'

        # distributed sampler
        self.world_size = world_size
        self.rank = rank
        self.epoch = None

    def __len__(self):
        return self.total_size // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # prepare RNG
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, 'use set_epoch() if distributed mode is used'
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        # random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        # random feat_idxs (same across each batch)
        n_batches = (self.total_size+self.batch_size-1) // self.batch_size
        feat_idxs = rng.integers(self.pool_size, size=n_batches)
        feat_idxs = np.broadcast_to(feat_idxs[:, None], (n_batches, self.batch_size))
        feat_idxs = feat_idxs.ravel()[:self.total_size]

        # put them together
        idxs = np.c_[sample_idxs, feat_idxs]  # shape = (total_size, 2)

        # Distributed sampler: we select a subset of batches
        # make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * ((self.total_size + self.world_size *
                                           self.batch_size-1) // (self.world_size * self.batch_size))
        idxs = idxs[self.rank*size_per_proc: (self.rank+1)*size_per_proc]

        yield from (tuple(idx) for idx in idxs)


def round_by(total, multiple, up=False):
    if up:
        total = total + multiple-1
    return (total//multiple) * multiple


class AnchorFrameSampler(BatchedRandomSampler):
    def __init__(self, dataset, batch_size, pool_size, world_size=1,
                 rank=0, drop_last=True, recent_buffer_size=10000):
        # Pass world_size and rank to parent to enable proper distributed sampling
        super().__init__(dataset, 1, pool_size, world_size=world_size, rank=rank, drop_last=drop_last)
        self.batch_size = 1                       # 每次产出一个"逻辑样本"
        self.image_num_batch = batch_size         # 逻辑样本内的图片/帧数量
        self.recent_buffer_size = int(recent_buffer_size)

    def __len__(self):
        return self.total_size 

    def __iter__(self):
        # RNG（按 epoch 可复现）
        if self.epoch is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        n_batches = (self.total_size + self.batch_size - 1) // self.batch_size  # == self.total_size
        feat_idxs = rng.integers(self.pool_size, size=n_batches).astype(np.int64)
        batch_size_pools = np.full(n_batches, self.image_num_batch, dtype=np.int64)

        L = self.total_size
        recent_mask = np.zeros(L, dtype=bool)
        recent_queue = deque(maxlen=min(self.recent_buffer_size, L))

        def _mark_recent(local_pos: int):
            if len(recent_queue) == recent_queue.maxlen:
                popped = recent_queue.popleft()
                recent_mask[popped] = False
            recent_queue.append(local_pos)
            recent_mask[local_pos] = True

       
        if self.image_num_batch == 24:
            valid_lengths = [1, 2, 4, 6, 8, 12]
        elif self.image_num_batch == 18:
            valid_lengths = [1, 2, 3, 6, 9]
        elif self.image_num_batch == 16:
            valid_lengths = [1, 2, 4, 8]
        elif self.image_num_batch == 12:
            valid_lengths = [1, 2, 4, 6]
        elif self.image_num_batch == 4:
            valid_lengths = [1, 2]
        else:
            raise ValueError(f"Invalid train_batch_images: {self.train_batch_images}")

        # Note: Distribution is now handled by Accelerate, not here
        # When world_size=1 and rank=0, we process all samples
        # Accelerate will automatically shard the dataloader across processes
        for i in range(self.total_size):
            length = int(rng.choice(valid_lengths))

            # 先从"未最近使用"里无放回采样，不够再全局带放回采样
            remaining_idx = np.where(~recent_mask)[0]
            if remaining_idx.size >= length:
                pick_local_pos = rng.choice(remaining_idx, size=length, replace=False)
            else:
                pick_local_pos = rng.integers(0, L, size=length)

            # 实际样本 id（注意我们对 sample_idxs 做过洗牌）
            sampled_ids = sample_idxs[np.atleast_1d(pick_local_pos)].tolist()

            # 更新最近使用
            for lp in np.atleast_1d(pick_local_pos):
                _mark_recent(int(lp))

            # 产出：多个 id + feat_idx + image_num_batch
            yield tuple(sampled_ids + [int(feat_idxs[i])] + [int(batch_size_pools[i])])
