"""NCCL bucketed broadcast: pack params into contiguous buckets then broadcast."""
import asyncio
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from weight_sync.base import WeightSyncer


class NCCLBucketedSyncer(WeightSyncer):
    """Pack parameters into 1GB buckets before broadcasting.

    One broadcast call per bucket instead of one per parameter.
    Significantly reduces NCCL call overhead.
    """

    def __init__(self, rank: int, world_size: int = 2,
                 master_addr: str = "localhost", master_port: int = 29500,
                 bucket_size_mb: int = 1024):
        self.rank = rank
        self.world_size = world_size
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self._initialized = False
        self._master_addr = master_addr
        self._master_port = master_port
        self._version = 0

    def _init_process_group(self):
        if self._initialized:
            return
        store = dist.TCPStore(
            self._master_addr, self._master_port,
            world_size=self.world_size,
            is_master=(self.rank == 0),
        )
        dist.init_process_group(
            backend="nccl", store=store,
            rank=self.rank, world_size=self.world_size,
        )
        self._initialized = True

    def _build_buckets(self, model: nn.Module):
        """Pack parameters into contiguous buckets grouped by dtype."""
        buckets = []
        current_params = []
        current_size = 0

        # Sort by dtype to avoid mixing
        params = list(model.named_parameters())
        params.sort(key=lambda x: str(x[1].dtype))

        for name, param in params:
            param_bytes = param.data.nelement() * param.data.element_size()
            if current_size + param_bytes > self.bucket_size_bytes and current_params:
                buckets.append(current_params)
                current_params = []
                current_size = 0
            current_params.append((name, param))
            current_size += param_bytes

        if current_params:
            buckets.append(current_params)

        return buckets

    def _flatten_bucket(self, params):
        """Flatten a bucket of parameters into a single contiguous tensor."""
        flat = torch.cat([p.data.reshape(-1) for _, p in params])
        return flat

    def _unflatten_bucket(self, flat, params):
        """Unflatten a contiguous tensor back into parameters."""
        offset = 0
        for name, param in params:
            numel = param.data.nelement()
            param.data.copy_(flat[offset:offset + numel].reshape(param.data.shape))
            offset += numel

    async def push(self, model: nn.Module, version: int) -> float:
        self._init_process_group()
        start = time.time()
        loop = asyncio.get_event_loop()

        def _broadcast():
            buckets = self._build_buckets(model)
            for bucket_params in buckets:
                flat = self._flatten_bucket(bucket_params)
                dist.broadcast(flat, src=0)
            # Broadcast version
            device = next(model.parameters()).device
            v = torch.tensor([version], dtype=torch.long, device=device)
            dist.broadcast(v, src=0)

        await loop.run_in_executor(None, _broadcast)
        self._version = version
        return time.time() - start

    async def pull(self, model: nn.Module) -> int:
        self._init_process_group()
        loop = asyncio.get_event_loop()

        def _receive():
            buckets = self._build_buckets(model)
            for bucket_params in buckets:
                flat = self._flatten_bucket(bucket_params)
                dist.broadcast(flat, src=0)
                self._unflatten_bucket(flat, bucket_params)
            device = next(model.parameters()).device
            v = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(v, src=0)
            return v.item()

        version = await loop.run_in_executor(None, _receive)
        self._version = version
        return version

    def cleanup(self):
        if self._initialized:
            dist.destroy_process_group()
            self._initialized = False
