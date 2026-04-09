"""NCCL per-parameter broadcast between trainer (rank 0) and inference (rank 1)."""
import asyncio
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from weight_sync.base import WeightSyncer


class NCCLBroadcastSyncer(WeightSyncer):
    """Per-parameter NCCL broadcast.

    Uses TCPStore for rendezvous between separately spawned processes.
    Trainer is rank 0, inference is rank 1.
    """

    def __init__(self, rank: int, world_size: int = 2,
                 master_addr: str = "localhost", master_port: int = 29500):
        self.rank = rank
        self.world_size = world_size
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

    async def push(self, model: nn.Module, version: int) -> float:
        """Broadcast each parameter from rank 0 to all other ranks."""
        self._init_process_group()
        start = time.time()
        loop = asyncio.get_event_loop()

        def _broadcast():
            device = next(model.parameters()).device
            for name, param in model.named_parameters():
                dist.broadcast(param.data, src=0)
            v = torch.tensor([version], dtype=torch.long, device=device)
            dist.broadcast(v, src=0)

        await loop.run_in_executor(None, _broadcast)
        self._version = version
        return time.time() - start

    async def pull(self, model: nn.Module) -> int:
        """Receive broadcast parameters from rank 0."""
        self._init_process_group()
        loop = asyncio.get_event_loop()

        def _receive():
            device = next(model.parameters()).device
            for name, param in model.named_parameters():
                dist.broadcast(param.data, src=0)
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
