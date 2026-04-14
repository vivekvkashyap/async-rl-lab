import json
import os
import shutil
import time
import glob as globmod
import asyncio
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from weight_sync.base import WeightSyncer


class FilesystemSyncer(WeightSyncer):
    """Save/load safetensors to disk for weight synchronization.

    Saves weights in HF-compatible directory format so vLLM can reload them
    via reload_weights(weights_path=...).
    """

    def __init__(self, checkpoint_dir: str, keep_last: int = 4):
        self.checkpoint_dir = checkpoint_dir
        # keep_last defaults to 4 so the currently-loading inference process
        # still has at least 3 older snapshots around during cleanup — avoids
        # racing against a just-picked directory being removed mid-load.
        self.keep_last = keep_last
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._model_config_copied = False

    def _version_dir(self, version: int) -> str:
        return os.path.join(self.checkpoint_dir, f"v{version:06d}")

    def _cleanup(self):
        # Glob for committed directories only (skip .tmp staging dirs).
        all_dirs = globmod.glob(os.path.join(self.checkpoint_dir, "v*"))
        committed = sorted(d for d in all_dirs if not d.endswith(".tmp"))
        while len(committed) > self.keep_last:
            shutil.rmtree(committed.pop(0), ignore_errors=True)

    def _copy_model_config(self, model, version_dir: str):
        """Copy config.json from the original model so vLLM can reload."""
        if hasattr(model, "config"):
            config_path = os.path.join(version_dir, "config.json")
            model.config.save_pretrained(version_dir)

    async def push(self, model: nn.Module, version: int) -> float:
        start = time.time()

        from utils.fsdp import is_fsdp_wrapped, gather_weights_on_master

        if is_fsdp_wrapped(model):
            # FSDP: gather shards to rank 0, only rank 0 saves
            state_dict = gather_weights_on_master(model)
            import torch.distributed as dist
            rank = dist.get_rank()
            if rank != 0:
                # Non-master ranks just wait for the gather and return
                dist.barrier()
                return time.time() - start
        else:
            state_dict = {}
            seen_data_ptrs = {}
            for k, v in model.state_dict().items():
                v = v.contiguous()
                ptr = v.data_ptr()
                if ptr in seen_data_ptrs:
                    state_dict[k] = v.clone()
                else:
                    seen_data_ptrs[ptr] = k
                    state_dict[k] = v

        # Stage into a .tmp dir, then atomically rename. Inference only picks
        # up committed v###### directories, so it never observes a half-written
        # safetensors file — and the commit is a single atomic rename.
        version_dir = self._version_dir(version)
        staging_dir = version_dir + ".tmp"
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        os.makedirs(staging_dir, exist_ok=True)

        safetensors_path = os.path.join(staging_dir, "model.safetensors")
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: save_file(state_dict, safetensors_path)
        )

        unwrapped = model
        if is_fsdp_wrapped(model):
            unwrapped = model.module
        self._copy_model_config(unwrapped, staging_dir)

        if os.path.exists(version_dir):
            shutil.rmtree(version_dir, ignore_errors=True)
        os.rename(staging_dir, version_dir)

        self._cleanup()

        # Signal non-master ranks that save is complete
        if is_fsdp_wrapped(model):
            import torch.distributed as dist
            dist.barrier()

        return time.time() - start

    async def pull(self, model: nn.Module) -> int:
        dirs = sorted(globmod.glob(os.path.join(self.checkpoint_dir, "v*")))
        if not dirs:
            return -1
        latest_dir = dirs[-1]
        version = int(os.path.basename(latest_dir).lstrip("v"))
        safetensors_path = os.path.join(latest_dir, "model.safetensors")
        state_dict = await asyncio.get_event_loop().run_in_executor(
            None, lambda: load_file(safetensors_path)
        )
        model.load_state_dict(state_dict)
        return version
