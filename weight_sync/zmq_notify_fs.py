"""Filesystem sync + ZMQ PUB/SUB notification (PRIME-RL style).

push() saves safetensors to disk then publishes a notification message.
pull() subscribes to notifications and loads weights when notified.
Avoids filesystem polling — subscriber reacts instantly.
"""
import asyncio
import json
import os
import time
import torch.nn as nn
from safetensors.torch import save_file, load_file
from weight_sync.base import WeightSyncer


class ZMQNotifyFSSyncer(WeightSyncer):
    """Combines filesystem sync with ZMQ PUB/SUB notification."""

    def __init__(self, checkpoint_dir: str, zmq_port: int = 5555,
                 role: str = "publisher"):
        import zmq
        self.checkpoint_dir = checkpoint_dir
        self.zmq_port = zmq_port
        os.makedirs(checkpoint_dir, exist_ok=True)

        self._ctx = zmq.Context()
        if role == "publisher":
            self._socket = self._ctx.socket(zmq.PUB)
            self._socket.bind(f"tcp://*:{zmq_port}")
        else:
            self._socket = self._ctx.socket(zmq.SUB)
            self._socket.connect(f"tcp://localhost:{zmq_port}")
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self._version = 0

    def _path(self, version: int) -> str:
        return os.path.join(self.checkpoint_dir, f"weights_v{version:06d}.safetensors")

    async def push(self, model: nn.Module, version: int) -> float:
        start = time.time()
        loop = asyncio.get_event_loop()

        # De-duplicate shared tensors
        seen_ptrs = {}
        state_dict = {}
        for k, v in model.state_dict().items():
            v = v.contiguous()
            ptr = v.data_ptr()
            if ptr in seen_ptrs:
                state_dict[k] = v.clone()
            else:
                seen_ptrs[ptr] = k
                state_dict[k] = v

        path = self._path(version)
        await loop.run_in_executor(None, lambda: save_file(state_dict, path))

        # Publish notification as JSON for safe parsing
        msg = json.dumps({"type": "weights_ready", "version": version, "path": path})
        self._socket.send_string(msg)
        self._version = version

        self._cleanup()
        return time.time() - start

    async def pull(self, model: nn.Module) -> int:
        loop = asyncio.get_event_loop()

        def _wait_and_load():
            msg = self._socket.recv_string()
            data = json.loads(msg)
            version = data["version"]
            path = data["path"]
            state_dict = load_file(path)
            model.load_state_dict(state_dict)
            return version

        version = await loop.run_in_executor(None, _wait_and_load)
        self._version = version
        return version

    def _cleanup(self, keep_last: int = 2):
        import glob
        files = sorted(glob.glob(os.path.join(self.checkpoint_dir, "weights_v*.safetensors")))
        while len(files) > keep_last:
            os.remove(files.pop(0))

    def close(self):
        self._socket.close()
        self._ctx.term()
