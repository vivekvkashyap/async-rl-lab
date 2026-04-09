"""Pattern 4: Redis XADD/XREAD, unbounded stream."""
import asyncio
import base64
import io
import json
import time
from typing import List

import numpy as np
import torch

from core.types import ScoredRollout
from buffers.base import RolloutBuffer


def _tensor_to_b64(t: torch.Tensor) -> str:
    buf = io.BytesIO()
    np.save(buf, t.detach().cpu().numpy())
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_tensor(s: str) -> torch.Tensor:
    buf = io.BytesIO(base64.b64decode(s))
    arr = np.load(buf)
    return torch.from_numpy(arr.copy())


def _rollout_to_dict(r: ScoredRollout) -> dict:
    d = {
        "prompt": r.prompt,
        "prompt_ids": _tensor_to_b64(r.prompt_ids),
        "completion": r.completion,
        "completion_ids": _tensor_to_b64(r.completion_ids),
        "logprobs": _tensor_to_b64(r.logprobs),
        "model_version": str(r.model_version),
        "generated_at": str(r.generated_at),
        "prompt_id": r.prompt_id,
        "ground_truth": str(r.ground_truth) if r.ground_truth is not None else "",
        "reward": str(r.reward) if r.reward is not None else "",
    }
    if r.sampling_mask is not None:
        d["sampling_mask"] = _tensor_to_b64(r.sampling_mask)
    if r.teacher_logprobs is not None:
        d["teacher_logprobs"] = _tensor_to_b64(r.teacher_logprobs)
    return d


def _dict_to_rollout(d: dict) -> ScoredRollout:
    return ScoredRollout(
        prompt=d["prompt"],
        prompt_ids=_b64_to_tensor(d["prompt_ids"]),
        completion=d["completion"],
        completion_ids=_b64_to_tensor(d["completion_ids"]),
        logprobs=_b64_to_tensor(d["logprobs"]),
        model_version=int(d["model_version"]),
        generated_at=float(d["generated_at"]),
        prompt_id=d["prompt_id"],
        ground_truth=float(d["ground_truth"]) if d.get("ground_truth") else None,
        sampling_mask=_b64_to_tensor(d["sampling_mask"]) if d.get("sampling_mask") else None,
        reward=float(d["reward"]) if d.get("reward") else None,
        teacher_logprobs=_b64_to_tensor(d["teacher_logprobs"]) if d.get("teacher_logprobs") else None,
    )


class RedisStreamBuffer(RolloutBuffer):
    """Unbounded buffer using Redis streams (XADD/XREAD).

    Uses Redis XADD for append and XREAD with blocking for consumption.
    Consumer tracks its position via last-read message ID.
    """

    def __init__(self, redis_url: str = "redis://localhost", stream_key: str = "rollouts"):
        import redis
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.stream_key = stream_key
        self._last_id = "0-0"
        # Clear old stream on init
        self.client.delete(self.stream_key)

    async def put(self, rollout: ScoredRollout) -> None:
        loop = asyncio.get_event_loop()
        data = _rollout_to_dict(rollout)
        await loop.run_in_executor(None, self.client.xadd, self.stream_key, data)

    async def get(self, batch_size: int) -> List[ScoredRollout]:
        loop = asyncio.get_event_loop()
        batch = []
        while len(batch) < batch_size:
            result = await loop.run_in_executor(
                None,
                lambda: self.client.xread(
                    {self.stream_key: self._last_id}, count=batch_size - len(batch), block=1000
                ),
            )
            if result:
                for stream_name, messages in result:
                    for msg_id, data in messages:
                        batch.append(_dict_to_rollout(data))
                        self._last_id = msg_id
        return batch

    def size(self) -> int:
        try:
            return self.client.xlen(self.stream_key)
        except Exception:
            return 0
