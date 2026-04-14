"""Interrupt strategies for weight-sync coordination.

Interrupts govern what happens to in-flight generation when the trainer
needs to push fresh weights. Since inference runs in a separate OS process,
these strategies must be backed by multiprocessing primitives (not asyncio
primitives) so coordinator and inference workers can coordinate.

Coordinator-side API (async):
  - prepare_for_sync()  before pushing weights
  - resume_after_sync() after pushing weights

Inference-side API (sync):
  - wait_if_paused()    called before starting a new generation
  - begin_generation()  called when generation starts
  - end_generation()    called when generation finishes
"""
from abc import ABC, abstractmethod


class InterruptStrategy(ABC):
    @abstractmethod
    async def prepare_for_sync(self):
        ...

    @abstractmethod
    async def resume_after_sync(self):
        ...

    # Inference-side hooks — default no-ops so base BatchSync doesn't care.
    def wait_if_paused(self) -> None:
        pass

    def begin_generation(self) -> None:
        pass

    def end_generation(self) -> None:
        pass
