from interrupts.base import InterruptStrategy


class BatchSync(InterruptStrategy):
    """No-op interrupt. Generation runs to completion before sync.

    Works across processes without any IPC — inference just runs normally,
    and the coordinator runs the sync between whole batches.
    """

    async def prepare_for_sync(self):
        pass

    async def resume_after_sync(self):
        pass
