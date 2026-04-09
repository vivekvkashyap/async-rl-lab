from interrupts.base import InterruptStrategy


class BatchSync(InterruptStrategy):
    """No-op interrupt strategy. Generation finishes before sync."""

    async def prepare_for_sync(self):
        pass

    async def resume_after_sync(self):
        pass
