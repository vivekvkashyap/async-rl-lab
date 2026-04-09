from abc import ABC, abstractmethod


class InterruptStrategy(ABC):
    @abstractmethod
    async def prepare_for_sync(self):
        """Called before weight sync. Handle in-flight generation."""
        ...

    @abstractmethod
    async def resume_after_sync(self):
        """Called after weight sync completes. Resume generation."""
        ...
