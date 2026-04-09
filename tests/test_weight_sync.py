"""Tests for weight sync implementations."""
import asyncio
import sys
import os
import tempfile
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weight_sync.filesystem_sync import FilesystemSyncer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)


class TestFilesystemSync:
    def test_push_pull_roundtrip(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmpdir:
                syncer = FilesystemSyncer(checkpoint_dir=tmpdir)

                model_a = SimpleModel()
                model_b = SimpleModel()

                # Models should differ initially
                for pa, pb in zip(model_a.parameters(), model_b.parameters()):
                    assert not torch.allclose(pa, pb) or pa.numel() == 0

                # Push model_a, pull into model_b
                duration = await syncer.push(model_a, version=1)
                assert duration > 0
                version = await syncer.pull(model_b)
                assert version == 1

                # Now they should match
                for pa, pb in zip(model_a.parameters(), model_b.parameters()):
                    assert torch.allclose(pa, pb), "Parameters should match after sync"

        asyncio.run(run())

    def test_versioning(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmpdir:
                syncer = FilesystemSyncer(checkpoint_dir=tmpdir, keep_last=2)
                model = SimpleModel()

                await syncer.push(model, version=1)
                await syncer.push(model, version=2)
                await syncer.push(model, version=3)

                # Should pull latest (version 3)
                model2 = SimpleModel()
                version = await syncer.pull(model2)
                assert version == 3

                # Should have cleaned up, keeping only 2 version directories
                import glob
                dirs = glob.glob(os.path.join(tmpdir, "v*"))
                assert len(dirs) == 2, f"Expected 2 dirs, got {len(dirs)}: {dirs}"

        asyncio.run(run())


if __name__ == "__main__":
    t = TestFilesystemSync()
    t.test_push_pull_roundtrip()
    print("test_push_pull_roundtrip PASSED")
    t.test_versioning()
    print("test_versioning PASSED")
    print("\nAll weight sync tests passed!")
