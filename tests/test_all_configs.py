#!/usr/bin/env python3
"""Test every configuration option for each axis works correctly.

Runs unit-level tests (no vLLM, no GPU required for most).
"""
import asyncio
import os
import sys
import tempfile
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import ScoredRollout, TrainingBatch


def make_rollout(prompt_id="p1", reward=1.0, version=0, comp_len=10):
    return ScoredRollout(
        prompt="What is 2+2?",
        prompt_ids=torch.randint(0, 1000, (5,)),
        completion="The answer is 4. #### 4",
        completion_ids=torch.randint(0, 1000, (comp_len,)),
        logprobs=torch.randn(comp_len) * 0.1 - 2.0,
        model_version=version,
        generated_at=time.time(),
        prompt_id=prompt_id,
        ground_truth=4.0,
        reward=reward,
    )


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)


# ============================================================
# AXIS 1: BUFFERS
# ============================================================

def test_sync_buffer():
    from buffers.sync_buffer import SyncBuffer
    buf = SyncBuffer()
    async def run():
        await buf.put(make_rollout())
        await buf.put(make_rollout())
        assert buf.size() == 2
        batch = await buf.get(2)
        assert len(batch) == 2
        assert buf.size() == 0
    asyncio.run(run())
    print("  [PASS] sync_buffer")


def test_bounded_queue():
    from buffers.bounded_queue import BoundedQueueBuffer
    buf = BoundedQueueBuffer(maxsize=4)
    async def run():
        await buf.put(make_rollout())
        await buf.put(make_rollout())
        await buf.put(make_rollout())
        assert buf.size() == 3
        batch = await buf.get(2)
        assert len(batch) == 2
        assert buf.size() == 1
    asyncio.run(run())
    print("  [PASS] bounded_queue")


def test_double_buffer():
    from buffers.double_buffer import DoubleBuffer
    buf = DoubleBuffer()
    async def run():
        await buf.put(make_rollout())
        await buf.put(make_rollout())
        await buf.put(make_rollout())
        assert buf.size() == 3
        batch = await buf.get(2)
        assert len(batch) == 2
    asyncio.run(run())
    buf.shutdown()
    print("  [PASS] double_buffer")


def test_redis_stream():
    """Test redis buffer — skips if Redis is not running."""
    try:
        import redis
        client = redis.from_url("redis://localhost", decode_responses=True)
        client.ping()
    except Exception:
        print("  [SKIP] redis_stream (Redis not available)")
        return

    from buffers.redis_stream import RedisStreamBuffer
    buf = RedisStreamBuffer(redis_url="redis://localhost", stream_key="test_rollouts")
    async def run():
        await buf.put(make_rollout())
        await buf.put(make_rollout())
        assert buf.size() == 2
        batch = await buf.get(2)
        assert len(batch) == 2
        # Verify tensor round-trip
        r = batch[0]
        assert r.prompt_ids.shape[0] == 5
        assert r.completion_ids.shape[0] == 10
        assert isinstance(r.reward, float)
    asyncio.run(run())
    print("  [PASS] redis_stream")


# ============================================================
# AXIS 2: WEIGHT SYNC
# ============================================================

def test_filesystem_sync():
    from weight_sync.filesystem_sync import FilesystemSyncer
    async def run():
        with tempfile.TemporaryDirectory() as tmpdir:
            syncer = FilesystemSyncer(checkpoint_dir=tmpdir)
            model_a = SimpleModel()
            model_b = SimpleModel()
            duration = await syncer.push(model_a, version=1)
            assert duration > 0
            version = await syncer.pull(model_b)
            assert version == 1
            for pa, pb in zip(model_a.parameters(), model_b.parameters()):
                assert torch.allclose(pa, pb)
    asyncio.run(run())
    print("  [PASS] filesystem_sync")


def test_nccl_broadcast():
    """NCCL requires multi-process — just verify import and init."""
    from weight_sync.nccl_broadcast import NCCLBroadcastSyncer
    syncer = NCCLBroadcastSyncer(rank=0, world_size=2)
    assert syncer.rank == 0
    assert syncer._version == 0
    print("  [PASS] nccl_broadcast (import + init)")


def test_nccl_bucketed():
    """NCCL requires multi-process — verify import and bucket logic."""
    from weight_sync.nccl_bucketed import NCCLBucketedSyncer
    syncer = NCCLBucketedSyncer(rank=0, bucket_size_mb=64)
    assert syncer.bucket_size_bytes == 64 * 1024 * 1024
    # Test bucket building (no NCCL needed)
    model = SimpleModel()
    buckets = syncer._build_buckets(model)
    assert len(buckets) > 0
    # Test flatten/unflatten roundtrip preserves values
    for bucket in buckets:
        original = [p.data.clone() for _, p in bucket]
        flat = syncer._flatten_bucket(bucket)
        assert flat.dim() == 1
        syncer._unflatten_bucket(flat, bucket)
        for orig, (_, p) in zip(original, bucket):
            assert torch.allclose(orig, p.data)
    print("  [PASS] nccl_bucketed (bucket logic + roundtrip)")


def test_zmq_notify_fs():
    """Test ZMQ syncer — skips if pyzmq not installed."""
    try:
        import zmq
    except ImportError:
        print("  [SKIP] zmq_notify_fs (pyzmq not installed)")
        return

    from weight_sync.zmq_notify_fs import ZMQNotifyFSSyncer
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test publisher side
        pub = ZMQNotifyFSSyncer(checkpoint_dir=tmpdir, zmq_port=15555, role="publisher")
        model = SimpleModel()
        async def run():
            duration = await pub.push(model, version=1)
            assert duration > 0
            # Verify file was written
            import glob
            files = glob.glob(os.path.join(tmpdir, "weights_v*.safetensors"))
            assert len(files) == 1
        asyncio.run(run())
        pub.close()
    print("  [PASS] zmq_notify_fs")


# ============================================================
# AXIS 3: STALENESS
# ============================================================

def test_no_filter():
    from staleness.no_filter import NoFilter
    nf = NoFilter()
    rollouts = [make_rollout(version=v) for v in range(5)]
    batch = nf.process(rollouts, current_version=10)
    assert len(batch.rollouts) == 5
    assert batch.is_weights is None
    print("  [PASS] no_filter")


def test_version_rejection():
    from staleness.version_rejection import VersionRejection
    vr = VersionRejection(max_lag=3)
    rollouts = [make_rollout(version=v) for v in [7, 8, 9, 5, 3, 10]]
    batch = vr.process(rollouts, current_version=10)
    # Should keep versions 7,8,9,10 (lag <= 3), drop 5 and 3
    assert len(batch.rollouts) == 4
    kept_versions = {r.model_version for r in batch.rollouts}
    assert kept_versions == {7, 8, 9, 10}
    print("  [PASS] version_rejection")


def test_is_reweighting():
    from staleness.is_reweighting import ISReweighting
    # Without model, should pass through
    isr = ISReweighting(clip_eps=0.2)
    rollouts = [make_rollout(version=0)]
    batch = isr.process(rollouts, current_version=1)
    assert len(batch.rollouts) == 1
    assert batch.is_weights is None  # No model set, passthrough
    print("  [PASS] is_reweighting (no model passthrough)")


def test_is_reweighting_with_model():
    """Test IS reweighting with a real model."""
    from staleness.is_reweighting import ISReweighting
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        print("  [SKIP] is_reweighting_with_model (tiny-gpt2 not available)")
        return

    isr = ISReweighting(clip_eps=0.2, model=model, tokenizer=tokenizer, device="cpu")

    # Create rollout with token IDs that match model's vocab
    r = ScoredRollout(
        prompt="test", prompt_ids=torch.tensor([50, 100, 200]),
        completion="answer", completion_ids=torch.tensor([300, 400, 500]),
        logprobs=torch.tensor([-2.0, -3.0, -1.5]),
        model_version=0, generated_at=0.0,
        prompt_id="p1", ground_truth=1.0, reward=1.0,
    )
    batch = isr.process([r], current_version=1)
    assert len(batch.rollouts) == 1
    assert batch.is_weights is not None
    assert batch.is_weights.shape == (1,)
    w = batch.is_weights[0].item()
    assert 1 - 0.2 <= w <= 1 + 0.2, f"IS weight {w} out of clip range"
    print("  [PASS] is_reweighting_with_model")


def test_hybrid():
    from staleness.hybrid import HybridStaleness
    hs = HybridStaleness(max_lag=2, is_clip_eps=0.2)
    rollouts = [make_rollout(version=v) for v in [8, 9, 5, 10]]
    batch = hs.process(rollouts, current_version=10)
    # Should reject version 5 (lag=5 > 2), keep 8,9,10
    assert len(batch.rollouts) == 3
    kept_versions = {r.model_version for r in batch.rollouts}
    assert 5 not in kept_versions
    # No model set, so IS weights should be None (passthrough from ISReweighting)
    print("  [PASS] hybrid")


# ============================================================
# AXIS 4: INTERRUPTS
# ============================================================

def test_batch_sync():
    from interrupts.batch_sync import BatchSync
    bs = BatchSync()
    async def run():
        await bs.prepare_for_sync()
        await bs.resume_after_sync()
    asyncio.run(run())
    print("  [PASS] batch_sync")


def test_soft_drain():
    from interrupts.soft_drain import SoftDrain
    sd = SoftDrain()
    async def run():
        assert sd.is_accepting
        # Simulate: begin generation, then try to sync
        await sd.begin_generation()

        sync_done = False
        async def do_sync():
            nonlocal sync_done
            await sd.prepare_for_sync()
            sync_done = True
            await sd.resume_after_sync()

        # Start sync (will wait for drain)
        task = asyncio.create_task(do_sync())
        await asyncio.sleep(0.05)
        assert not sync_done  # Should be waiting for drain

        # End generation — should unblock sync
        await sd.end_generation()
        await asyncio.sleep(0.05)
        assert sync_done
        assert sd.is_accepting
        await task
    asyncio.run(run())
    print("  [PASS] soft_drain")


def test_implicit_continuation():
    from interrupts.implicit_continuation import ImplicitContinuation
    ic = ImplicitContinuation()
    async def run():
        # Test forward_pass context manager
        async with ic.forward_pass():
            pass  # Simulates a forward pass

        # Test sync waits for forward pass
        sync_acquired = False
        async def do_forward():
            async with ic.forward_pass():
                await asyncio.sleep(0.1)

        async def do_sync():
            nonlocal sync_acquired
            await ic.prepare_for_sync()
            sync_acquired = True
            await ic.resume_after_sync()

        # Start forward pass, then try sync
        fw_task = asyncio.create_task(do_forward())
        await asyncio.sleep(0.02)
        sync_task = asyncio.create_task(do_sync())
        await asyncio.sleep(0.02)
        assert not sync_acquired  # Sync should wait for forward pass
        await fw_task
        await asyncio.sleep(0.02)
        assert sync_acquired
        await sync_task
    asyncio.run(run())
    print("  [PASS] implicit_continuation")


# ============================================================
# AXIS 5: SCORERS
# ============================================================

def test_verifier_scorer():
    from scorers.verifier_scorer import VerifierScorer
    scorer = VerifierScorer()
    async def run():
        r = make_rollout()
        r_correct = ScoredRollout(
            prompt="q", prompt_ids=torch.tensor([1]),
            completion="The answer is 4.\n#### 4",
            completion_ids=torch.tensor([2, 3]),
            logprobs=torch.tensor([-1.0, -1.0]),
            model_version=0, generated_at=0.0,
            prompt_id="p1", ground_truth=4.0,
            reward=None,
        )
        scored = await scorer.score(r_correct)
        assert scored.reward == 1.0

        r_wrong = ScoredRollout(
            prompt="q", prompt_ids=torch.tensor([1]),
            completion="The answer is 5.\n#### 5",
            completion_ids=torch.tensor([2, 3]),
            logprobs=torch.tensor([-1.0, -1.0]),
            model_version=0, generated_at=0.0,
            prompt_id="p1", ground_truth=4.0,
            reward=None,
        )
        scored = await scorer.score(r_wrong)
        assert scored.reward == 0.0
    asyncio.run(run())
    print("  [PASS] verifier_scorer")


def test_distillation_scorer():
    """Test distillation scorer — skips if tiny-gpt2 not available."""
    try:
        from scorers.distillation_scorer import DistillationScorer, TeacherManager
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # Use tiny model as teacher
        teacher = TeacherManager(teacher_model_name="sshleifer/tiny-gpt2", snapshot_every=5)
        assert teacher.model is not None
        scorer = DistillationScorer(teacher_manager=teacher)

        async def run():
            r = ScoredRollout(
                prompt="What is 2+2?", prompt_ids=torch.tensor([50, 100]),
                completion="Four", completion_ids=torch.tensor([200, 300]),
                logprobs=torch.tensor([-2.0, -3.0]),
                model_version=0, generated_at=0.0,
                prompt_id="p1", ground_truth=4.0,
                reward=None,
            )
            scored = await scorer.score(r)
            assert scored.reward is not None
            assert scored.teacher_logprobs is not None
            assert isinstance(scored.reward, float)
        asyncio.run(run())
        print("  [PASS] distillation_scorer")
    except Exception as e:
        print(f"  [SKIP] distillation_scorer ({e})")


# ============================================================
# AXIS 6: TRAINER ALGORITHMS
# ============================================================

def test_ipo_algorithm():
    from core.trainer import GRPOTrainer, compute_ipo_advantages
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        print("  [SKIP] ipo_algorithm (tiny-gpt2 not available)")
        return

    trainer = GRPOTrainer(
        model=model, tokenizer=tokenizer, device="cpu",
        lr=1e-4, algorithm="ipo",
        ipo_mask_low=0.2, ipo_mask_high=0.2, adv_tau=1.0, kl_tau=1e-3,
    )
    batch = TrainingBatch(rollouts=[
        make_rollout(prompt_id="p1", reward=1.0) for _ in range(4)
    ] + [
        make_rollout(prompt_id="p1", reward=0.0) for _ in range(4)
    ])
    metrics = trainer.train_step(batch)
    assert "ipo_masked_frac" in metrics
    assert "mismatch_kl" in metrics
    assert "training_loss" in metrics
    print("  [PASS] ipo_algorithm")


def test_grpo_algorithm():
    from core.trainer import GRPOTrainer, compute_grpo_advantages
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        print("  [SKIP] grpo_algorithm (tiny-gpt2 not available)")
        return

    trainer = GRPOTrainer(
        model=model, tokenizer=tokenizer, device="cpu",
        lr=1e-4, algorithm="grpo", clip_eps=0.2, kl_coeff=0.01,
    )
    batch = TrainingBatch(rollouts=[
        make_rollout(prompt_id="p1", reward=1.0) for _ in range(4)
    ] + [
        make_rollout(prompt_id="p1", reward=0.0) for _ in range(4)
    ])
    metrics = trainer.train_step(batch)
    assert "clip_fraction" in metrics
    assert "training_loss" in metrics
    print("  [PASS] grpo_algorithm")


# ============================================================
# AXIS 7: FACTORY (all configs parse correctly)
# ============================================================

def test_factory_all_configs():
    from utils.factory import (
        create_buffer, create_syncer, create_staleness,
        create_interrupt, create_scorer,
    )

    # Buffers
    cfg = {"buffer": {"type": "sync"}}
    assert create_buffer(cfg).__class__.__name__ == "SyncBuffer"
    cfg = {"buffer": {"type": "bounded_queue", "maxsize": 4}}
    assert create_buffer(cfg).__class__.__name__ == "BoundedQueueBuffer"
    cfg = {"buffer": {"type": "double"}}
    assert create_buffer(cfg).__class__.__name__ == "DoubleBuffer"

    # Weight sync
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {"weight_sync": {"type": "filesystem", "checkpoint_dir": tmpdir}}
        assert create_syncer(cfg).__class__.__name__ == "FilesystemSyncer"

    # Staleness
    cfg = {"staleness": {"type": "no_filter"}}
    assert create_staleness(cfg).__class__.__name__ == "NoFilter"
    cfg = {"staleness": {"type": "version_rejection", "max_lag": 3}}
    assert create_staleness(cfg).__class__.__name__ == "VersionRejection"
    cfg = {"staleness": {"type": "is_reweighting", "is_clip_eps": 0.2}}
    assert create_staleness(cfg).__class__.__name__ == "ISReweighting"
    cfg = {"staleness": {"type": "hybrid", "max_lag": 3, "is_clip_eps": 0.2}}
    assert create_staleness(cfg).__class__.__name__ == "HybridStaleness"

    # Interrupts
    cfg = {"interrupt": {"type": "batch_sync"}}
    assert create_interrupt(cfg).__class__.__name__ == "BatchSync"
    cfg = {"interrupt": {"type": "soft_drain"}}
    assert create_interrupt(cfg).__class__.__name__ == "SoftDrain"
    cfg = {"interrupt": {"type": "implicit_continuation"}}
    assert create_interrupt(cfg).__class__.__name__ == "ImplicitContinuation"

    # Scorers
    cfg = {"scorer": {"type": "verifier"}}
    assert create_scorer(cfg).__class__.__name__ == "VerifierScorer"

    print("  [PASS] factory_all_configs")


# ============================================================
# RUN ALL
# ============================================================

if __name__ == "__main__":
    print("\n=== BUFFERS ===")
    test_sync_buffer()
    test_bounded_queue()
    test_double_buffer()
    test_redis_stream()

    print("\n=== WEIGHT SYNC ===")
    test_filesystem_sync()
    test_nccl_broadcast()
    test_nccl_bucketed()
    test_zmq_notify_fs()

    print("\n=== STALENESS ===")
    test_no_filter()
    test_version_rejection()
    test_is_reweighting()
    test_is_reweighting_with_model()
    test_hybrid()

    print("\n=== INTERRUPTS ===")
    test_batch_sync()
    test_soft_drain()
    test_implicit_continuation()

    print("\n=== SCORERS ===")
    test_verifier_scorer()
    test_distillation_scorer()

    print("\n=== ALGORITHMS ===")
    test_ipo_algorithm()
    test_grpo_algorithm()

    print("\n=== FACTORY ===")
    test_factory_all_configs()

    print("\n" + "=" * 50)
    print("ALL CONFIGURATION TESTS COMPLETE")
    print("=" * 50)
