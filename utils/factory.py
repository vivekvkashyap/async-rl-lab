"""Factory functions for creating pipeline components from config."""
import random
import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dtype(dtype_str: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16, "float16": torch.float16,
        "fp32": torch.float32, "float32": torch.float32,
    }[dtype_str]


def create_buffer(config: dict):
    from buffers.sync_buffer import SyncBuffer
    from buffers.bounded_queue import BoundedQueueBuffer
    from buffers.double_buffer import DoubleBuffer
    buf_type = config["buffer"]["type"]
    if buf_type == "sync":
        return SyncBuffer()
    if buf_type == "bounded_queue":
        return BoundedQueueBuffer(maxsize=config["buffer"].get("maxsize", 4))
    if buf_type == "double":
        return DoubleBuffer()
    if buf_type == "redis_stream":
        from buffers.redis_stream import RedisStreamBuffer
        return RedisStreamBuffer(redis_url=config["buffer"].get("redis_url", "redis://localhost"))
    raise ValueError(f"Unknown buffer type: {buf_type}")


def create_syncer(config: dict):
    from weight_sync.filesystem_sync import FilesystemSyncer
    sync_type = config["weight_sync"]["type"]
    if sync_type == "filesystem":
        return FilesystemSyncer(
            checkpoint_dir=config["weight_sync"].get("checkpoint_dir", "/tmp/async-rl-lab/checkpoints")
        )
    if sync_type == "nccl_broadcast":
        from weight_sync.nccl_broadcast import NCCLBroadcastSyncer
        return NCCLBroadcastSyncer(rank=0)
    if sync_type == "nccl_bucketed":
        from weight_sync.nccl_bucketed import NCCLBucketedSyncer
        return NCCLBucketedSyncer(rank=0, bucket_size_mb=config["weight_sync"].get("bucket_size_mb", 1024))
    if sync_type == "zmq_notify_fs":
        from weight_sync.zmq_notify_fs import ZMQNotifyFSSyncer
        return ZMQNotifyFSSyncer(
            checkpoint_dir=config["weight_sync"].get("checkpoint_dir", "/tmp/async-rl-lab/checkpoints"),
            zmq_port=config["weight_sync"].get("zmq_port", 5555), role="publisher",
        )
    raise ValueError(f"Unknown weight_sync type: {sync_type}")


def create_staleness(config: dict, model=None, tokenizer=None, device="cuda:0"):
    from staleness.no_filter import NoFilter
    from staleness.version_rejection import VersionRejection
    from staleness.is_reweighting import ISReweighting
    from staleness.hybrid import HybridStaleness
    st_type = config["staleness"]["type"]
    if st_type == "no_filter":
        return NoFilter()
    if st_type == "version_rejection":
        return VersionRejection(max_lag=config["staleness"].get("max_lag", 3))
    if st_type == "is_reweighting":
        return ISReweighting(clip_eps=config["staleness"].get("is_clip_eps", 0.2),
                             model=model, tokenizer=tokenizer, device=device)
    if st_type == "hybrid":
        return HybridStaleness(max_lag=config["staleness"].get("max_lag", 3),
                               is_clip_eps=config["staleness"].get("is_clip_eps", 0.2),
                               model=model, tokenizer=tokenizer, device=device)
    raise ValueError(f"Unknown staleness type: {st_type}")


def create_interrupt(config: dict):
    from interrupts.batch_sync import BatchSync
    from interrupts.soft_drain import SoftDrain
    from interrupts.implicit_continuation import ImplicitContinuation
    int_type = config["interrupt"]["type"]
    if int_type == "batch_sync":
        return BatchSync()
    if int_type == "soft_drain":
        return SoftDrain()
    if int_type == "implicit_continuation":
        return ImplicitContinuation()
    raise ValueError(f"Unknown interrupt type: {int_type}")


def create_scorer(config: dict):
    from scorers.verifier_scorer import VerifierScorer
    sc_type = config["scorer"]["type"]
    if sc_type == "verifier":
        return VerifierScorer()
    if sc_type == "distillation":
        from scorers.distillation_scorer import DistillationScorer, TeacherManager
        teacher = TeacherManager(teacher_model_name=config["scorer"].get("teacher_model"),
                                 snapshot_every=config["scorer"].get("teacher_snapshot_every", 10))
        return DistillationScorer(teacher_manager=teacher)
    raise ValueError(f"Unknown scorer type: {sc_type}")


def create_trainer(model, tokenizer, device, config):
    """Create GRPOTrainer with algorithm-specific params from config."""
    from core.trainer import GRPOTrainer
    training = config["training"]
    algorithm = training.get("algorithm", "ipo")

    return GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        lr=training["lr"],
        algorithm=algorithm,
        # GRPO params
        clip_eps=training.get("clip_eps", 0.2),
        kl_coeff=training.get("kl_coeff", 0.01),
        # IPO params
        ipo_mask_low=training.get("ipo_mask_low", 0.2),
        ipo_mask_high=training.get("ipo_mask_high", 0.2),
        adv_tau=training.get("adv_tau", 1.0),
        kl_tau=training.get("kl_tau", 1e-3),
        # Shared
        max_grad_norm=training.get("max_grad_norm", 1.0),
    )
