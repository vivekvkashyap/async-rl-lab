"""FSDP utilities for distributed training across multiple GPUs.

Handles:
  - Device mesh creation for HSDP (Hybrid Sharded Data Parallel)
  - Model wrapping with FSDP2 (per-parameter sharding)
  - Weight gathering from FSDP shards to full tensors on rank 0
  - Mixed precision configuration

Following prime-rl's approach: wrap individual transformer blocks with FSDP,
then wrap the full model. This enables per-block gradient checkpointing and
efficient prefetching.
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from typing import Optional, Dict, Set
import functools


def setup_distributed(rank: int, world_size: int, master_port: int = 29500):
    """Initialize torch.distributed process group for FSDP."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Destroy the process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_transformer_block_class(model: nn.Module):
    """Auto-detect the transformer block class for FSDP wrapping.

    Looks for common transformer block class names across model architectures:
    Qwen2DecoderLayer, LlamaDecoderLayer, MistralDecoderLayer, etc.
    """
    block_class_names = {
        "Qwen2DecoderLayer", "Qwen2MoeDecoderLayer",
        "LlamaDecoderLayer",
        "MistralDecoderLayer",
        "GemmaDecoderLayer", "Gemma2DecoderLayer",
        "Phi3DecoderLayer",
        "GPTNeoXLayer",
        "GPT2Block",
        "BloomBlock",
        "FalconDecoderLayer",
        "DeepseekV2DecoderLayer", "DeepseekV3DecoderLayer",
    }

    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name in block_class_names:
            return type(module)

    # Fallback: look for any module named "layers" that is a ModuleList
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            child = module[0]
            # Heuristic: if it has self_attn and mlp, it's a transformer block
            if hasattr(child, "self_attn") and hasattr(child, "mlp"):
                return type(child)

    return None


def wrap_model_fsdp(
    model: nn.Module,
    train_gpu_ids: list,
    dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = False,
    reshard_after_forward: bool = True,
) -> nn.Module:
    """Wrap a HuggingFace model with FSDP for multi-GPU training.

    Args:
        model: The model to wrap (should already be on the correct device).
        train_gpu_ids: List of GPU IDs used for training.
        dtype: Parameter dtype for mixed precision.
        cpu_offload: Whether to offload parameters to CPU.
        reshard_after_forward: Reshard params after forward (saves memory, more comm).

    Returns:
        FSDP-wrapped model.
    """
    rank = dist.get_rank()
    device = torch.device(f"cuda:{train_gpu_ids[rank]}")

    # Mixed precision policy
    mp_policy = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )

    # CPU offload policy
    offload = CPUOffload(offload_params=True) if cpu_offload else None

    # Sharding strategy: FULL_SHARD across all training GPUs
    # With >2 GPUs, HYBRID_SHARD could be used (replicate across nodes, shard within)
    # For single-node, FULL_SHARD is optimal
    sharding_strategy = ShardingStrategy.FULL_SHARD

    # Auto-detect transformer block class for wrapping policy
    block_class = get_transformer_block_class(model)

    if block_class is not None:
        # Wrap individual transformer blocks — enables per-block checkpointing
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={block_class},
        )
    else:
        auto_wrap_policy = None

    # Wrap with FSDP
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mp_policy,
        cpu_offload=offload,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
        use_orig_params=True,  # Required for torch.compile compatibility
    )

    if rank == 0:
        print(f"[FSDP] Wrapped model with {sharding_strategy.name}, "
              f"block_class={block_class.__name__ if block_class else 'None'}, "
              f"dtype={dtype}, cpu_offload={cpu_offload}")

    return model


def gather_weights_on_master(
    model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
) -> Optional[Dict[str, torch.Tensor]]:
    """Gather FSDP-sharded weights into a full state dict on rank 0.

    All ranks must call this (it involves collective communication).
    Only rank 0 gets the full state dict; other ranks get None.

    Following prime-rl's gather_weights_on_master approach.
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    # Configure to gather full state dict on rank 0 only
    full_sd_config = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_sd_config):
        state_dict = model.state_dict()

    rank = dist.get_rank()
    if rank == 0:
        # Cast to target dtype and ensure contiguous
        result = {}
        seen_data_ptrs = {}
        for k, v in state_dict.items():
            v = v.to(dtype).contiguous()
            ptr = v.data_ptr()
            if ptr in seen_data_ptrs:
                result[k] = v.clone()
            else:
                seen_data_ptrs[ptr] = k
                result[k] = v
        return result

    return None


def is_fsdp_wrapped(model: nn.Module) -> bool:
    """Check if a model is wrapped with FSDP."""
    return isinstance(model, FSDP)


def broadcast_batch_tensors(tensors: Dict[str, torch.Tensor], device: torch.device):
    """Rank 0 broadcasts batch tensors to all FSDP ranks."""
    keys = ["input_ids", "old_logprobs", "loss_mask", "advantages"]
    for key in keys:
        t = tensors[key]
        shape_tensor = torch.tensor(t.shape, dtype=torch.long, device=device)
        dist.broadcast(shape_tensor, src=0)
        dist.broadcast(t.contiguous(), src=0)

    has_is_weights = torch.tensor(
        [1 if "is_weights" in tensors else 0], dtype=torch.long, device=device
    )
    dist.broadcast(has_is_weights, src=0)
    if has_is_weights.item() == 1:
        t = tensors["is_weights"]
        shape_tensor = torch.tensor(t.shape, dtype=torch.long, device=device)
        dist.broadcast(shape_tensor, src=0)
        dist.broadcast(t.contiguous(), src=0)


def receive_batch_tensors(device: torch.device) -> Dict[str, torch.Tensor]:
    """Non-master rank receives batch tensors from rank 0."""
    tensors = {}
    keys = ["input_ids", "old_logprobs", "loss_mask", "advantages"]
    dtypes = [torch.long, torch.float32, torch.float32, torch.float32]

    for key, dt in zip(keys, dtypes):
        shape_tensor = torch.zeros(2, dtype=torch.long, device=device)
        dist.broadcast(shape_tensor, src=0)
        shape = tuple(shape_tensor.tolist())
        t = torch.zeros(shape, dtype=dt, device=device)
        dist.broadcast(t, src=0)
        tensors[key] = t

    has_is_weights = torch.zeros(1, dtype=torch.long, device=device)
    dist.broadcast(has_is_weights, src=0)
    if has_is_weights.item() == 1:
        shape_tensor = torch.zeros(1, dtype=torch.long, device=device)
        dist.broadcast(shape_tensor, src=0)
        shape = tuple(shape_tensor.tolist())
        t = torch.zeros(shape, dtype=torch.float32, device=device)
        dist.broadcast(t, src=0)
        tensors["is_weights"] = t

    return tensors


def run_fsdp_follower(trainer, stop_event, config: dict):
    """Non-master FSDP rank: mirrors rank 0's training steps.

    Listens for signals from rank 0 and performs synchronized
    forward/backward passes through FSDP.
    """
    from core.trainer import selective_log_softmax

    max_steps = config["training"]["max_steps"]

    for step in range(max_steps):
        if stop_event.is_set():
            break

        # Receive signal from rank 0: 1 = train step, 0 = skip, -1 = stop
        signal = torch.zeros(1, dtype=torch.long, device=trainer.device)
        dist.broadcast(signal, src=0)

        if signal.item() == -1:
            break
        if signal.item() == 0:
            continue

        # Receive batch data from rank 0
        tensors = receive_batch_tensors(trainer.device)

        # Synchronized forward/backward (FSDP handles the sharding)
        trainer.model.train()

        input_ids = tensors["input_ids"]
        old_logprobs = tensors["old_logprobs"]
        loss_mask = tensors["loss_mask"]
        advantages = tensors["advantages"]
        is_weights = tensors.get("is_weights")

        attention_mask = (input_ids != (trainer.tokenizer.pad_token_id or 0)).long()
        outputs = trainer.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        current_logprobs = selective_log_softmax(logits, target_ids)

        old_logprobs = old_logprobs[:, 1:]
        loss_mask = loss_mask[:, 1:]
        advantages = advantages[:, 1:]

        # Use same loss function as rank 0
        if trainer.algorithm == "ipo":
            total_loss, _ = trainer._compute_loss_ipo(
                current_logprobs, old_logprobs, loss_mask, advantages, is_weights
            )
        else:
            total_loss, _ = trainer._compute_loss_grpo(
                current_logprobs, old_logprobs, loss_mask, advantages, is_weights
            )

        trainer.optimizer.zero_grad()
        total_loss.backward()
        trainer.model.clip_grad_norm_(trainer.max_grad_norm)
        trainer.optimizer.step()
        trainer.version += 1

        # Weight sync: all ranks must participate in gather
        gather_weights_on_master(trainer.model)
        dist.barrier()
