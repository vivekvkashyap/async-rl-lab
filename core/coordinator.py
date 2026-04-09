import asyncio
import multiprocessing as mp
import random
import time
import torch
import torch.distributed as dist
from typing import Dict, Any, List, Optional
from core.types import TrainingBatch, ScoredRollout
from core.trainer import GRPOTrainer
from buffers.base import RolloutBuffer
from weight_sync.base import WeightSyncer
from staleness.base import StalenessManager
from interrupts.base import InterruptStrategy
from scorers.base import Scorer
from utils.metrics import MetricsTracker
from utils.gsm8k import load_gsm8k, extract_model_answer


class Coordinator:
    """Asyncio event loop orchestrating the async RL pipeline.

    Training runs in the main process on GPU 0.
    Inference processes are started separately (see launch_inference_process).
    They communicate via a shared multiprocessing.Queue.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        trainer: GRPOTrainer,
        buffer: RolloutBuffer,
        syncer: WeightSyncer,
        staleness: StalenessManager,
        interrupt: InterruptStrategy,
        scorer: Scorer,
        metrics: MetricsTracker,
        rollout_queue: mp.Queue,
        version_val: mp.Value,
        stop_event: mp.Event,
        infer_gpu_ids: List[int],
    ):
        self.config = config
        self.trainer = trainer
        self.buffer = buffer
        self.syncer = syncer
        self.staleness = staleness
        self.interrupt = interrupt
        self.scorer = scorer
        self.metrics = metrics
        self.rollout_queue = rollout_queue
        self.version_val = version_val
        self.stop_event = stop_event
        self.infer_gpu_ids = infer_gpu_ids

        self.train_data = load_gsm8k("train")
        self.eval_data = None  # loaded lazily on first eval

        self.max_steps = config["training"]["max_steps"]
        self.batch_size = config["training"]["batch_size"]
        self.group_size = config["training"]["group_size"]
        self.eval_every = config.get("metrics", {}).get("eval_every", 0)
        self.eval_samples = config.get("metrics", {}).get("eval_samples", 200)

    async def _collect_from_queue(self, needed: int, timeout: float = 120.0) -> List[ScoredRollout]:
        """Collect rollouts from inference processes via shared queue."""
        loop = asyncio.get_event_loop()
        batch = []
        deadline = time.time() + timeout
        while len(batch) < needed and time.time() < deadline:
            try:
                item = await loop.run_in_executor(
                    None, lambda: self.rollout_queue.get(timeout=2)
                )
                batch.append(item)
            except Exception:
                pass
        return batch

    def _get_gpu_utilization(self) -> Dict[str, float]:
        """Snapshot GPU compute utilization via pynvml."""
        result = {}
        try:
            import pynvml
            pynvml.nvmlInit()
            # Training GPU
            train_gpu_idx = int(self.trainer.device.split(":")[-1]) if "cuda" in self.trainer.device else 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(train_gpu_idx)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            result["gpu_util_train"] = util.gpu / 100.0
            result["gpu_mem_train"] = util.memory / 100.0

            # Inference GPUs (average)
            infer_utils = []
            for gpu_id in self.infer_gpu_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                infer_utils.append(util.gpu)
            if infer_utils:
                result["gpu_util_infer"] = sum(infer_utils) / len(infer_utils) / 100.0
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return result

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run held-out GSM8K evaluation. Returns accuracy (0-1)."""
        from utils.gsm8k import load_gsm8k, extract_answer, extract_model_answer, format_prompt

        if self.eval_data is None:
            self.eval_data = load_gsm8k("test")

        samples = self.eval_data[:self.eval_samples]
        self.trainer.model.eval()

        correct = 0
        total = 0
        for sample in samples:
            prompt_text = format_prompt(sample["question"])
            inputs = self.trainer.tokenizer(
                prompt_text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.trainer.device)

            outputs = self.trainer.model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=self.trainer.tokenizer.pad_token_id,
            )
            generated = self.trainer.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            pred = extract_model_answer(generated)
            gold = extract_answer(sample["answer"])
            if pred is not None and gold is not None and abs(pred - gold) < 1e-3:
                correct += 1
            total += 1

        self.trainer.model.train()
        return correct / max(total, 1)

    async def run(self):
        n_infer = len(self.infer_gpu_ids)
        algo = self.trainer.algorithm.upper()
        print(f"Starting {algo} training for {self.max_steps} steps")
        print(f"  Train GPU: {self.trainer.device} | Infer GPUs: {self.infer_gpu_ids} (vLLM)")
        print(f"  batch_size={self.batch_size}, group_size={self.group_size}")
        print(f"  buffer={self.buffer.__class__.__name__}")
        print(f"  staleness={self.staleness.__class__.__name__}")
        print(f"  syncer={self.syncer.__class__.__name__}")
        if self.eval_every > 0:
            print(f"  eval_every={self.eval_every}, eval_samples={self.eval_samples}")

        print(f"\nWaiting for inference processes to generate first batch...")

        try:
            await self._training_loop()
        finally:
            # Signal FSDP followers to stop
            if self.trainer.is_fsdp:
                signal = torch.tensor([-1], dtype=torch.long, device=self.trainer.device)
                dist.broadcast(signal, src=0)
            self.stop_event.set()

        path = self.metrics.save()
        print(f"\nTraining complete. Metrics saved to {path}")
        summary = self.metrics.summary()
        print(f"Final loss: {summary.get('training_loss_last', 'N/A')}")
        print(f"Final reward mean: {summary.get('reward_mean_last', 'N/A')}")
        if "gsm8k_accuracy_last" in summary:
            print(f"Final GSM8K accuracy: {summary['gsm8k_accuracy_last']:.4f}")
        self.metrics.finish()

    async def _training_loop(self):
        for step in range(self.max_steps):
            step_start = time.time()

            # Collect rollouts from vLLM inference processes
            gen_start = time.time()
            scored = await self._collect_from_queue(self.batch_size)
            gen_time = time.time() - gen_start

            if not scored:
                print(f"[step {step}] No rollouts collected, skipping...")
                if self.trainer.is_fsdp:
                    signal = torch.zeros(1, dtype=torch.long, device=self.trainer.device)
                    dist.broadcast(signal, src=0)
                continue

            for s in scored:
                await self.buffer.put(s)

            batch_rollouts = await self.buffer.get(min(self.batch_size, len(scored)))

            # Apply staleness filter/reweight
            batch = self.staleness.process(batch_rollouts, self.trainer.version)

            if not batch.rollouts:
                print(f"[step {step}] All rollouts filtered by staleness, skipping...")
                if self.trainer.is_fsdp:
                    signal = torch.zeros(1, dtype=torch.long, device=self.trainer.device)
                    dist.broadcast(signal, src=0)
                continue

            # Train
            train_start = time.time()
            train_metrics = self.trainer.train_step(batch)
            train_time = time.time() - train_start

            # Weight sync
            await self.interrupt.prepare_for_sync()
            sync_time = await self.syncer.push(self.trainer.model, self.trainer.version)
            await self.interrupt.resume_after_sync()

            # Notify inference of new version
            self.version_val.value = self.trainer.version

            # Log metrics
            step_metrics = {
                **train_metrics,
                "generation_time": gen_time,
                "train_time": train_time,
                "sync_duration": sync_time,
                "buffer_depth": self.buffer.size(),
                "num_infer_gpus": len(self.infer_gpu_ids),
                "batch_staleness_mean": sum(
                    self.trainer.version - r.model_version for r in batch_rollouts
                ) / max(len(batch_rollouts), 1),
                "tokens_per_second": sum(
                    r.completion_ids.shape[0] for r in scored
                ) / max(gen_time, 1e-6),
                "wall_clock_time": time.time() - step_start,
            }

            # GPU utilization snapshot
            gpu_util = self._get_gpu_utilization()
            step_metrics.update(gpu_util)

            # Periodic evaluation on held-out GSM8K test set
            # Skip eval with FSDP — model.generate() requires full (unsharded) model
            if self.eval_every > 0 and (step + 1) % self.eval_every == 0 and not self.trainer.is_fsdp:
                acc = self.evaluate()
                step_metrics["gsm8k_accuracy"] = acc
                print(f"  [eval] GSM8K accuracy: {acc:.4f} ({int(acc * self.eval_samples)}/{self.eval_samples})")

            self.metrics.log(step, step_metrics)
