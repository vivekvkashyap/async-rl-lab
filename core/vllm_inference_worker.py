"""vLLM-based inference worker for high-throughput generation.

Uses vLLM's LLM engine for PagedAttention, continuous batching,
and much higher throughput.

IMPORTANT: CUDA_VISIBLE_DEVICES must be set BEFORE this module is imported
in the inference process. The coordinator handles this.
"""
import time
import torch
from typing import List, Dict
from core.types import Rollout
from utils.gsm8k import format_prompt


class VLLMInferenceWorker:
    """Inference worker using vLLM for fast generation."""

    def __init__(
        self,
        model_name: str,
        config: dict,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
    ):
        from vllm import LLM, SamplingParams

        self.model_name = model_name
        self.group_size = config.get("group_size", 8)
        self.max_new_tokens = config.get("max_completion_length", 512)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)

        # vLLM sees only the GPUs set via CUDA_VISIBLE_DEVICES
        # so we always use device index 0 within this process
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=2048,
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            logprobs=1,
        )

    def generate_rollouts(
        self, prompts: List[Dict], model_version: int
    ) -> List[Rollout]:
        """Generate group_size completions per prompt using vLLM."""
        all_texts = []
        all_prompt_data = []
        for prompt_data in prompts:
            text = format_prompt(prompt_data["question"], tokenizer=self.tokenizer)
            for _ in range(self.group_size):
                all_texts.append(text)
                all_prompt_data.append(prompt_data)

        outputs = self.llm.generate(all_texts, self.sampling_params)

        rollouts = []
        for i, output in enumerate(outputs):
            prompt_data = all_prompt_data[i]
            prompt_text = all_texts[i]
            prompt_ids = torch.tensor(output.prompt_token_ids, dtype=torch.long)

            for completion in output.outputs:
                completion_ids = torch.tensor(completion.token_ids, dtype=torch.long)

                logprobs_list = []
                if completion.logprobs:
                    for lp_dict in completion.logprobs:
                        if lp_dict:
                            top_lp = next(iter(lp_dict.values()))
                            logprobs_list.append(top_lp.logprob)
                        else:
                            logprobs_list.append(0.0)

                logprobs = torch.tensor(logprobs_list, dtype=torch.float32)

                rollouts.append(Rollout(
                    prompt=prompt_text,
                    prompt_ids=prompt_ids,
                    completion=completion.text,
                    completion_ids=completion_ids,
                    logprobs=logprobs,
                    model_version=model_version,
                    generated_at=time.time(),
                    prompt_id=prompt_data["id"],
                    ground_truth=prompt_data["answer"],
                    sampling_mask=None,
                ))

        return rollouts

    def update_weights(self, weight_dir: str):
        """Hot-swap model weights without restarting the engine."""
        try:
            self.llm.collective_rpc(
                "reload_weights",
                kwargs={"weights_path": weight_dir, "is_checkpoint_format": True},
            )
            self.llm.reset_prefix_cache()
            return True
        except Exception as e:
            print(f"[vLLM] Weight update failed: {e}. Continuing with current weights.")
            return False
