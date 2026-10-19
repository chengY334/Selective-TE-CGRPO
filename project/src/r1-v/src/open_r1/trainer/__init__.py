# TE-CGRPO modification: defer the optional legacy vLLM backend import.
from .grpo_trainer import Qwen2VLGRPOTrainer


def __getattr__(name):
    # A non-vLLM job must not require importing the optional vLLM backend.
    if name == "Qwen2VLGRPOVLLMTrainerModified":
        from .vllm_grpo_trainer_modified import Qwen2VLGRPOVLLMTrainerModified
        return Qwen2VLGRPOVLLMTrainerModified
    raise AttributeError(name)

__all__ = [
    "Qwen2VLGRPOTrainer",
    "Qwen2VLGRPOVLLMTrainerModified"
]
