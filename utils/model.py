from typing import Dict

from models.base_model import BaseModelWrapper
from models.instruct_model import InstructModel, LargeInstructModel
from models.image_instruct_model import ImageInstructModel
from models.qwen_vl_image_model import QwenVLImageModel
from models.api_model import APIModelWrapper

try:
    from models.chatts_model import ChatTSHFWrapper
except ImportError:
    ChatTSHFWrapper = None
    print("ChatTS HF model unavailable — missing dependency")

try:
    from models.vllm_chatts_model import ChatTSVLLMWrapper
except ImportError:
    ChatTSVLLMWrapper = None
    print("ChatTS vLLM model unavailable — vllm not installed")

from models.baselines import (
    RandomBaseline, KNNBaseline,
    ZeroedTSBaseline, EmptyTSBaseline, EmptyAllTSBaseline,
    EmptyAllTSChatTSBaseline, DinoKNNCLSABaseline,
)

method_wrapper_dict: Dict[str, BaseModelWrapper] = {
    # --- Small text LLMs (HF, single RTX 4090) ---
    "Qwen/Qwen3-4B-Instruct-2507": InstructModel,
    "Qwen/Qwen3-0.6B": InstructModel,
    "Qwen/Qwen2.5-7B-Instruct": InstructModel,
    "Qwen/Qwen2-7B-Instruct": InstructModel,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": InstructModel,
    "mistralai/Mistral-7B-Instruct-v0.3": InstructModel,

    # --- Large text LLMs (vLLM, 2× RTX 6000) ---
    "Qwen/Qwen3.6-27B": LargeInstructModel,
    "Qwen/Qwen3.6-27B-FP8": LargeInstructModel,

    # --- Vision LLMs (input_mode="separate": receive <ts><ts/> placeholders) ---
    "Qwen/Qwen3.6-27B-image-ts": ImageInstructModel,   # TS → matplotlib plot, vLLM
    "Qwen/Qwen3-VL-8B-Instruct": QwenVLImageModel,     # TS → matplotlib plot, HF

    # --- ChatTS (input_mode="separate": patch-embedding TS tokens) ---
    "bytedance-research/ChatTS-8B": ChatTSHFWrapper,
    "bytedance-research/ChatTS-14B": ChatTSHFWrapper,
    "bytedance-research/ChatTS-8B-vllm": ChatTSVLLMWrapper,
    "bytedance-research/ChatTS-14B-vllm": ChatTSVLLMWrapper,

    # --- Cloud API models ---
    "openai": APIModelWrapper,
    "openai_o1": APIModelWrapper,
    "anthropic": APIModelWrapper,
    "gemini": APIModelWrapper,
    "deepseek_v3": APIModelWrapper,
    "ollama": APIModelWrapper,

    # --- Baselines (CPU / no weights) ---
    "random_baseline": RandomBaseline,
    "knn_baseline": KNNBaseline,
    "zeroed_ts_baseline": ZeroedTSBaseline,
    "empty_ts_baseline": EmptyTSBaseline,
    "empty_all_ts_baseline": EmptyAllTSBaseline,        # needs --quantization 8bit
    "empty_all_ts_chatts_baseline": EmptyAllTSChatTSBaseline,
    "dino_knn_clsa_baseline": DinoKNNCLSABaseline,      # delay-embed + DINOv2-Large 1-NN
}
