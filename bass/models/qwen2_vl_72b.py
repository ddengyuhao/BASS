import torch
from transformers import Qwen2VLForConditionalGeneration

from .qwen_base import QwenVLBase


class Qwen2_VL_72B_Wrapper(QwenVLBase):
    model_cls = Qwen2VLForConditionalGeneration
    default_path = "Qwen/Qwen2-VL-72B-Instruct"

    def __init__(self, model_path=None):
        if torch.cuda.device_count() < 2:
            print("Warning: the 72B model typically requires 2+ A100 GPUs.")
        super().__init__(model_path)
