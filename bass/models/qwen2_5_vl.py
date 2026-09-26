from transformers import Qwen2_5_VLForConditionalGeneration

from .qwen_base import QwenVLBase


class Qwen2_5_VLWrapper(QwenVLBase):
    model_cls = Qwen2_5_VLForConditionalGeneration
    default_path = "Qwen/Qwen2.5-VL-7B-Instruct"
