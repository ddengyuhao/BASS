import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenVLBase:
    """Shared generation logic for the Qwen-VL backbones."""

    model_cls = None
    default_path = None

    def __init__(self, model_path=None):
        model_path = model_path or self.default_path
        print(f"[{self.__class__.__name__}] Loading model from {model_path} ...")
        try:
            self.model = self.model_cls.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        except Exception as e:
            print(f"Flash Attention load failed, falling back to default attention: {e}")
            self.model = self.model_cls.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def count_image_tokens(self, image):
        """Number of visual tokens the LMM produces for one image."""
        out = self.processor.image_processor(images=[image], return_tensors="pt")
        merge = getattr(self.processor.image_processor, "merge_size", 2)
        return int(out["image_grid_thw"].prod(dim=-1).sum()) // (merge * merge)

    def generate(self, frames, prompt, max_new_tokens=512, **kwargs):
        """
        Args:
            frames: list of PIL images, passed to the model as separate images.
            prompt: text prompt.
        """
        content = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": prompt})
        return self._run([{"role": "user", "content": content}], max_new_tokens, **kwargs)

    def generate_text(self, prompt, max_new_tokens=512, **kwargs):
        content = [{"type": "text", "text": prompt}]
        return self._run([{"role": "user", "content": content}], max_new_tokens, **kwargs)

    def _run(self, messages, max_new_tokens, **kwargs):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Greedy decoding (temperature 0.0, top-p 1.0)
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False, **kwargs}
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
