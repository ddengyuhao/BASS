"""
Proprietary LMM backbones (Sec. 5.7, Table 4).

GPT-4o uses the OpenAI API (OPENAI_API_KEY); Gemini uses the Google GenAI API
(GEMINI_API_KEY). Both expose the same interface as the open-source backbones,
with greedy decoding (temperature 0.0, top-p 1.0).
"""

import io
import time
import base64


def _retry(fn, attempts=5, wait=5.0):
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:
            if k == attempts - 1:
                raise
            print(f"API call failed ({e}); retrying in {wait * (k + 1):.0f}s")
            time.sleep(wait * (k + 1))


class GPT4oWrapper:
    def __init__(self, model_name=None, detail="high"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name or "gpt-4o-2024-05-13"
        self.detail = detail
        self._image_tokens = {}

    def _image_part(self, img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return {"type": "image_url", "image_url": {"url": url, "detail": self.detail}}

    def _create(self, content, max_new_tokens):
        return _retry(lambda: self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
        ))

    def count_image_tokens(self, image):
        """Measured from the API: prompt tokens with the image minus without it."""
        key = image.size
        if key not in self._image_tokens:
            text = {"type": "text", "text": "."}
            with_img = self._create([self._image_part(image), text], 1).usage.prompt_tokens
            without = self._create([text], 1).usage.prompt_tokens
            self._image_tokens[key] = with_img - without
        return self._image_tokens[key]

    def generate(self, frames, prompt, max_new_tokens=512, **kwargs):
        content = [self._image_part(img) for img in frames]
        content.append({"type": "text", "text": prompt})
        return self._create(content, max_new_tokens).choices[0].message.content or ""

    def generate_text(self, prompt, max_new_tokens=512, **kwargs):
        return self._create([{"type": "text", "text": prompt}], max_new_tokens).choices[0].message.content or ""


class GeminiWrapper:
    def __init__(self, model_name=None):
        from google import genai
        from google.genai import types
        self.client = genai.Client()
        self.types = types
        self.model_name = model_name or "gemini-1.5-pro"

    def count_image_tokens(self, image):
        return int(self.client.models.count_tokens(
            model=self.model_name, contents=[image]).total_tokens)

    def _generate(self, contents, max_new_tokens):
        config = self.types.GenerateContentConfig(
            temperature=0.0, top_p=1.0, max_output_tokens=max_new_tokens)
        resp = _retry(lambda: self.client.models.generate_content(
            model=self.model_name, contents=contents, config=config))
        return resp.text or ""

    def generate(self, frames, prompt, max_new_tokens=512, **kwargs):
        return self._generate(list(frames) + [prompt], max_new_tokens)

    def generate_text(self, prompt, max_new_tokens=512, **kwargs):
        return self._generate([prompt], max_new_tokens)
