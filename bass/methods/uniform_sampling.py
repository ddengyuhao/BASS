import torch
from PIL import Image

from .base_method import BaseMethod
from ..utils import format_options

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None


DIRECT_PROMPT = (
    "The images are frames uniformly sampled from a video, in temporal order.\n\n"
    "Question: {question}\n"
    "Options:\n{options}\n\n"
    "End your response with 'The answer is X.'"
)


class UniformSampling(BaseMethod):
    """
    Uniform-sampling baseline under the same visual-token budget (Sec. 5.7, Table 4):
    floor(B / tokens_per_frame) frames are sampled uniformly over the whole video and
    passed to the LMM in a single call.
    """

    def __init__(self, args, model):
        super().__init__(args, model)
        self.frame_size = getattr(args, 'frame_size', 336)
        self.max_new_tokens = getattr(args, 'max_new_tokens', 512)
        probe = Image.new("RGB", (self.frame_size, self.frame_size))
        self.tokens_per_frame = self.model.count_image_tokens(probe)
        self.num_frames = max(1, self.token_budget // self.tokens_per_frame)
        self.last_trace = None

    def process_and_inference(self, video_path, question, options):
        if VideoReader is None:
            raise ImportError("decord is required for video reading.")
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        n = min(self.num_frames, total)
        ids = torch.linspace(0, total - 1, n + 2)[1:-1].round().long().tolist() if n < total \
            else list(range(total))
        frames = [Image.fromarray(f).resize((self.frame_size, self.frame_size), Image.BICUBIC)
                  for f in vr.get_batch(ids).asnumpy()]
        del vr

        prompt = DIRECT_PROMPT.format(question=question, options=format_options(options))
        reply = self.model.generate(frames, prompt, max_new_tokens=self.max_new_tokens)
        self.last_trace = {"num_frames": len(frames),
                           "visual_tokens": float(len(frames) * self.tokens_per_frame)}
        return reply
