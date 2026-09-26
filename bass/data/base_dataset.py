import os

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Multiple-choice long-video QA dataset. Subclasses populate `self.samples`
    with dicts of the form:
        {"id", "video_path", "question", "options", "answer"}
    where `options` is a list of option strings and `answer` is an option letter.
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def resolve_video(candidates):
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None
