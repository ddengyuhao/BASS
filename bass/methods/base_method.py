from abc import ABC, abstractmethod


class BaseMethod(ABC):
    def __init__(self, args, model):
        """
        Args:
            args: parsed command-line arguments (budget and hyperparameters).
            model: loaded LMM backbone wrapper.
        """
        self.args = args
        self.model = model
        self.token_budget = args.token_budget

    @abstractmethod
    def process_and_inference(self, video_path, question, options):
        """
        Answers a multiple-choice query over a video.

        Args:
            video_path (str): path to the video file.
            question (str): the user query.
            options (list): candidate options.

        Returns:
            str: the model response, ending with "The answer is X."
        """
