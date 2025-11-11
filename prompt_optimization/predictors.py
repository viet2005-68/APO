from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from liquid import Template

import utils
import tasks

class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass

class BinaryPredictor(GPT4Predictor):
    categories = ['No', 'Yes']

    def inference(self, ex, prompt):
        prompt = Template(prompt).render(text=ex['text'])
        responses = utils.chatgpt(
            prompt, max_tokens=102400, n=1, timeout=30, 
            temperature=self.opt['temperature'])
        if not responses or len(responses) == 0:
            return 0  # Default to 0 if no response
        response = responses[0]
        if response is None or not isinstance(response, str):
            return 0  # Default to 0 if response is None or not a string
        pred = 1 if response.strip().upper().startswith('YES') else 0
        return pred
