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
            prompt, max_tokens=1024, n=1, timeout=30, 
            temperature=self.opt['temperature'])
        if not responses or len(responses) == 0:
            return 0  # Default to 0 if no response
        response = responses[0]
        if response is None or not isinstance(response, str):
            return 0  # Default to 0 if response is None or not a string
        pred = 1 if response.strip().upper().startswith('YES') else 0
        return pred

    def inference_with_conf(self, ex, prompt):
        prompt = Template(prompt).render(text=ex['text'])
        responses_pred, responses_conf = utils.chatgpt_with_confidence(
            prompt, max_tokens=1024, n=1, timeout=30, 
            temperature=self.opt['temperature'])
        if not responses_pred or len(responses_pred) == 0:
            return 0, 0.5  # Default to 0 if no response
        response_pred = responses_pred[0]
        response_conf = float(responses_conf[0])
        if response_pred is None or not isinstance(response_pred, str):
            return 0, 0.5  # Default to 0 if response is None or not a string
        pred = 1 if response_pred.strip().upper().startswith('YES') else 0
        return pred, response_conf
