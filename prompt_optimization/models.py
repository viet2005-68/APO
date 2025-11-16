class Prompt():
    def __init__(self, prompt, feedbacks_idx_used, examplers_idx_used, parent_score, score=0):
        self.prompt = prompt
        self.feedbacks_idx_used = feedbacks_idx_used
        self.examplers_idx_used = examplers_idx_used
        self.parent_score = parent_score
        self.score = score