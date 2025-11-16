class Prompt():
    def __init__(self, prompt, feedbacks_idx_used, examplers_idx_used, parent_score, score=0):
        self.prompt = prompt
        self.feedbacks_idx_used = feedbacks_idx_used
        self.examplers_idx_used = examplers_idx_used
        self.parent_score = parent_score
        self.score = score

    def __str__(self):
        return (f"Prompt(\n"
                f"  prompt: {self.prompt},\n"
                f"  feedbacks_idx_used: {self.feedbacks_idx_used},\n"
                f"  examplers_idx_used: {self.examplers_idx_used},\n"
                f"  parent_score: {self.parent_score},\n"
                f"  score: {self.score}"
                f")")
                
    __repr__ = __str__ 