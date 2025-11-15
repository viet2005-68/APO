import json
import numpy as np
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import utils

class PromptOptimizer(ABC):
    def __init__(self, args, evaluator_fn, scorer, max_threads=1, bf_eval=None):
        self.opt = args
        self.evaluator_fn = evaluator_fn
        self.scorer = scorer
        self.max_threads = max_threads
        self.bf_eval = bf_eval

    @abstractmethod
    def expand_candidates(self, prompts, task, gpt4, train_exs):
        pass


class MyOptimizer(PromptOptimizer):
    """ProTeGi: Prompt Optimization with Textual Gradients"""

    def _sample_error_str(self, texts, labels, preds, task, n=4):
        """Sample n error strings from the given texts, labels, and preds"""
        error_idxs = []
        for i, (l, p) in enumerate(zip(labels, preds)):
            if l != p:
                error_idxs.append(i)

        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))

        sample_texts = [texts[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]
        error_string = ""
        num_errors = 0
        error_idx = 0
        for i, (t, l, p) in enumerate(zip(sample_texts, sample_labels, sample_preds)):
            error_string += f"## Example {error_idx+1}\n"
            error_string += f'Text: "{t.strip()}"\nLabel: {task.stringify_prediction(l)}\nPrediction: {task.stringify_prediction(p)}\n\n'
            error_idx += 1
        return error_string.strip()

    def parse_tagged_text(self, text, start_tag, end_tag):
        """Parse text that is tagged with start and end tags."""
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            text = text[end_index + len(end_tag) :]
        return texts

    def _get_gradients(self, prompt, error_string, num_feedbacks=5, n=1):
        """Get "gradients" for a prompt based on the error string."""
        gradient_prompt = f"""
        I'm trying to write a zero-shot classifier prompt.
    
        My current prompt is:
        "{prompt}"

        But this prompt gets the following examples wrong:
        {error_string}

        give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
        MUST wrap each reason with <ANSWER> and </ANSWER>
        The {num_feedbacks} reasons are:
        """
        gradient_prompt = "\n".join(
            [line.lstrip() for line in gradient_prompt.split("\n")]
        )
        res = utils.chatgpt(gradient_prompt, n=n)
        feedbacks = []
        new_prompts = []
        for r in res:
            feedbacks += self.parse_tagged_text(r, "<ANSWER>", "</ANSWER>")
        print("Gradient String: ", res[0])
        print("Gradient llm feedback response: ", feedbacks)
        print("Gradient llm feedback len: ", len(feedbacks))
        return feedbacks

    def apply_gradient(self, prompt, error_str, feedback_str, steps_per_gradient, n=1):
        """Incorporate feedback gradient into a prompt."""
        transformation_prompt = f"""
        I'm trying to write a zero-shot classifier.
        
        My current prompt is:
        "{prompt}"

        But it gets the following examples wrong:
        {error_str}

        Based on these examples the problem with this prompt is that {feedback_str}

        My task:
        - Generate {steps_per_gradient} *substantively different* improved versions of the prompt.
        - Each improved prompt must explicitly FIX the issues described in the feedback.
        - Each improved prompt must introduce a *new structural idea*, constraint, or reasoning step.
        - Each improved prompt must be complete and standalone.
        - Each improved prompt must be wrapped individually like this:

        <ANSWER>
        [one full improved prompt here, no lists, no numbering]
        </ANSWER>

        Output exactly {steps_per_gradient} such blocks.
        Do not output anything outside the <ANSWER> tags.
        Begin now.
        """
        transformation_prompt = "\n".join(
            [line.lstrip() for line in transformation_prompt.split("\n")]
        )
        res = utils.chatgpt(transformation_prompt, n=n)
        new_prompts = []
        for r in res:
            new_prompts += self.parse_tagged_text(r, "<ANSWER>", "</ANSWER>")
        print("Gradient llm prompt response: ", res)
        return new_prompts

    def generate_synonyms(self, prompt_section, n=3):
        """Generate synonyms for a prompt section."""
        rewriter_prompt = f"Generate a variation of the following instruction while keeping the semantic meaning.\n\nInput: {prompt_section}\n\nOutput:"
        new_instructions = utils.chatgpt(rewriter_prompt, n=n)
        new_instructions = [x for x in new_instructions if x]
        return new_instructions

    def get_gradients(self, prompt, task_section, task, gpt4, texts, labels, preds):
        """Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        for _ in tqdm(
            range(self.opt["n_gradients"]),
            total=self.opt["n_gradients"],
            desc="gradients..",
        ):
            error_string = self._sample_error_str(
                texts, labels, preds, task, n=self.opt["errors_per_gradient"]
            )
            gradients = self._get_gradients(
                task_section, error_string, self.opt["gradients_per_error"], n=1
            )
            prompt_feedbacks += [(t, error_string) for t in gradients]
        return prompt_feedbacks

    def genetic_algorithm_expansion(self, prompt1, prompt2):
        instruction = f"""
                    You are performing genetic algorithm evolution on classifier prompts.

                    PARENT A:
                    {prompt1}

                    PARENT B:
                    {prompt2}

                    Your task:

                    1. **CROSSOVER**
                    - Combine only the *useful mechanisms* from both parents.
                    - DO NOT copy entire sentences or templates.
                    - Merge conceptual rules, not wording.

                    2. **MUTATE**
                    Apply at least **two high-impact structural mutations**, such as:
                    - Add a multi-step reasoning procedure
                    - Add an explicit scoring method
                    - Add constraints or banned behaviors
                    - Add a verification or calibration step
                    - Change label definitions or output format
                    - Add an uncertainty threshold
                    - Introduce consistency checks
                    - Add chain-of-thought *format rules* without revealing chain-of-thought

                    3. **ANTI-COLLAPSE REQUIREMENTS**
                    - The child MUST be less than 40% similar in wording to either parent.
                    - The child MUST introduce at least one conceptual mechanism **not present** in either parent.
                    - The child MUST NOT follow the common template used in the parents 
                    - The child MUST NOT reuse wording like "evaluate the Statement", "consider the context", etc.

                    4. **OUTPUT FORMAT**
                    Return EXACTLY one offspring prompt, wrapped:

                    <ANSWER>
                    [offspring prompt]
                    </ANSWER>

                    No other text.
                    """

        instruction = "\n".join([line.lstrip() for line in instruction.split("\n")])
        res = utils.chatgpt(instruction, n=1)
        new_prompts = []
        for r in res:
            new_prompts += self.parse_tagged_text(r, "<ANSWER>", "</ANSWER>")
        print("GA llm raw prompt: ", res)
        print("GA llm prompt: ", new_prompts)
        print("GA llm prompt len: ", len(new_prompts))
        return new_prompts


    def expand_candidates(self, prompts, task, gpt4, train_exs):
        """Expand a list of prompts by generating gradient-based successors and
        synonyms for each section.
        """
        minibatch = random.sample(train_exs, k=self.opt["minibatch_size"])

        new_prompts = []
        for prompt in tqdm(prompts, desc=f"expanding {len(prompts)} prompts"):
            sections = utils.parse_sectioned_prompt(prompt)
            task_section = sections["task"].strip()

            # evaluate prompt on minibatch
            _, texts, labels, preds = task.evaluate(gpt4, prompt, minibatch)

            # get gradients
            new_task_sections = []
            if self.opt["n_gradients"] > 0:
                gradients = self.get_gradients(
                    prompt, task_section, task, gpt4, texts, labels, preds
                )
                print("gradients: ", gradients)
                print("len gradients: ", len(gradients))
                new_task_sections = []
                for feedback, error_string in tqdm(
                    gradients, desc="applying gradients"
                ):
                    tmp = self.apply_gradient(
                        task_section,
                        error_string,
                        feedback,
                        self.opt["steps_per_gradient"],
                    )
                    new_task_sections += tmp
                print("new promt: ", new_task_sections)
                print("len new prompt: ", len(new_task_sections))
            # generate synonyms
            mc_sampled_task_sections = []
            if self.opt["mc_samples_per_step"] > 0:
                for sect in tqdm(new_task_sections + [task_section], desc="mc samples"):
                    mc_sects = self.generate_synonyms(
                        sect, n=self.opt["mc_samples_per_step"]
                    )
                    mc_sampled_task_sections += mc_sects

            # Genetic algorithm
            ea_sampled_task_sections = []
            if self.opt["ea_samples_per_step"] > 0:
                for i in tqdm(range(self.opt["ea_samples_per_step"]), desc="evolution algorithm"):
                    if len(new_task_sections + [task_section]) < 2:
                        break
                    parents = random.sample(new_task_sections + [task_section], 2)
                    prompt1 = parents[0]
                    prompt2 = parents[1]
                    ea_prompt = self.genetic_algorithm_expansion(prompt1, prompt2)
                    ea_sampled_task_sections += ea_prompt

            # combine
            new_sections = new_task_sections + mc_sampled_task_sections
            new_sections = list(set(new_sections))  # dedup
            tmp_new_prompts = [
                prompt.replace(task_section, tmp) for tmp in new_sections
            ]

            # filter a little
            if len(new_sections) > self.opt["max_expansion_factor"]:
                if self.opt["reject_on_errors"]:
                    error_exs = []
                    for i, (t, l, p) in enumerate(zip(texts, labels, preds)):
                        if l != p:
                            error_exs.append({"text": t, "label": l})
                    error_exs = random.sample(error_exs, min(len(error_exs), 16))

                    # speed up a little
                    tmp_new_prompts = random.sample(
                        tmp_new_prompts,
                        min(len(tmp_new_prompts), self.opt["max_expansion_factor"] * 2),
                    )

                    error_scores = self.bf_eval(
                        tmp_new_prompts,
                        error_exs,
                        task,
                        gpt4,
                        self.scorer,
                        max_threads=self.max_threads,
                    )
                    tmp_new_prompts = [
                        tmp_new_prompts[i]
                        for i in np.argsort(error_scores)[
                            -self.opt["max_expansion_factor"] :
                        ]
                    ]
                else:
                    sample_k = min(
                        len(tmp_new_prompts), self.opt["max_expansion_factor"]
                    )
                    if sample_k > 0:
                        tmp_new_prompts = random.sample(tmp_new_prompts, k=sample_k)
                    else:
                        tmp_new_prompts = []

            new_prompts += tmp_new_prompts

        new_prompts += prompts  # add originals
        new_prompts = list(set(new_prompts))  # dedup

        return new_prompts

    def score_candidates(self, prompts, task, gpt4, train_exs):
        """Score a list of prompts."""
        if len(prompts) == 1:
            return [1.0]

        evals = self.evaluator_fn(
            prompts,
            train_exs,
            task,
            gpt4,
            scorer=self.scorer,
            rounds=self.opt["eval_rounds"],
            num_prompts_per_round=self.opt["eval_prompts_per_round"],
            samples_per_eval=self.opt["samples_per_eval"],
            max_threads=self.max_threads,
        )
        return evals