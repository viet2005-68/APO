import json
import numpy as np
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import utils
from FlagEmbedding import BGEM3FlagModel
from models import Prompt

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

class FeedbackMemory():
    def __init__(self, embedding_model, similarity_threshold=0.8):
        self.feedbacks = []
        self.error_string = []
        self.scores = []
        self.embeddings = []
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    def __str__(self):
        max_score = max(self.scores) if self.scores else None
        min_score = min(self.scores) if self.scores else None
        return (f"FeedbackMemory(\n"
                f"  feedbacks: {self.feedbacks} items,\n"
                f"  scores: {self.scores} items,\n"
                f"  max score: {max_score}\n"
                f"  min score: {min_score}"
                f")")

    def _encode(self, text):
        return self.embedding_model.encode(text)['dense_vecs']

    def _is_duplicate(self, new_emb):
        """Check similarity against stored feedback."""
        if not self.embeddings:
            return False
        emb_matrix = np.stack(self.embeddings)
        sims = emb_matrix @ new_emb.T
        if np.max(sims) >= self.similarity_threshold:
            print("SIMILAR FEEDBACK ALREADY OCCUR WITH SIMILARITY ", np.max(sims)) 
        return np.max(sims) >= self.similarity_threshold

    def add_feedback(self, feedback, error_string, score=0.5):
        new_emb = self._encode(feedback)

        if self._is_duplicate(new_emb):
            return False

        self.feedbacks.append(feedback)
        self.error_string.append(error_string)
        self.scores.append(score)
        self.embeddings.append(new_emb)
        return True

    def retrieve_feedback(self, num=5, T=1):
        priority = np.array(self.scores)
        exp_scores = np.exp((priority - np.max(priority)) / T)
        probs = exp_scores / exp_scores.sum()
        retrieved_idx = np.random.choice(len(self.feedbacks), size=min(num, len(self.feedbacks)), replace=False, p=probs)
        return [(i, self.feedbacks[i], self.error_string[i]) for i in retrieved_idx]
    
    def update_feedback(self, feedback_idx, score_gain, beta=1, theta=0.1):
        current_score = self.scores[feedback_idx]
        new_score = current_score + score_gain * beta
        self.scores[feedback_idx] = new_score
        if new_score < theta:
            self.scores[feedback_idx] = 0

class ExemplarMemory():
    def __init__(self, embedding_model, similarity_threshold=0.8, replace_prob=0.5):
        self.exemplars = []
        self.scores = []
        self.embeddings = []
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.replace_prob = replace_prob

    def __str__(self):
        max_score = max(self.scores) if self.scores else None
        min_score = min(self.scores) if self.scores else None
        return (f"ExemplarMemory(\n"
                f"  exemplars: {self.exemplars} items,\n"
                f"  scores: {self.scores} items,\n"
                f"  max score: {max_score}\n"
                f"  min score: {min_score}"
                f")")


    def _encode(self, text):
        return self.embedding_model.encode(text)['dense_vecs']

    def _find_duplicate(self, new_emb):
        if not self.embeddings:
            return None

        emb_matrix = np.stack(self.embeddings)
        sims = emb_matrix @ new_emb.T

        idx = np.argmax(sims)
        if sims[idx] >= self.similarity_threshold:
            print("SIMILAR EXAMPLER ALREADY OCCUR WITH SIMILARITY ", sims[idx]) 
            return idx
        return None
    
    def add_exemplar(self, text, priority_score=0.5):
        new_emb = self._encode(text)
        dup_idx = self._find_duplicate(new_emb)

        if dup_idx is None:
            self.exemplars.append(text)
            self.scores.append(priority_score)
            self.embeddings.append(new_emb)
            return True

        if random.random() < self.replace_prob:
            self.exemplars[dup_idx] = text
            self.embeddings[dup_idx] = new_emb
            return True
        else:
            return False

    def retrieve_exemplar(self, question="", num=5, temperature=1, inference=False):
        # q_emb = self._encode(question)
        # emb_matrix = np.stack(self.embeddings)
        # sims = emb_matrix @ q_emb.T
        # weighted_scores = np.array(self.scores) * sims
        weighted_scores = np.array(self.scores)
        if inference:
            top_indices = np.argsort(weighted_scores)[-num:][::-1]
        else:
            shifted = weighted_scores / temperature
            shifted -= np.max(shifted)
            exp_scores = np.exp(shifted)
            probs = exp_scores / exp_scores.sum()
            top_indices = np.random.choice(len(self.exemplars), size=min(num, len(self.exemplars)), replace=False, p=probs)

        return [(i, self.exemplars[i]) for i in top_indices]

    def update_exemplar(self, exemplar_idx, prompt_score, beta=1, theta=0.1):
        current_score = self.scores[exemplar_idx]
        new_score = current_score + prompt_score * beta
        self.scores[exemplar_idx] = new_score
        if new_score < theta:
            self.scores[exemplar_idx] = 0

class MyOptimizer(PromptOptimizer):
    """Update ProTeGi: Prompt Optimization with Textual Gradients"""

    def __init__(self, args, evaluator_fn, scorer, max_threads=1, bf_eval=None):
        super().__init__(args, evaluator_fn, scorer, max_threads, bf_eval)
        embedding_model = BGEM3FlagModel('BAAI/bge-m3')
        self.feedback_memory = FeedbackMemory(embedding_model=embedding_model)
        self.exemplar_memory = ExemplarMemory(embedding_model=embedding_model)

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
    
    def _get_examplers(self, prompt, error_string, num_examplers=5):
        examplers_prompt = f"""
        I'm trying to write and complete a zero-shot classifier prompt from difficult or erroneous
        examples.
        My current prompt is:
        {prompt}
        But this prompt gets the following examples wrong:
        {error_string}
        To improve my understanding and performance, I would like to identify {num_examplers} typical
        examples from the above cases where the current prompt fails.
        These examples should be diverse to cover a range of different issues.
        For each example, provide the following input and label, not the prediction and wrap each example with <ANSWER>
        and </ANSWER>:
        <ANSWER>
        [one full exampler here, no lists, no numbering]
        </ANSWER>
        Output exactly {num_examplers} such blocks.
        Do not output anything outside the <ANSWER> tags.
        Begin now.
        """
        transformation_prompt = "\n".join(
            [line.lstrip() for line in examplers_prompt.split("\n")]
        )
        res = utils.chatgpt(transformation_prompt, n=1)
        examplers = []
        for r in res:
            examplers += self.parse_tagged_text(r, "<ANSWER>", "</ANSWER>")
        print("LLM examplers: ", examplers)
        print("LLM examplers size: ", len(examplers))
        return examplers

    def _get_feedbacks(self, prompt, error_string, num_feedbacks=5):
        feedback_prompt = f"""
        I'm trying to write a zero-shot classifier prompt.
    
        My current prompt is:
        "{prompt}"

        But this prompt gets the following examples wrong:
        {error_string}

        give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
        For each feedback, provide the following format and wrap each feedback with <ANSWER>
        and </ANSWER>:
        <ANSWER>
        [one full feedback here, no lists, no numbering]
        </ANSWER>
        Output exactly {num_feedbacks} such blocks.
        Do not output anything outside the <ANSWER> tags.
        Begin now.
        """
        gradient_prompt = "\n".join(
            [line.lstrip() for line in feedback_prompt.split("\n")]
        )
        res = utils.chatgpt(gradient_prompt, n=1)
        feedbacks = []
        for r in res:
            feedbacks += self.parse_tagged_text(r, "<ANSWER>", "</ANSWER>")
        print("LLM feedbacks: ", feedbacks)
        print("LLM feedbacks size: ", len(feedbacks))
        return feedbacks

    def get_feedbacks(self, task_section, task, gpt4, texts, labels, preds):
        """Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        for _ in tqdm(
            range(self.opt["n_gradients"]),
            total=self.opt["n_gradients"],
            desc="fetching feedbacks..",
        ):
            error_string = self._sample_error_str(
                texts, labels, preds, task, n=self.opt["errors_per_gradient"]
            )
            gradients = self._get_feedbacks(
                task_section, error_string, self.opt["gradients_per_error"]
            )
            prompt_feedbacks += [(t, error_string) for t in gradients]
        return prompt_feedbacks
    
    def get_examplers(self, task_section, task, gpt4, texts, labels, preds):
        """Get "gradients" for a prompt based on sampled error strings."""
        examplers_feedbacks = []
        for _ in tqdm(
            range(self.opt["n_gradients"]),
            total=self.opt["n_gradients"],
            desc="fetching examplers..",
        ):
            error_string = self._sample_error_str(
                texts, labels, preds, task, n=self.opt["errors_per_gradient"]
            )
            examplers = self._get_examplers(
                task_section, error_string, self.opt["gradients_per_error"]
            )
            examplers_feedbacks += examplers
        return examplers_feedbacks 
    
    def optimize_prompt(self, prompt, error_samples, feedback, n=1):
        # Here are some examples of issues and their labels:
        # {error_samples}
        transformation_prompt = f"""
        I'm trying to write and complete a zero-shot classifier prompt from difficult or erroneous
        examples.
        My current prompt is:
        {prompt}
        Here are some suggestions for improving the prompt:
        {feedback}
        Based on the above information, I refine the prompt to make the model predict correctly.
        The improved prompt must be wrapped individually like this:
        <ANSWER>
        [one full improved prompt here, no lists, no numbering]
        </ANSWER>
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
            sections = utils.parse_sectioned_prompt(prompt.prompt)
            print("PROMPT: ", prompt)
            task_section = sections["task"].strip()
            exemplar_section = sections["exemplar"].strip()
            # evaluate prompt on minibatch
            _, texts, labels, preds = task.evaluate(gpt4, prompt.prompt, minibatch)

            # get gradients
            new_task_sections = []
            new_exemplar_sections = []
            if self.opt["n_gradients"] > 0:
                examplers = self.get_examplers(
                    task_section, task, gpt4, texts, labels, preds
                )
                feedbacks = self.get_feedbacks(
                    task_section, task, gpt4, texts, labels, preds
                )
                for feedback, error_string in feedbacks:
                    self.feedback_memory.add_feedback(feedback, error_string)
                for exemplar in examplers:
                    self.exemplar_memory.add_exemplar(exemplar)
                # retrieved_feedbacks = self.feedback_memory.retrieve_feedback(len(feedbacks))
                # for idx, feedback, error_string in tqdm(
                #     retrieved_feedbacks, desc="applying gradients"
                # ):
                for i in tqdm(
                    range(10), desc="applying gradients"
                ):
                    retrieved_feedbacks = self.feedback_memory.retrieve_feedback()
                    retrieved_exemplars = self.exemplar_memory.retrieve_exemplar()
                    feedback_idx = [i[0] for i in retrieved_feedbacks]
                    feedback_str = [i[1] for i in retrieved_feedbacks]
                    exemplar_idx = [i[0] for i in retrieved_exemplars]
                    exemplar_str = [i[1] for i in retrieved_exemplars]
                    feedback = "\n\n".join(feedback_str)
                    exemplar = "\n\n".join(exemplar_str)
                    tmp = self.optimize_prompt(
                        task_section,
                        "",
                        feedback
                    )
                    for i in tmp:
                        new_task_sections.append(Prompt(i, set(feedback_idx), set(exemplar_idx), prompt.score, 0))
                        new_exemplar_sections.append(exemplar)
                print("new promt: ", new_task_sections)
                print("len new prompt: ", len(new_task_sections))
            # generate synonyms
            mc_sampled_task_sections = []
            if self.opt["mc_samples_per_step"] > 0:
                for sect in tqdm(new_task_sections + [Prompt(task_section, set(), set(), 0, 0)], desc="mc samples"):
                    mc_sects = self.generate_synonyms(
                        sect.prompt, n=self.opt["mc_samples_per_step"]
                    )
                    for i in mc_sects:
                        mc_sampled_task_sections.append(Prompt(i, sect.feedbacks_idx_used, sect.examplers_idx_used, sect.score, 0))

            # Genetic algorithm
            ea_sampled_task_sections = []
            if self.opt["ea_samples_per_step"] > 0:
                for i in tqdm(range(self.opt["ea_samples_per_step"]), desc="evolution algorithm"):
                    if len(new_task_sections + 1) < 2:
                        break
                    parents = random.sample(new_task_sections + [Prompt(task_section, set(), set(), 0, 0)], 2)
                    prompt1 = parents[0]
                    prompt2 = parents[1]
                    ea_prompt = self.genetic_algorithm_expansion(prompt1, prompt2)
                    ea_sampled_task_sections += Prompt(ea_prompt, set(), set(), 0, 0)

            # combine
            new_sections = new_task_sections + mc_sampled_task_sections + ea_sampled_task_sections
            # new_sections = new_task_sections
            tmp_new_prompts = [
                Prompt(prompt.prompt.replace(task_section, tmp.prompt).replace(exemplar_section, tmp_exemplar), tmp.feedbacks_idx_used, tmp.examplers_idx_used, tmp.parent_score, tmp.score) for tmp, tmp_exemplar in zip(new_sections, new_exemplar_sections)
            ]
            print(tmp_new_prompts[0])
            # # filter a little
            # if len(new_sections) > self.opt["max_expansion_factor"]:
            #     if self.opt["reject_on_errors"]:
            #         error_exs = []
            #         for i, (t, l, p) in enumerate(zip(texts, labels, preds)):
            #             if l != p:
            #                 error_exs.append({"text": t, "label": l})
            #         error_exs = random.sample(error_exs, min(len(error_exs), 16))

            #         # speed up a little
            #         tmp_new_prompts = random.sample(
            #             tmp_new_prompts,
            #             min(len(tmp_new_prompts), self.opt["max_expansion_factor"] * 2),
            #         )

            #         error_scores = self.bf_eval(
            #             tmp_new_prompts,
            #             error_exs,
            #             task,
            #             gpt4,
            #             self.scorer,
            #             max_threads=self.max_threads,
            #         )
            #         tmp_new_prompts = [
            #             tmp_new_prompts[i]
            #             for i in np.argsort(error_scores)[
            #                 -self.opt["max_expansion_factor"] :
            #             ]
            #         ]
            #     else:
            #         sample_k = min(
            #             len(tmp_new_prompts), self.opt["max_expansion_factor"]
            #         )
            #         if sample_k > 0:
            #             tmp_new_prompts = random.sample(tmp_new_prompts, k=sample_k)
            #         else:
            #             tmp_new_prompts = []

            new_prompts += tmp_new_prompts

        new_prompts += prompts  # add originals
        # new_prompts = list(set(new_prompts))  # dedup

        return new_prompts

    def score_candidates(self, prompts, task, gpt4, train_exs):
        """Score a list of prompts."""
        evals = self.evaluator_fn(
            [prompt.prompt for prompt in prompts],
            train_exs,
            task,
            gpt4,
            scorer=self.scorer,
            rounds=self.opt["eval_rounds"],
            num_prompts_per_round=self.opt["eval_prompts_per_round"],
            samples_per_eval=self.opt["samples_per_eval"],
            max_threads=self.max_threads,
        )
        for prompt, eval in zip(prompts, evals):
            prompt.score = eval
            for feedback_idx in prompt.feedbacks_idx_used:
                self.feedback_memory.update_feedback(feedback_idx, prompt.score - prompt.parent_score)
            for exemplar_idx in prompt.examplers_idx_used:
                self.exemplar_memory.update_exemplar(exemplar_idx, prompt.score - prompt.parent_score)
        print("Feedback Memory: ", self.feedback_memory)
        print("Exemplar Memory: ", self.exemplar_memory)
        return evals