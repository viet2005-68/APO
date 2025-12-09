import json
import numpy as np
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import utils
from FlagEmbedding import BGEM3FlagModel
from models import Prompt

random.seed(42)

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

    def retrieve_latest_exemplar(self, num=5):
        retrieved_idx = list(range(len(self.exemplars) - num, len(self.exemplars)))
        return [(i, self.exemplars[i]) for i in retrieved_idx]

    def update_exemplar(self, exemplar_idx, prompt_score, beta=0.3, theta=0.1):
        current_score = self.scores[exemplar_idx]
        new_score = current_score + prompt_score * beta
        self.scores[exemplar_idx] = new_score
        if new_score < theta:
            self.scores[exemplar_idx] = 0

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


class ProTeGi(PromptOptimizer):
    """ProTeGi: Prompt Optimization with Textual Gradients"""
    def __init__(self, args, evaluator_fn, scorer, max_threads=1, bf_eval=None):
        super().__init__(args, evaluator_fn, scorer, max_threads, bf_eval)
        embedding_model = BGEM3FlagModel('BAAI/bge-m3')
        self.exemplar_memory = ExemplarMemory(embedding_model=embedding_model)

    def _sample_error_str(self, texts, labels, preds, confs, task, n=4):
        """Sample n error strings from the given texts, labels, and preds"""
        error_idxs = []
        for i, (l, p) in enumerate(zip(labels, preds)):
            if l != p:
                error_idxs.append(i)

        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))

        sample_texts = [texts[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_confs = [confs[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]
        error_string = ""
        num_errors = 0
        error_idx = 0
        for i, (t, l, p, c) in enumerate(zip(sample_texts, sample_labels, sample_preds, sample_confs)):
            error_string += f"## Example {error_idx+1}\n"
            error_string += f'Text: "{t.strip()}"\nLabel: {task.stringify_prediction(l)}\nPrediction: {task.stringify_prediction(p)}\nConfidence: {c}\n\n'
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

        But this prompt gets the following examples wrong (each includes: Input, Prediction, GroundTruth, Confidence):
        {error_string}

        Your task:
        - Give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
        - The reasons MUST explicitly use the confidence information from each example.
        - Treat HIGH-CONFIDENCE errors (confidence ≥ 0.85) as indicators of major structural flaws in the prompt.
        - Treat MEDIUM-CONFIDENCE errors (0.60–0.85) as indicators of ambiguous or incomplete instructions.
        - Treat LOW-CONFIDENCE errors (< 0.60) as indicators of under-specified instructions or borderline cases requiring small adjustments.
        - If an example is correct but low-confidence, treat it as a sign of underspecification and propose a clarification.
        - Each reason must clearly reference how confidence affects the interpretation of the error.

        Formatting requirements:
        - Output exactly {num_feedbacks} reasons.
        - Each reason MUST be wrapped individually like this:

        <ANSWER>
        [one full reason here — must be self-contained, must reference confidence, must propose a fix]
        </ANSWER>

        Do not output anything outside the <ANSWER> blocks.
        Begin now.
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
        I am optimizing a zero-shot classification prompt.

        CURRENT PROMPT:
        {prompt}

        FAILURE CASES:
        Each failure case includes:
        - the input text,
        - the true label,
        - the model’s predicted label,
        - the model’s confidence score for that prediction,

        {error_str}

        SUMMARY OF PROBLEMS IN THE CURRENT PROMPT:
        {feedback_str}

        UPDATE INTENSITY RULE (LEARNING RATE):
        Use the confidence score only to determine how much you are allowed to modify the prompt. 
        IMPORTANT: The updated prompts MUST NOT mention confidence, failure cases, or any optimization details.

        - HIGH-CONFIDENCE ERROR (confidence ≥ 0.85):
        → Apply a large structural update to the prompt. You may revise or replace multiple sentences, add new rules, constraints, or reasoning steps.

        - MEDIUM-CONFIDENCE ERROR (0.60 ≤ confidence < 0.85):
        → Apply a moderate update. Adjust several phrases or sections; refine reasoning or add clarifying constraints.

        - LOW-CONFIDENCE ERROR (confidence < 0.60):
        → Apply a small update. Only lightly edit or clarify wording; avoid major structural changes.

        YOUR TASK:
        - Generate {steps_per_gradient} substantively different improved versions of the prompt.
        - Each improved prompt must:
        • Explicitly fix the issues described in the problem summary.
        • Apply the correct update intensity based on the confidence patterns.
        • Introduce a new structural idea, rule, constraint, or reasoning step.
        • Be complete and standalone.
        • Produce deterministic and consistent classification behavior.
        • NOT reference confidence, training examples, error cases, “update intensity,” or any details from this optimization process.

        OUTPUT FORMAT:
        Each improved prompt must be wrapped individually as:

        <ANSWER>
        [one full improved prompt here — context-free, self-contained, and without any mention of confidence or errors]
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

    def get_gradients(self, prompt, task_section, task, gpt4, texts, labels, preds, confs):
        """Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        for _ in tqdm(
            range(self.opt["n_gradients"]),
            total=self.opt["n_gradients"],
            desc="gradients..",
        ):
            error_string = self._sample_error_str(
                texts, labels, preds, confs, task, n=self.opt["errors_per_gradient"]
            )
            gradients = self._get_gradients(
                task_section, error_string, self.opt["gradients_per_error"], n=1
            )
            prompt_feedbacks += [(t, error_string) for t in gradients]
        return prompt_feedbacks

    def get_examplers(self, task_section, task, gpt4, texts, labels, preds, confs):
        """Get "gradients" for a prompt based on sampled error strings."""
        examplers_feedbacks = []
        for _ in tqdm(
            range(self.opt["n_gradients"]),
            total=self.opt["n_gradients"],
            desc="fetching examplers..",
        ):
            error_string = self._sample_error_str(
                texts, labels, preds, confs, task, n=self.opt["errors_per_gradient"]
            )
            examplers = self._get_examplers(
                task_section, error_string
            )
            examplers_feedbacks += examplers
        return examplers_feedbacks 

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

    def expand_candidates(self, prompts, task, gpt4, train_exs):
        """Expand a list of prompts by generating gradient-based successors and
        synonyms for each section.
        """
        minibatch = random.sample(train_exs, k=min(self.opt["minibatch_size"], len(train_exs)))

        new_prompts = []
        for prompt in tqdm(prompts, desc=f"expanding {len(prompts)} prompts"):
            sections = utils.parse_sectioned_prompt(prompt.prompt)
            task_section = sections["task"].strip()
            exemplar_section = sections["exemplar"].strip()
            # evaluate prompt on minibatch
            _, texts, labels, preds, confs = task.evaluate_with_conf(gpt4, prompt.prompt, minibatch)
            print(confs)
            # get gradients
            new_task_sections = []
            new_exemplar_sections = []
            if self.opt["n_gradients"] > 0:
                examplers = self.get_examplers(
                    task_section, task, gpt4, texts, labels, preds, confs
                )
                for exemplar in examplers:
                    self.exemplar_memory.add_exemplar(exemplar)
                gradients = self.get_gradients(
                    prompt.prompt, task_section, task, gpt4, texts, labels, preds, confs
                )
                print("gradients: ", gradients)
                print("len gradients: ", len(gradients))
                new_task_sections = []
                for feedback, error_string in tqdm(
                    gradients, desc="applying gradients"
                ):
                    retrieved_exemplars = self.exemplar_memory.retrieve_exemplar()
                    retrieved_exemplars += self.exemplar_memory.retrieve_latest_exemplar()
                    retrieved_exemplars_set = set(retrieved_exemplars)
                    exemplar_idx = [i[0] for i in retrieved_exemplars_set]
                    exemplar_str = [i[1] for i in retrieved_exemplars_set]
                    exemplar = "\n\n".join(exemplar_str)
                    tmp = self.apply_gradient(
                        task_section,
                        error_string,
                        feedback,
                        self.opt["steps_per_gradient"],
                    )
                    for i in tmp:
                        new_task_sections.append(Prompt(i, set(), set(exemplar_idx), prompt.score, 0))
                        new_exemplar_sections.append(exemplar)
                print("new promt: ", new_task_sections)
                print("len new prompt: ", len(new_task_sections))
            # generate synonyms
            mc_sampled_task_sections = []
            if self.opt["mc_samples_per_step"] > 0:
                for ind, sect in tqdm(enumerate(new_task_sections), desc="mc samples"):
                    mc_sects = self.generate_synonyms(
                        sect.prompt, n=self.opt["mc_samples_per_step"]
                    )
                    for i in mc_sects:
                        mc_sampled_task_sections.append(Prompt(i, set(), sect.examplers_idx_used, sect.parent_score, 0))
                        new_exemplar_sections.append(new_exemplar_sections[ind])

            # combine
            new_sections = new_task_sections + mc_sampled_task_sections
            # new_sections = list(set(new_sections))  # dedup
            # tmp_new_prompts = [
            #     prompt.replace(task_section, tmp) for tmp in new_sections
            # ]
            tmp_new_prompts = [
                Prompt(prompt.prompt.replace(task_section, tmp.prompt).replace(exemplar_section, tmp_exemplar), set(), tmp.examplers_idx_used, tmp.parent_score, tmp.score) for tmp, tmp_exemplar in zip(new_sections, new_exemplar_sections)
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
                        [prompt.prompt for prompt in tmp_new_prompts],
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
        # new_prompts = list(set(new_prompts))  # dedup

        return new_prompts

    def score_candidates(self, prompts, task, gpt4, train_exs):
        """Score a list of prompts."""
        # if len(prompts) == 1:
        #     return [1.0]

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
            for exemplar_idx in prompt.examplers_idx_used:
                self.exemplar_memory.update_exemplar(exemplar_idx, prompt.score - prompt.parent_score)
        print("Exemplar Memory: ", self.exemplar_memory)
        return evals