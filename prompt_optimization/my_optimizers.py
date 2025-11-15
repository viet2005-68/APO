import json
import numpy as np
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import math 
import utils

class FeedbackMemory:
    def __init__(self, theta=0.1, beta=0.5, temperature=1.0):
        self.memory = []
        self.theta = theta
        self.beta = beta
        self.temperature = temperature
    
    def store_feedbacks(self, feedbacks, scores):
        for f, s in zip(feedbacks, scores):
            self.memory.append({'feedback': f, 'score': s})
    
    def retrieve_feedbacks(self, k):
        if not self.memory:
            return []
        # softmax selection
        exp_scores = [math.exp(f['score']/self.temperature) for f in self.memory]
        total = sum(exp_scores)
        probs = [s/total for s in exp_scores]
        selected = np.random.choice(len(self.memory), size=min(k, len(self.memory)), p=probs, replace=False)
        return [self.memory[i]['feedback'] for i in selected]
    
    def update_feedbacks(self, selected_feedbacks, improved_flags):
        # increase score if improved, decrease otherwise
        for f, flag in zip(selected_feedbacks, improved_flags):
            for mem in self.memory:
                if mem['feedback'] == f:
                    mem['score'] = (1 - self.beta)*mem['score'] + self.beta*(1.0 if flag else 0.0)
        # forget low-score feedbacks
        self.memory = [f for f in self.memory if f['score'] >= self.theta]


class ExemplarFactory:
    def __init__(self, p_replace=0.5, tau_e=1.0):
        self.memory = []
        self.p_replace = p_replace
        self.tau_e = tau_e
    
    def store_exemplars(self, exemplars, evaluate_fn=lambda e: True):
        for e in exemplars:
            if evaluate_fn(e):
                # check for duplicates
                dup_idx = next((i for i, ex in enumerate(self.memory) if ex['exemplar']==e), -1)
                if dup_idx >= 0:
                    if random.random() < self.p_replace:
                        self.memory[dup_idx] = {'exemplar': e, 'score': 1.0}
                else:
                    self.memory.append({'exemplar': e, 'score': 1.0})
    
    def retrieve_exemplars(self, query, n=5, semantic_similarity_fn=lambda e,q: 1.0):
        # softmax selection based on semantic similarity * score
        if not self.memory:
            return []
        scores = [semantic_similarity_fn(ex['exemplar'], query) * ex['score'] for ex in self.memory]
        exp_scores = [math.exp(s/self.tau_e) for s in scores]
        total = sum(exp_scores)
        probs = [s/total for s in exp_scores]
        selected_idx = np.random.choice(len(self.memory), size=min(n, len(self.memory)), p=probs, replace=False)
        return [self.memory[i]['exemplar'] for i in selected_idx]

    def update_priority(self, exemplars, improved_flags):
        for e, flag in zip(exemplars, improved_flags):
            for mem in self.memory:
                if mem['exemplar'] == e:
                    mem['score'] = (0.5*mem['score'] + 0.5*(1.0 if flag else 0.0))
        # remove low-priority exemplars
        self.memory = [e for e in self.memory if e['score'] > 0.1]

class PromptOptimizer(ABC):
    def __init__(self, config, evaluator_fn, scorer, max_threads=1, bf_eval=None):
        self.opt = config
        self.evaluator_fn = evaluator_fn
        self.scorer = scorer
        self.max_threads = max_threads
        self.bf_eval = bf_eval
        self.fb_memory = FeedbackMemory(config)
        self.ex_factory = ExemplarFactory(
            p_replace=config.get("p_replace", 0.5),
            tau_e=config.get("tau_e", 1.0)
            )
    
    @abstractmethod
    def expand_candidates(self, prompts, task, gpt4, train_exs):
        pass


class ProTeGi(PromptOptimizer):
    """ProTeGi: Prompt Optimization with Textual Gradients"""

    def init_prompt_generation(self, original_prompt, examples):
        instruction = f"""
                        You are given an instruction on a certain task and some example inputs, outputs. Here is the current instruction: 
                        {original_prompt}
                        And here are some correct input-output pairs:
                        {examples}
                        Generate new instruction contains the following parts. Based on the input-output pairs provided,
                        give me the final complete instruction in English without any explanation:
                        # Task
                        Task: This is a <...> task.
                        Task detailed description: <Task detailed description>
                        You should follow the reasoning process: <add several reasoning steps if it's necessary>
                        Tips: <add several useful tips from a professional point of view to accomplish this task better>
                    """
        instruction = "\n".join(
            [line.lstrip() for line in instruction.split("\n")]
        )
        init_prompt = utils.chatgpt(instruction)[0]
        init_prompt += """
                        # Output format
                        Answer ONLY Yes or No as labels
                        # Prediction
                        Text: {{ text }}
                        Label:
                       """
        init_prompt = "\n".join(
            [line.lstrip() for line in init_prompt.split("\n")]
        )
        return init_prompt

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

    def _sample_correct_str(self, texts, labels, preds, task, n=4):
        """Sample n correct strings from the given texts, labels and preds"""
        correct_idxs = []
        for i, (l, p) in enumerate(zip(labels, preds)):
            if l == p:
                correct_idxs.append(i)

        sample_idxs = random.sample(correct_idxs, min(len(correct_idxs), n))

        sample_texts = [texts[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]
        correct_string = ""
        num_errors = 0
        correct_idx = 0
        for i, (t, l, p) in enumerate(zip(sample_texts, sample_labels, sample_preds)):
            correct_string += f"## Example {correct_idx+1}\n"
            correct_string += f'Text: "{t.strip()}"\nLabel: {task.stringify_prediction(l)}\nPrediction: {task.stringify_prediction(p)}\n\n'
            correct_idx += 1
        return correct_string.strip()

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
        Wrap each reason with <START> and <END>
        """
        gradient_prompt = "\n".join(
            [line.lstrip() for line in gradient_prompt.split("\n")]
        )
        res = utils.chatgpt(gradient_prompt, n=n)
        feedbacks = []
        new_prompts = []
        for r in res:
            feedbacks += self.parse_tagged_text(r, "<START>", "<END>")
        return feedbacks

    def _get_positive_feedback(self, prompt, correct_string, num_feedbacks=5, n=1):
        """Get "gradients" for a prompt based on the correct string."""
        gradient_prompt = f"""
        I'm trying to write a zero-shot classifier prompt.
    
        My current prompt is:
        "{prompt}"

        And this prompt gets the following examples correct:
        {correct_string}

        give {num_feedbacks} most valuable key points to improve the accuracy in solving this type of task.
        Wrap each reason with <START> and <END>
        """
        gradient_prompt = "\n".join(
            [line.lstrip() for line in gradient_prompt.split("\n")]
        )
        res = utils.chatgpt(gradient_prompt, n=n)
        feedbacks = []
        new_prompts = []
        for r in res:
            feedbacks += self.parse_tagged_text(r, "<START>", "<END>")
        return feedbacks


    def reflect_feedbacks(self, prompt_section, feedback_tuples):
        """Use reflection to make each feedback actionable and precise."""
        improved = []
        reflection_temp = self.opt.get("reflection_temperature", 0.0)
        for feedback, error_string in tqdm(
            feedback_tuples,
            desc="reflect gradients",
            leave=False,
        ):
            improved_feedback = feedback
            try:
                reflection_prompt = (
                    "You are an expert prompt engineer reviewing feedback for improving a binary"
                    " text-classification prompt.\n\n"
                    "Prompt section under improvement:\n"
                    f"{prompt_section}\n\n"
                    "Examples the prompt handled incorrectly:\n"
                    f"{error_string}\n\n"
                    "Original feedback:\n"
                    f"{feedback}\n\n"
                    "Rewrite the feedback so that it clearly identifies the problems in the prompt and"
                    " gives concrete, actionable guidance to fix them."
                    " If the feedback is already strong, restate it succinctly."
                    " Output only the improved feedback without commentary."
                )
                responses = utils.chatgpt(
                    reflection_prompt,
                    temperature=reflection_temp,
                    max_tokens=256,
                    n=1,
                )
                response = responses[0].strip() if responses else ""
                if response:
                    improved_feedback = response
            except Exception:
                improved_feedback = feedback
            improved.append((improved_feedback, error_string))
        return improved

    def refine_prompt_with_reflection(self, original_prompt, candidate_prompt):
        """Let the LLM reflect on a candidate and return refined prompt plus score."""
        reflection_temp = self.opt.get("reflection_temperature", 0.0)
        try:
            reflection_prompt = (
                "You are refining a prompt for a binary text-classification task.\n"
                "Maintain the original structure, headings, and intent, but fix weaknesses.\n"
                "Return a JSON object with the keys 'score' (0 to 1, indicating how much better the"
                " refined prompt is than the candidate) and 'prompt' (the improved prompt text).\n\n"
                "Original prompt:\n"
                f"{original_prompt}\n\n"
                "Candidate prompt generated after an improvement step:\n"
                f"{candidate_prompt}\n\n"
                "Identify any mistakes or missing instructions and produce an improved prompt that"
                " addresses them while preserving the formatting (including section headers such as"
                " '# Task', '# Output format', etc.). Output only the JSON object."
            )
            responses = utils.chatgpt(
                reflection_prompt,
                temperature=reflection_temp,
                max_tokens=512,
                n=1,
            )
            response = responses[0].strip() if responses else ""
            if response:
                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    data = None
                if isinstance(data, dict):
                    refined_prompt = data.get("prompt", "").strip()
                    score = data.get("score")
                    try:
                        score = float(score)
                    except (TypeError, ValueError):
                        score = None
                    if refined_prompt:
                        return refined_prompt, score
                else:
                    # fallback: treat raw text as prompt, no score available
                    return response, None
        except Exception:
            pass
        return candidate_prompt, None

    def apply_candidate_reflection(self, original_prompt, candidates, passes):
        """Iteratively refine candidate prompts via reflection passes."""
        if not candidates:
            return candidates

        refined = candidates[:]
        threshold = float(self.opt.get("reflection_candidate_threshold", 0.5))
        for pass_idx in range(passes):
            refined_with_progress = list(
                tqdm(
                    refined,
                    desc=f"reflect prompts pass {pass_idx + 1}/{passes}",
                    leave=False,
                )
            )
            next_candidates = []
            for candidate in refined_with_progress:
                refined_candidate, score = self.refine_prompt_with_reflection(
                    original_prompt, candidate
                )
                if not refined_candidate.strip():
                    refined_candidate = candidate
                if "# " in original_prompt and "# " not in refined_candidate:
                    refined_candidate = candidate
                if score is not None and score < threshold:
                    refined_candidate = candidate
                next_candidates.append(refined_candidate)
            refined = next_candidates

        # deduplicate while preserving order
        seen = set()
        unique_refined = []
        for candidate in refined:
            if candidate not in seen:
                seen.add(candidate)
                unique_refined.append(candidate)

        return unique_refined

    def generate_strategy(self, prompt, example, experience):
        transformation_prompt = f"""
        As an expert in prompt engineer, your task is to create a step-by-step strategy guide on how to use specific
        experience based on provided prompt.
        # Begin Demos
        <demo>
        <prompt>read the given paragraph and identify the most logical answer among the options.</prompt>
        <example>
        question: The following paragraphs each describe a set of five objects arranged in a fixed order. The
        statements are logically consistent within each paragraph. In a golf tournament, there were five golfers:
        Eve, Amy, Ada, Rob, and Joe. Amy finished second-to-last. Rob finished below Eve. Ada finished above
        Joe. Joe finished second.
        Options:
        (A) Eve finished last
        (B) Amy finished last
        (C) Ada finished last
        (D) Rob finished last
        (E) Joe finished last
        Answer: (B) Amy finished last
        Target: (D) Rob finished last
        </example>
        <experience> One primary reason mistakes occur in this task is due to misunderstanding or
        misinterpretation of the logical order and relationships presented in the paragraphs </experience>
        <strategy>
        Here is a strategy guide how to achieve "understanding or interpretation of the logical order and
        relationships":
        1. Carefully read the entire paragraph to understand the context and the objects or individuals involved.
        2. Identify the logical relationships or orderings described in the paragraph.
        3. Create a visual aid such as a list or a diagram. Place the objects or individuals from left to right based
        on the logical relationships. The leftmost object or individual would be the first in the order and the
        rightmost would be the last.
        4. As you read each relationship, adjust the positions of the objects or individuals in your visual aid
        accordingly.
        5. Once all relationships have been considered, your visual aid should represent the correct order of the
        objects or individuals.
        </strategy>
        </demo>
        # End Demos
        My current prompt is:
        <prompt>{prompt}</prompt>
        And here is the task data:
        <example>{example}</example>
        Through comprehensive analysis of the data, I've gained an experience that can improve the prompt:
        <experience>{experience}</experience>
        Based on my current prompt, please generate a strategy to address the above experience.
        The strategy is:
        """
        transformation_prompt = "\n".join(
            [line.lstrip() for line in transformation_prompt.split("\n")]
        )
        res = utils.chatgpt(transformation_prompt, n=1)
        return res

    def prompt_rewriter(self, prompt, example, steps_per_gradient, strategy, n=1):
        transformation_prompt = f"""
        My current instruction is:
        <prompt>{prompt}</prompt>
        And Here are some task data:
        <example>{example}</example>
        Through comprehensive analysis of the data, I get a experience and corresponding strategy:
        # Experience
        <experience>experience</experience>
        # Strategy
        <strategy>{strategy}</strategy>
        Based on my current prompt, refer to this experience and the strategy, write {steps_per_gradient} different improved prompt.
        Each prompt is wrapped with <START> and <END>.
        The {steps_per_gradient} improved prompts are:
        """
        transformation_prompt = "\n".join(
            [line.lstrip() for line in transformation_prompt.split("\n")]
        )
        res = utils.chatgpt(transformation_prompt, n=n)
        new_prompts = []
        for r in res:
            new_prompts += self.parse_tagged_text(r, "<START>", "<END>")
        return new_prompts
        
    def apply_gradient(self, prompt, error_str, feedback_str, steps_per_gradient, step_size, n=1):
        """Incorporate feedback gradient into a prompt."""
        # I am allowed to change up to {step_size} words in the current prompt.
        transformation_prompt = f"""
        I'm trying to write a zero-shot classifier.
        
        My current prompt is:
        "{prompt}"

        But it gets the following examples wrong:
        {error_str}

        Based on these examples the problem with this prompt is that {feedback_str}

        Based on the above information, I wrote {steps_per_gradient} different improved prompts.
        Each prompt is wrapped with <START> and <END>.

        The {steps_per_gradient} new prompts are:
        """
        transformation_prompt = "\n".join(
            [line.lstrip() for line in transformation_prompt.split("\n")]
        )
        res = utils.chatgpt(transformation_prompt, n=n)
        new_prompts = []
        for r in res:
            new_prompts += self.parse_tagged_text(r, "<START>", "<END>")
        return new_prompts

    def apply_gradient_v2(self, prompt, error_str, correct_str, error_feedback_str, correct_feedback_str, steps_per_gradient, step_size, n=1):
        """Incorporate feedback gradient into a prompt."""
        # I am allowed to change up to {step_size} words in the current prompt.
        transformation_prompt = f"""
        I'm trying to write a zero-shot classifier.
        
        My current prompt is:
        "{prompt}"


        But it gets the following examples wrong:
        {error_str}

        And it gets the following examples rght:
        {correct_str}

        Based on these examples the problem with this prompt is that {error_feedback_str}
        and the key points why this prompt predict those examples right is that {correct_feedback_str}

        Based on the above information, I wrote {steps_per_gradient} different improved prompts.
        Each prompt is wrapped with <START> and <END>.

        The {steps_per_gradient} new prompts are:
        """
        transformation_prompt = "\n".join(
            [line.lstrip() for line in transformation_prompt.split("\n")]
        )
        res = utils.chatgpt(transformation_prompt, n=n)
        new_prompts = []
        for r in res:
            new_prompts += self.parse_tagged_text(r, "<START>", "<END>")
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
            desc="gradients (negative feedback)..",
        ):
            error_string = self._sample_error_str(
                texts, labels, preds, task, n=self.opt["errors_per_gradient"]
            )
            gradients = self._get_gradients(
                task_section, error_string, self.opt["gradients_per_error"], n=1
            )
            prompt_feedbacks += [(t, error_string) for t in gradients]
        return prompt_feedbacks

    def get_gradients_v2(self, prompt, task_section, task, gpt4, texts, labels, preds):
        """Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        for _ in tqdm(
            range(self.opt["n_gradients"]),
            total=self.opt["n_gradients"],
            desc="gradients (negative feedback)..",
        ):
            error_string = self._sample_error_str(
                texts, labels, preds, task, n=self.opt["errors_per_gradient"]
            )
            gradients = self._get_gradients(
                task_section, error_string, self.opt["gradients_per_error"], n=1
            )
            prompt_feedbacks += ", ".join(gradients)
        return prompt_feedbacks, error_string

    def get_positive_feedback(self, prompt, task_section, task, gpt4, texts, labels, preds):
        prompt_feedbacks = []
        for _ in tqdm(
            range(self.opt["n_gradients"]),
            total=self.opt["n_gradients"],
            desc="gradients (positive feedback)..",
        ):
            correct_string = self._sample_correct_str(
                texts, labels, preds, task, n=self.opt["errors_per_gradient"]
            )
            gradients = self._get_positive_feedback(
                task_section, correct_string, self.opt["gradients_per_error"], n=1
            )
            prompt_feedbacks += [(t, correct_string) for t in gradients]
        return prompt_feedbacks

    def get_positive_feedback_v2(self, prompt, task_section, task, gpt4, texts, labels, preds):
        prompt_feedbacks = []
        for _ in tqdm(
            range(self.opt["n_gradients"]),
            total=self.opt["n_gradients"],
            desc="gradients (positive feedback)..",
        ):
            correct_string = self._sample_correct_str(
                texts, labels, preds, task, n=self.opt["errors_per_gradient"]
            )
            gradients = self._get_positive_feedback(
                task_section, correct_string, self.opt["gradients_per_error"], n=1
            )
            prompt_feedbacks += ", ".join(gradients)
        return prompt_feedbacks, correct_string

    def expand_candidates(self, prompts, task, gpt4, train_exs, step_size):
        """Expand a list of prompts using ERM, MC sampling, feedbacks, and exemplars."""
        minibatch = random.sample(train_exs, k=self.opt["minibatch_size"])
        new_prompts = []

        for prompt in tqdm(prompts, desc=f"expanding {len(prompts)} prompts"):
            sections = utils.parse_sectioned_prompt(prompt)
            task_section = sections["task"].strip()

            # Evaluate current prompt
            _, texts, labels, preds = task.evaluate(gpt4, prompt, minibatch)
            error_str = self._sample_error_str(texts, labels, preds, task, n=self.opt['errors_per_gradient'])

            # Generate and store feedbacks in memory
            feedbacks = self._get_gradients(task_section, error_str, num_feedbacks=self.opt['gradients_per_error'])
            self.fb_memory.store_feedbacks(feedbacks, [1.0]*len(feedbacks))

            # Generate and store exemplars
            exemplars = [(t, l, f"COT {t}") for t, l in zip(texts, labels)]
            self.ex_factory.store_exemplars(exemplars)

            # Retrieve top feedbacks and exemplars
            selected_feedbacks = self.fb_memory.retrieve_feedbacks(self.opt['n_gradients'])
            selected_exemplars = self.ex_factory.retrieve_exemplars("dummy query", n=self.opt['n_gradients'])

            # Apply feedbacks iteratively for ERM
            erm_task_sections = []
            for fb in selected_feedbacks:
                refined_prompts = self.apply_gradient(task_section, error_str, fb, steps_per_gradient=self.opt['steps_per_gradient'], step_size=step_size)
                erm_task_sections += refined_prompts

            # Generate synonyms via Monte Carlo
            mc_sampled_sections = []
            for sect in tqdm(erm_task_sections + [task_section], desc="MC sampling"):
                mc_sampled_sections += self.generate_synonyms(sect, n=self.opt["mc_samples_per_step"])

            # Merge ERM + MC, dedup
            combined_sections = list(set(erm_task_sections + mc_sampled_sections))

            # Apply candidate reflection if enabled
            if self.opt.get("reflect_candidates"):
                passes = max(1, int(self.opt.get("reflection_candidate_passes", 1)))
                combined_sections = self.apply_candidate_reflection(prompt, combined_sections, passes)

            # Build new prompts
            tmp_new_prompts = [prompt.replace(task_section, sec) for sec in combined_sections]

            # Filter top-K or randomly if too many
            if len(tmp_new_prompts) > self.opt['max_expansion_factor']:
                tmp_new_prompts = random.sample(tmp_new_prompts, k=self.opt['max_expansion_factor'])

            new_prompts += tmp_new_prompts

        # Add original prompts and dedup
        new_prompts += prompts
        new_prompts = list(set(new_prompts))

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
