import json
import numpy as np
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import utils
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class PromptOptimizer(ABC):
    def __init__(self, args, evaluator_fn, scorer, max_threads=1, bf_eval=None):
        self.opt = args
        self.evaluator_fn = evaluator_fn
        self.scorer = scorer
        self.max_threads = max_threads
        self.bf_eval = bf_eval
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.n_clusters = 5

    @abstractmethod
    def expand_candidates(self, prompts, task, gpt4, train_exs):
        pass


class ProTeGi(PromptOptimizer):
    """ProTeGi: Prompt Optimization with Textual Gradients"""

    def get_ngrams(self, text, n=3):
        tokens = text.split()
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    def ngram_overlap(self, a, b, n=3):
        A = self.get_ngrams(a, n)
        B = self.get_ngrams(b, n)
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    def token_overlap(self, a, b):
        A = set(a.split())
        B = set(b.split())
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)

    def embed_prompts(self, prompts):
        embeddings = self.model.encode(prompts, show_progress_bar=True)
        return embeddings
    
    def cluster_embeddings(self, embeddings):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        return clusters, kmeans.cluster_centers_
    
    def cluster_prompts(self, prompts):
        if not prompts:
            return []

        embeddings = self.embed_prompts(prompts)
        n_clusters = min(self.n_clusters, len(prompts))  # tránh số cluster > số prompt
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        clustered = {i: [] for i in range(n_clusters)}
        for label, prompt in zip(labels, prompts):
            clustered[label].append(prompt)

        # lấy 1 prompt từ mỗi cluster nếu không rỗng
        clustered_candidates = []
        for cluster_list in clustered.values():
            if cluster_list:  # chỉ thêm nếu cluster không rỗng
                clustered_candidates.append(cluster_list[0])

        return clustered_candidates

    def compute_diversity_penalty(self, prompt, other_prompts, w_ngram=0.3, w_token=0.3, w_sem=0.4):
        """Compute diversity penalty for a prompt relative to other prompts."""
        if not other_prompts:
            return 0.0

        # ngram & token overlaps
        ngram_sims = [self.ngram_overlap(prompt, p) for p in other_prompts]
        token_sims = [self.token_overlap(prompt, p) for p in other_prompts]

        # semantic similarity
        embeddings = self.model.encode([prompt] + other_prompts)
        sem_sims = cosine_similarity([embeddings[0]], embeddings[1:])[0]

        penalty = w_ngram * np.mean(ngram_sims) + w_token * np.mean(token_sims) + w_sem * np.mean(sem_sims)
        return penalty
    
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

    def apply_gradient(self, prompt, error_str, feedback_str, steps_per_gradient, n=1):
        """Incorporate feedback gradient into a prompt."""
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
                if self.opt.get("reflect_gradients"):
                    gradient_passes = max(
                        1, int(self.opt.get("reflection_gradient_passes", 1))
                    )
                    for _ in range(gradient_passes):
                        gradients = self.reflect_feedbacks(task_section, gradients)
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

            # generate synonyms
            mc_sampled_task_sections = []
            if self.opt["mc_samples_per_step"] > 0:
                for sect in tqdm(new_task_sections + [task_section], desc="mc samples"):
                    mc_sects = self.generate_synonyms(
                        sect, n=self.opt["mc_samples_per_step"]
                    )
                    mc_sampled_task_sections += mc_sects

            # combine
            new_sections = new_task_sections + mc_sampled_task_sections
            new_sections = list(set(new_sections))  # dedup
            tmp_new_prompts = [
                prompt.replace(task_section, tmp) for tmp in new_sections
            ]

            if self.opt.get("reflect_candidates"):
                candidate_passes = max(
                    1, int(self.opt.get("reflection_candidate_passes", 1))
                )
                tmp_new_prompts = self.apply_candidate_reflection(
                    prompt, tmp_new_prompts, candidate_passes
                )

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

    def apo_iteration(self, beams, task, gpt4, train_exs, top_k=5):
        """One iteration of APO: expand, cluster, score, penalize, select top-k."""
        print(f"Starting APO iteration with {len(beams)} beams")
        
        # 1. Expand candidates
        new_candidates = []
        for prompt in beams:
            expanded = self.expand_candidates([prompt], task, gpt4, train_exs)
            new_candidates.extend(expanded)
            print(f"Expanded beam -> {len(expanded)} candidates")
        
        new_candidates = list(set(new_candidates))
        print(f"Total unique candidates after expansion: {len(new_candidates)}")
        
        if not new_candidates:
            print("No new candidates generated, returning original beams")
            return beams

        # 2. Cluster embeddings
        try:
            embeddings = self.embed_prompts(new_candidates)
            
            # Xử lý clustering an toàn
            if len(new_candidates) <= 1:
                cluster_labels = [0] * len(new_candidates)
            else:
                n_clusters = min(self.n_clusters, len(new_candidates))
                print(f"Clustering {len(new_candidates)} candidates into {n_clusters} clusters")
                cluster_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(embeddings)
                
            # Thống kê cluster
            from collections import Counter
            cluster_counts = Counter(cluster_labels)
            print(f"Cluster distribution: {dict(cluster_counts)}")
            
        except Exception as e:
            print(f"Error in clustering: {e}")
            # Fallback: không clustering
            cluster_labels = [0] * len(new_candidates)

        # 3. Calculate rewards
        print("Scoring candidates...")
        rewards = self.score_candidates(new_candidates, task, gpt4, train_exs)
        print(f"Reward range: {min(rewards):.3f} - {max(rewards):.3f}")

        # 4. Diversity penalty within cluster
        print("Calculating diversity penalties...")
        penalties = []
        for i, cand in enumerate(new_candidates):
            cluster = cluster_labels[i]
            # Lấy các candidate khác trong cùng cluster
            other_prompts = [p for j, p in enumerate(new_candidates) 
                            if cluster_labels[j] == cluster and j != i]
            
            if not other_prompts:
                # Nếu là candidate duy nhất trong cluster, penalty = 0
                penalty = 0.0
            else:
                penalty = self.compute_diversity_penalty(cand, other_prompts)
            
            penalties.append(penalty)

        # 5. Final score = reward - penalty
        scores = [max(0.0, r - p) for r, p in zip(rewards, penalties)]
        print(f"Final score range: {min(scores):.3f} - {max(scores):.3f}")

        # 6. Top-k selection
        actual_top_k = min(top_k, len(scores))
        top_idxs = np.argsort(scores)[-actual_top_k:]
        top_beams = [new_candidates[i] for i in top_idxs]
        top_scores = [scores[i] for i in top_idxs]

        print(f"Selected top {actual_top_k} beams with scores: {[f'{s:.3f}' for s in top_scores]}")
        
        return top_beams