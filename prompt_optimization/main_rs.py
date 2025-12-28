import requests
import os
import evaluators
import concurrent.futures
from tqdm import tqdm
import time
import json
import argparse
import scorers
import tasks
import predictors
import optimizers
import my_optimizer_v2
import optimizers_fewshot
import optimizers_logits
import numpy as np
import sys
import random
import utils
from models import Prompt
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_task_class(task_name):
    if task_name == "ethos":
        return tasks.DefaultHFBinaryTask
    elif task_name == "jailbreak":
        return tasks.JailbreakBinaryTask
    elif task_name == "liar":
        return tasks.DefaultHFBinaryTask
    elif task_name == "ar_sarcasm":
        return tasks.DefaultHFBinaryTask
    elif task_name == "clickbait":
        return tasks.DefaultHFBinaryTask
    elif task_name == "casual_judgement":
        return tasks.DefaultHFBinaryTask
    elif task_name == "web_of_lies":
        return tasks.DefaultHFBinaryTask
    elif task_name == "sports_understanding":
        return tasks.DefaultHFBinaryTask
    else:
        return tasks.DefaultHFBinaryTask


def get_evaluator(evaluator):
    if evaluator == "bf":
        return evaluators.BruteForceEvaluator
    elif evaluator in {"ucb", "ucb-e"}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {"sr", "s-sr"}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == "sh":
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f"Unsupported evaluator: {evaluator}")


def get_scorer(scorer):
    if scorer == "01":
        return scorers.Cached01Scorer
    elif scorer == "ll":
        return scorers.CachedLogLikelihoodScorer
    else:
        raise Exception(f"Unsupported scorer: {scorer}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ethos")
    parser.add_argument("--data_dir", default="data/ethos")
    parser.add_argument("--prompts", default="prompts/ethos.md")
    # parser.add_argument('--config', default='default.json')
    parser.add_argument("--out", default="test_out.txt")
    parser.add_argument("--max_threads", default=32, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)

    parser.add_argument("--optimizer", default="nl-gradient")
    parser.add_argument("--rounds", default=6, type=int)
    parser.add_argument("--beam_size", default=4, type=int)
    parser.add_argument("--n_test_exs", default=400, type=int)

    parser.add_argument("--minibatch_size", default=64, type=int)
    parser.add_argument("--n_gradients", default=4, type=int)
    parser.add_argument("--errors_per_gradient", default=4, type=int)
    parser.add_argument("--gradients_per_error", default=1, type=int)
    parser.add_argument("--steps_per_gradient", default=1, type=int)
    parser.add_argument("--mc_samples_per_step", default=4, type=int)
    parser.add_argument("--max_expansion_factor", default=8, type=int)

    parser.add_argument("--engine", default="chatgpt", type=str)

    parser.add_argument("--evaluator", default="bf", type=str)
    parser.add_argument("--scorer", default="01", type=str)
    parser.add_argument("--eval_rounds", default=8, type=int)
    parser.add_argument("--eval_prompts_per_round", default=8, type=int)
    # calculated by s-sr and sr
    parser.add_argument("--samples_per_eval", default=32, type=int)
    parser.add_argument(
        "--c",
        default=1.0,
        type=float,
        help="exploration param for UCB. higher = more exploration",
    )
    parser.add_argument("--knn_k", default=2, type=int)
    parser.add_argument("--knn_t", default=0.993, type=float)
    parser.add_argument("--reject_on_errors", action="store_true")
    parser.add_argument(
        "--reflect_gradients",
        action="store_true",
        help="apply reflection to textual gradients before editing prompts",
    )
    parser.add_argument(
        "--reflect_candidates",
        action="store_true",
        help="filter generated prompts using reflection scores",
    )
    parser.add_argument(
        "--reflection_candidate_threshold",
        default=0.5,
        type=float,
        help="minimum reflection score to keep a candidate prompt",
    )
    parser.add_argument(
        "--reflection_gradient_passes",
        default=1,
        type=int,
        help="number of reflection passes to run over gradients when --reflect_gradients is enabled",
    )
    parser.add_argument(
        "--reflection_candidate_passes",
        default=1,
        type=int,
        help="number of reflection passes to run over generated prompts when --reflect_candidates is enabled",
    )
    parser.add_argument(
        "--reflection_temperature",
        default=0.0,
        type=float,
        help="temperature used for reflection LLM calls",
    )
    parser.add_argument("--ea-samples-per-step", default=4, type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    config = vars(args)

    config["eval_budget"] = (
        config["samples_per_eval"]
        * config["eval_rounds"]
        * config["eval_prompts_per_round"]
    )

    task = get_task_class(args.task)(args.data_dir, args.max_threads)
    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator("bf")(config)
    gpt4 = predictors.BinaryPredictor(config)

    optimizer = optimizers.ProTeGi(config, evaluator, scorer, args.max_threads, bf_eval)
    # optimizer = optimizers_fewshot.ProTeGi(config, evaluator, scorer, args.max_threads, bf_eval)
    # optimizer = optimizers_logits.ProTeGi(config, evaluator, scorer, args.max_threads, bf_eval)
    # optimizer = my_optimizer_v2.MyOptimizer(config, evaluator, scorer, args.max_threads, bf_eval)

    train_exs = task.get_train_examples()
    error_count = [0 for i in range(len(train_exs))]

    val_size = int(0.2 * len(train_exs))
    val_exs = random.sample(train_exs, val_size)
    # train_exs = [ex for ex in train_exs if ex not in val_exs]

    test_exs = task.get_test_examples()

    if os.path.exists(args.out):
        os.remove(args.out)

    print(config)

    with open(args.out, "a") as outf:
        outf.write(json.dumps(config) + "\n")

    candidates = [open(fp.strip()).read() for fp in args.prompts.split(",")]
    # candidates = [Prompt(open(fp.strip()).read(), set(), set(), 0, 0.5) for fp in args.prompts.split(",")]
    # sampled_examples = random.sample(train_exs, 5)
    # candidates = [optimizer.init_prompt_generation(i, sampled_examples) for i in candidates]

    # Instruction Optimization
    for round in tqdm(range(config["rounds"] + 1), desc="Instruction Optimization"):
        print("STARTING ROUND ", round)
        start = time.time()
        # expand candidates
        if round > 0:
            candidates = optimizer.expand_candidates(candidates, task, gpt4, train_exs, error_count)
            print("Error count: ", error_count)
        # score candidates
        scores = optimizer.score_candidates(candidates, task, gpt4, train_exs)
        [scores, candidates] = list(
            zip(*sorted(list(zip(scores, candidates)),key=lambda x: x[0], reverse=True))
        )

        # select candidates
        candidates = candidates[: config["beam_size"]]
        scores = scores[: config["beam_size"]]

        # record candidates, estimated scores, and true scores
        with open(args.out, "a") as outf:
            outf.write(f"======== ROUND {round}\n")
            outf.write(f"Time: {time.time() - start}\n")
            outf.write(f"Prompt: {candidates}\n")
            outf.write(f"Training accuracy: {scores}\n")
        val_metrics = []
        test_metrics = []
        for candidate, score in zip(candidates, scores):
            f1, texts, labels, preds = task.evaluate(
                gpt4, candidate, test_exs, n=len(test_exs)
            )
            test_metrics.append(f1)
        with open(args.out, "a") as outf:
            outf.write(f"Test accuracy: {test_metrics}\n")
            
    # Random Search Hyperparameters
    Q = 12          # population size
    k_min = 3       # min exemplars
    k_max = 6       # max exemplars
    t = 6           # number of random search rounds

    base_prompt = candidates[0]
    best_prompt = base_prompt
    best_score = -float("inf")

    def uniform_sample_without_replacement(exs, k):
        return random.sample(exs, k)

    for _ in tqdm(range(t), desc="Random Search (Keep Best)"):
        sections = utils.parse_sectioned_prompt(base_prompt)
        task_section = sections["task"].strip()

        # Sample uniformly at random
        populations = [
            uniform_sample_without_replacement(
                train_exs, random.randint(k_min, k_max)
            )
            for _ in range(Q)
        ]

        final_prompts = []
        for ex_list in populations:
            exemplar_block = "\n".join(utils.format_exemplar(ex) for ex in ex_list)
            prompt_with_ex = (
                task_section
                + "\n\n# Here are some examples:\n"
                + exemplar_block
            )

            start_marker = "# Task"
            end_marker = "# Output format"
            start_idx = base_prompt.find(start_marker)
            end_idx = base_prompt.rfind(end_marker)

            if start_idx != -1 and end_idx != -1:
                task_line_end = base_prompt.find("\n", start_idx) + 1
                final_prompt = (
                    base_prompt[:task_line_end]
                    + prompt_with_ex
                    + "\n"
                    + base_prompt[end_idx:]
                )
                final_prompts.append(final_prompt)

        # Metric-only scoring
        scores = optimizer.score_candidates(
            final_prompts, task, gpt4, train_exs
        )

        # Update global best
        round_best_idx = int(np.argmax(scores))
        round_best_score = scores[round_best_idx]

        if round_best_score > best_score:
            best_score = round_best_score
            best_prompt = final_prompts[round_best_idx]

        with open(args.out, "a") as outf:
            outf.write(f"======== EXEMPLAR OPTIMIZATION ROUND {round}\n")
            outf.write(f"Time: {time.time() - start}\n")
            outf.write(f"Prompt: {best_prompt}\n")
            outf.write(f"Training accuracy: {best_score}\n")
        f1, texts, labels, preds = task.evaluate(
            gpt4, best_prompt, test_exs, n=len(test_exs)
        )
        with open(args.out, "a") as outf:
            outf.write(f"Test accuracy: {f1}\n")

    print("DONE!")
