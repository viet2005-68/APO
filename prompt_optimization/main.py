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
import numpy as np
import sys
import random
from models import Prompt

def get_task_class(task_name):
    if task_name == "ethos":
        return tasks.DefaultHFBinaryTask
    elif task_name == "jailbreak":
        return tasks.JailbreakBinaryTask
    elif task_name == "liar":
        return tasks.DefaultHFBinaryTask
    elif task_name == "ar_sarcasm":
        return tasks.DefaultHFBinaryTask
    else:
        raise Exception(f"Unsupported task: {task_name}")


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

    # optimizer = optimizers.ProTeGi(config, evaluator, scorer, args.max_threads, bf_eval)
    optimizer = my_optimizer_v2.MyOptimizer(config, evaluator, scorer, args.max_threads, bf_eval)

    train_exs = task.get_train_examples()
    # random.shuffle(train_exs)
    batch_train_exs = [
        train_exs[i:i + config["minibatch_size"]]
        for i in range(0, len(train_exs), config["minibatch_size"])
    ]
    num_batches = len(batch_train_exs)

    validation_exs = task.get_validation_examples()
    test_exs = task.get_test_examples()

    if os.path.exists(args.out):
        os.remove(args.out)

    print(config)

    with open(args.out, "a") as outf:
        outf.write(json.dumps(config) + "\n")

    # candidates = [open(fp.strip()).read() for fp in args.prompts.split(",")]
    candidates = [Prompt(open(fp.strip()).read(), set(), set(), 0, 0.5) for fp in args.prompts.split(",")]
    sampled_examples = random.sample(train_exs, 5)
    # candidates = [optimizer.init_prompt_generation(i, sampled_examples) for i in candidates]
    initial_step_size = 200
    final_step_size = 20
    for round in tqdm(range(config["rounds"] + 1)):
        print("STARTING ROUND ", round)
        start = time.time()
        current_batch = batch_train_exs[round % num_batches]
        # expand candidates
        if round > 0:
            current_step_size = int(final_step_size + 0.5*(initial_step_size - final_step_size) * (1 + np.cos(np.pi * ((round-1) / (config["rounds"]-1)))))
            candidates = optimizer.expand_candidates(candidates, task, gpt4, train_exs)

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
                gpt4, candidate.prompt, validation_exs, n=len(validation_exs)
            )
            val_metrics.append(f1)
        with open(args.out, "a") as outf:
            outf.write(f"Validation accuracy: {val_metrics}\n")
        for candidate, score in zip(candidates, scores):
            f1, texts, labels, preds = task.evaluate(
                gpt4, candidate.prompt, test_exs, n=len(test_exs)
            )
            test_metrics.append(f1)
        with open(args.out, "a") as outf:
            outf.write(f"Test accuracy: {test_metrics}\n")

    print("DONE!")
