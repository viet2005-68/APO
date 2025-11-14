import os
import time
import json
import argparse
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import numpy as np

import evaluators
import scorers
import tasks
import predictors
import optimizers

# =========================
# Helper functions
# =========================
def get_task_class(task_name):
    mapping = {
        "ethos": tasks.EthosBinaryTask,
        "jailbreak": tasks.JailbreakBinaryTask,
        "liar": tasks.DefaultHFBinaryTask,
        "ar_sarcasm": tasks.DefaultHFBinaryTask
    }
    if task_name not in mapping:
        raise Exception(f"Unsupported task: {task_name}")
    return mapping[task_name]

def get_evaluator(evaluator):
    mapping = {
        "bf": evaluators.BruteForceEvaluator,
        "ucb": evaluators.UCBBanditEvaluator,
        "ucb-e": evaluators.UCBBanditEvaluator,
        "sr": evaluators.SuccessiveRejectsEvaluator,
        "s-sr": evaluators.SuccessiveRejectsEvaluator,
        "sh": evaluators.SuccessiveHalvingEvaluator
    }
    if evaluator not in mapping:
        raise Exception(f"Unsupported evaluator: {evaluator}")
    return mapping[evaluator]

def get_scorer(scorer):
    mapping = {
        "01": scorers.Cached01Scorer,
        "ll": scorers.CachedLogLikelihoodScorer
    }
    if scorer not in mapping:
        raise Exception(f"Unsupported scorer: {scorer}")
    return mapping[scorer]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="ethos")
    parser.add_argument("--data_dir", default="data/ethos")
    parser.add_argument("--prompts", default="prompts/ethos.md")
    parser.add_argument("--out", default="results/out.txt")
    parser.add_argument("--max_threads", default=32, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--n_gradients", type=int, default=4)
    parser.add_argument("--errors_per_gradient", default=4, type=int)
    parser.add_argument("--gradients_per_error", default=1, type=int)
    parser.add_argument("--mc_samples_per_step", default=2, type=int)
    parser.add_argument("--max_expansion_factor", default=8, type=int)
    parser.add_argument("--eval_rounds", default=8, type=int)
    parser.add_argument("--eval_prompts_per_round", default=8, type=int)
    parser.add_argument("--samples_per_eval", default=32, type=int)
    parser.add_argument("--eval_budget", type=int, default=1024)
    parser.add_argument("--reject_on_errors", action="store_true")

    parser.add_argument("--optimizer", default="nl-gradient")
    parser.add_argument("--rounds", default=6, type=int)
    parser.add_argument("--beam_size", default=4, type=int)
    parser.add_argument("--n_test_exs", default=400, type=int)

    parser.add_argument("--steps_per_gradient", default=1, type=int)
    parser.add_argument("--minibatch_size", default=64, type=int)

    parser.add_argument("--similarity_threshold", default=0.95, type=float)
    parser.add_argument("--n_clusters", default=5, type=int)

    parser.add_argument("--evaluator", default="bf", type=str)
    parser.add_argument("--scorer", default="01", type=str)
    args = parser.parse_args()
    return args

# =========================
# Main
# =========================
if __name__ == "__main__":
    args = get_args()
    config = vars(args)

    # =========================
    # Load task, scorer, evaluator, optimizer
    # =========================
    task = get_task_class(args.task)(args.data_dir, args.max_threads)
    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator("bf")(config)
    gpt4 = predictors.BinaryPredictor(config)
    optimizer = optimizers.ProTeGi(config, evaluator, scorer, args.max_threads, bf_eval)

    train_exs = task.get_train_examples()
    test_exs = task.get_test_examples()

    if os.path.exists(args.out):
        os.remove(args.out)

    with open(args.out, "a") as outf:
        outf.write(json.dumps(config) + "\n")

    candidates = [open(fp.strip()).read() for fp in args.prompts.split(",")]

    # =========================
    # Main optimization loop
    # =========================
    for round in tqdm(range(config["rounds"] + 1)):
        print(f"=== ROUND {round} ===")
        start = time.time()

        # Round 0: chỉ đánh giá prompts ban đầu
        if round == 0:
            print("Initial round - evaluating starting prompts")
            
        # Round > 0: expand và optimize với APO iteration
        else:
            print(f"Running APO iteration with {len(candidates)} beams...")
            
            # SỬ DỤNG apo_iteration từ optimizer
            candidates = optimizer.apo_iteration(
                candidates, 
                task, 
                gpt4, 
                train_exs, 
                top_k=config["beam_size"]
            )
            print(f"After APO iteration: {len(candidates)} candidates")

        # =========================
        # Evaluation và ghi kết quả (cho cả round 0 và round > 0)
        # =========================
        
        # Tính scores cho logging (chỉ để hiển thị, không dùng để selection)
        if candidates:
            scores = optimizer.score_candidates(candidates, task, gpt4, train_exs)
            
            # Beam selection cho round 0 (vì round 0 không chạy apo_iteration)
            if round == 0 and len(candidates) > config["beam_size"]:
                scored_candidates = sorted(zip(scores, candidates), reverse=True)
                scores, candidates = zip(*scored_candidates)
                beam_size = min(config["beam_size"], len(candidates))
                candidates = list(candidates[:beam_size])
                scores = list(scores[:beam_size])
                print(f"Selected top {beam_size} beams for round 0")
        else:
            scores = []

        # Ghi kết quả
        with open(args.out, "a") as outf:
            outf.write(f"ROUND {round}\n")
            outf.write(f"Time: {time.time() - start:.2f}s\n")
            outf.write(f"Candidates: {candidates}\n")
            outf.write(f"Scores: {scores if scores else 'N/A'}\n")

        # Evaluate trên test set
        if candidates:
            metrics = []
            for candidate in candidates:
                f1, texts, labels, preds = task.evaluate(gpt4, candidate, test_exs, n=args.n_test_exs)
                metrics.append(f1)
            with open(args.out, "a") as outf:
                outf.write(f"Test Metrics: {metrics}\n")
            print(f"Test F1 scores: {metrics}")

    print("DONE!")