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
import moe_optimizers
import numpy as np
import sys
import random
import utils
from models import Prompt
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import embedder
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

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
    embedder = embedder.Embedder()

    optimizer = moe_optimizers.MOEProTeGi(config, evaluator, scorer, args.max_threads, bf_eval)

    train_exs = task.get_train_examples()

    test_exs = task.get_test_examples()

    if os.path.exists(args.out):
        os.remove(args.out)

    print(config)

    with open(args.out, "a") as outf:
        outf.write(json.dumps(config) + "\n")
    EXPERT_NUM = 3
    candidates = [open(fp.strip()).read() for fp in args.prompts.split(",")]
    experts = defaultdict(list)
    for i in range(EXPERT_NUM):
        experts[i] = candidates

    # Instruction Optimization
    for round in tqdm(range(config["rounds"] + 1), desc="Instruction Optimization"):
        print("STARTING ROUND ", round)
        start = time.time()
        # expand candidates
        if round == 0:
            _, texts, labels, preds = task.evaluate(gpt4, candidates[0], train_exs)
            error_items = []
            for t, l, p in zip(texts, labels, preds):
                if l != p:
                    error_items.append({
                        "text": t,
                        "label": l,
                        "pred": p,
                        "error_string": (
                            f"Input text:\n{t.strip()}\n\n"
                            f"Model prediction: {task.stringify_prediction(p)}\n"
                            f"Correct answer: {task.stringify_prediction(l)}"
                        )
                    })
            error_strings = [e["error_string"] for e in error_items]
            embeddings = embedder.embed_texts(error_strings)
            X = normalize(embeddings)
            all_texts = [t for t in texts]
            all_embeds = embedder.embed_texts(all_texts)
            all_X = normalize(all_embeds)
            kmeans = KMeans(
                n_clusters=EXPERT_NUM,
                n_init=20,
                random_state=42
            )
            cluster_ids = kmeans.fit_predict(X)
            clusters = defaultdict(lambda: {
                "texts": [],
                "labels": [],
                "preds": []
            })
            for idx, cid in enumerate(cluster_ids):
                clusters[cid]["texts"].append(error_items[idx]['text'])
                clusters[cid]["labels"].append(error_items[idx]['label'])
                clusters[cid]["preds"].append(error_items[idx]['pred'])
            # Reassign exs to cluster
            centroids = kmeans.cluster_centers_
            centroids = normalize(centroids)
            assignments = defaultdict(list)
            for idx, (l, p) in enumerate(zip(labels, preds)):
                if l == p:
                    sims = cosine_similarity(
                        all_X[idx].reshape(1, -1),
                        centroids
                    )[0]
                    best_cid = np.argmax(sims)
                    assignments[best_cid].append(idx)
            expert_exs = defaultdict(list)
            # add errors
            for cid, items in clusters.items():
                for text, label in zip(items["texts"], items["labels"]):
                    expert_exs[cid].append({
                        "text": text,
                        "label": label
                    })
            # add corrects
            for cid, idxs in assignments.items():
                for idx in idxs:
                    expert_exs[cid].append({
                        "text": texts[idx],
                        "label": labels[idx]
                    })
            for i in tqdm(range(EXPERT_NUM), desc=f"Optimizing expert"):
                start = time.time()
                candidates = optimizer.expand_candidates(experts[i], task, gpt4, expert_exs[i])
                scores = optimizer.score_candidates(candidates, task, gpt4, expert_exs[i])
                [scores, candidates] = list(
                    zip(*sorted(list(zip(scores, candidates)),key=lambda x: x[0], reverse=True))
                )
                experts[i] = candidates[: config["beam_size"]]
                scores = scores[: config["beam_size"]]
                with open(args.out, "a") as outf:
                    outf.write(f"======== ROUND {round} | EXPERT {i}\n")
                    outf.write(f"Time: {time.time() - start}\n")
                    outf.write(f"Prompt: {experts[i]}\n")
                    outf.write(f"Training accuracy: {scores}\n")

        elif round > 0:
            # embed all experts' best prompts
            expert_prompts = [experts[i][0] for i in range(EXPERT_NUM)]
            expert_embeds = embedder.embed_texts(expert_prompts)
            expert_embeds = normalize(expert_embeds)
            # route each example
            expert_buffers = defaultdict(list)
            for idx, x_embed in enumerate(all_X):
                sims = cosine_similarity(x_embed.reshape(1, -1), expert_embeds)[0]
                best_expert = np.argmax(sims)
                expert_buffers[best_expert].append(train_exs[idx])

            for i in tqdm(range(EXPERT_NUM), desc=f"Opimizing expert {i}"):
                start = time.time()
                exs_i = expert_buffers[i]
                if len(exs_i) == 0:
                    continue
                candidates = optimizer.expand_candidates(experts[i], task, gpt4, exs_i)
                scores = optimizer.score_candidates(candidates, task, gpt4, exs_i)
                [scores, candidates] = list(
                    zip(*sorted(list(zip(scores, candidates)),key=lambda x: x[0], reverse=True))
                )
                experts[i] = candidates[: config["beam_size"]]
                scores = scores[: config["beam_size"]]
                with open(args.out, "a") as outf:
                    outf.write(f"======== ROUND {round} | EXPERT {i}\n")
                    outf.write(f"Time: {time.time() - start}\n")
                    outf.write(f"Prompt: {experts[i]}\n")
                    outf.write(f"Training accuracy: {scores}\n")

        # Expert distillation
        expert_prompts = [experts[i][0] for i in range(EXPERT_NUM)]
        experts_str = ""
        for i in range(EXPERT_NUM):
            sections = utils.parse_sectioned_prompt(expert_prompts[i])
            task_section = sections["task"].strip()

            experts_str += (
                f"\n### Expert {i}\n"
                f"{task_section}\n"
            )

        final_prompt = optimizer.distill_moe(experts=experts_str)

        start_marker = "# Task"
        end_marker = "# Output format"
        start_idx = candidates[0].find(start_marker)
        end_idx = candidates[0].rfind(end_marker)
        if start_idx != -1 and end_idx != -1:
            task_line_end = candidates[0].find("\n", start_idx) + 1
            final_prompt = candidates[0][:task_line_end] + final_prompt + "\n" + candidates[0][end_idx:]
        print(final_prompt)

        # record candidates, estimated scores, and true scores
        with open(args.out, "a") as outf:
            outf.write(f"======== ROUND {round}\n")
            outf.write(f"Time: {time.time() - start}\n")
            outf.write(f"Prompt: {final_prompt}\n")
        test_metrics = []
        f1, texts, labels, preds = task.evaluate(
            gpt4, final_prompt, test_exs, n=len(test_exs)
        )
        with open(args.out, "a") as outf:
            outf.write(f"Test accuracy: {f1}\n")
            
    # Exemplar Optimization
    val_size = int(0.2 * len(train_exs))
    val_exs = random.sample(train_exs, val_size)
    train_exs = [ex for ex in train_exs if ex not in val_exs]

    best_prompt = final_prompt
    Q = 12 # Population Size
    k_min = 3 # Min exemplars
    k_max = 6 # Max exemplars
    t = 6 # Optimization rounds
    lambda_len = 0.01 # Length penalize parameter
    lambda_div = 0.3 # Diversity penalize parameter
    populations = [
        random.sample(train_exs, random.randint(k_min, k_max))
        for _ in range(Q)
    ]
    print([len(p) for p in populations])
    for round in tqdm(range(t), desc="Exemplar Optimization"):
        start = time.time()
        sections = utils.parse_sectioned_prompt(best_prompt)
        task_section = sections['task'].strip()
        best_prompt_with_exemplar = []
        for ex_list in populations:
            exemplar_block = "\n".join(utils.format_exemplar(ex) for ex in ex_list)
            prompt_with_ex = task_section + "\n\n# Here are some examples:\n" + exemplar_block
            best_prompt_with_exemplar.append(prompt_with_ex)

        start_marker = "# Task"
        end_marker = "# Output format"
        final_prompts = []
        for tmp in best_prompt_with_exemplar:
            start_idx = best_prompt.find(start_marker)
            end_idx = best_prompt.rfind(end_marker)
            if start_idx != -1 and end_idx != -1:
                task_line_end = best_prompt.find("\n", start_idx) + 1
                final_prompt = best_prompt[:task_line_end] + tmp + "\n" + best_prompt[end_idx:]
                final_prompts.append(final_prompt)

        scores = optimizer.score_candidates(final_prompts, task, gpt4, val_exs)
        scores_after_regularize = [
            s - (lambda_len * len(pop)) - (lambda_div * optimizer.diversity_penalize(pop))
            for s, pop in zip(scores, populations)
        ]

        scores, scores_after_regularize, full_population_prompts, populations = zip(
            *sorted(
                zip(scores, scores_after_regularize, final_prompts, populations),
                key=lambda x: x[1],
                reverse=True
            )
        )
        scores = list(scores)
        scores_after_regularize = list(scores_after_regularize)
        full_population_prompts = list(full_population_prompts)
        populations = list(populations)
        best_prompt_with_ex = full_population_prompts[0]
        
        # Mutations
        num_elites = Q // 2
        elites = populations[:num_elites]
        children = []
        for parent in elites:
            child = list(parent)
            mutation_type = random.choice(["replace", "add", "remove"])
            if mutation_type == "add" and len(child) < k_max:
                child.append(random.choice(train_exs))
            elif mutation_type == "remove" and len(child) > k_min:
                remove_idx = random.randrange(len(child))
                child.pop(remove_idx)
            elif mutation_type == "replace" and len(child) > 0:
                replace_idx = random.randrange(len(child))
                child[replace_idx] = random.choice(train_exs)
            children.append(child)

        populations = elites + children

        with open(args.out, "a") as outf:
            outf.write(f"======== EXEMPLAR OPTIMIZATION ROUND {round}\n")
            outf.write(f"Time: {time.time() - start}\n")
            outf.write(f"Prompt: {best_prompt_with_ex}\n")
            outf.write(f"Training accuracy: {scores[0]} | Penalized score: {scores_after_regularize[0]}\n")
        f1, texts, labels, preds = task.evaluate(
            gpt4, best_prompt_with_ex, test_exs, n=len(test_exs)
        )
        with open(args.out, "a") as outf:
            outf.write(f"Test accuracy: {f1}\n")

    print("DONE!")
