
# Introduction

This is code for [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495) (EMNLP 2023).

The ProTeGi program offers a framework for optimizing and evaluating prompts in text generation tasks. The program supports a variety of evaluation strategies, scoring mechanisms, and is designed to work with tasks that involve binary classification.

The main entrypoint is `main.py`

# Quickstart:
```
time python main.py --task ethos --prompts prompts/ethos.md --data_dir data/ethos --out expt7_datasets/treatment.ucb.ethos.out --evaluator ucb
```

This will run an experiment with UCB bandits for candidate selection. The program will print configuration settings and provide progress updates with each optimization round. The results, including candidate prompts and their associated scores, will be written to the specified output file.

```
python main.py --help
```

For usage instructions. Some of the arguments include:

* `--task`: Task name like 'ethos', 'jailbreak', etc.
* `--data_dir`: Directory where the task data resides.
* `--prompts`: Path to the prompt markdown file.
* `--out`: Output file name.
* `--max_threads`: Maximum number of threads to be used.
* `...`: Various other parameters related to optimization and evaluation.

# Liar Task
python main.py \
  --task liar \
  --prompts prompts/liar.md \
  --data_dir data/liar \
  --beam_size 4 \
  --steps_per_gradient 1 \
  --minibatch_size 256 \
  --out experiments/liar-14-12.out \
  --evaluator ucb \
  --reject_on_error

# Ethos Task
python main.py \
  --task ethos \
  --prompts prompts/ethos.md \
  --data_dir data/ethos \
  --beam_size 4 \
  --steps_per_gradient 1 \
  --minibatch_size 64 \
  --out experiments/ethos.out \
  --evaluator bf \
  --reject_on_error

# Ar Sacarsm
python main.py \
  --task ethos \
  --prompts prompts/ar_sarcasm.md \
  --data_dir data/ar_sarcasm \
  --beam_size 4 \
  --steps_per_gradient 1 \
  --minibatch_size 256 \
  --out experiments/ar_sarcasm.out \
  --evaluator ucb \
  --reject_on_error \
  --rounds 0
  
# Clickbait
python main.py \
  --task clickbait \
  --prompts prompts/clickbait.md \
  --data_dir data/clickbait \
  --beam_size 4 \
  --steps_per_gradient 1 \
  --minibatch_size 64 \
  --out experiments/clickbait.out \
  --evaluator ucb \
  --reject_on_error

# Casual Judgement (BBH)
python main.py \
  --task casual_judgement \
  --prompts prompts/casual_judgement.md \
  --data_dir data/casual_judgement \
  --beam_size 4 \
  --steps_per_gradient 1 \
  --minibatch_size 64 \
  --out experiments/casual_judgement.out \
  --evaluator bf 