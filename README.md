# Automatic Prompt Optimization with "Gradient Descent" and Beam Search

[![EMNLP 2023](https://img.shields.io/badge/EMNLP-2023-blue)](https://arxiv.org/abs/2305.03495)

## Overview

This is the official implementation of [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495) (EMNLP 2023).

The ProTeGi program provides a comprehensive framework for optimizing and evaluating prompts in text generation tasks. The program supports:

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

Run an experiment with UCB bandits for candidate selection:

```bash
time python main.py \
  --task ethos \
  --prompts prompts/ethos.md \
  --data_dir data/ethos \
  --out expt7_datasets/treatment.ucb.ethos.out \
  --evaluator ucb
```

This command will:

- Run an optimization experiment using UCB bandits
- Print configuration settings
- Provide progress updates for each optimization round
- Write results (candidate prompts and scores) to the specified output file

## Command Line Arguments

### Required Arguments

| Argument     | Description                    | Example                                  |
| ------------ | ------------------------------ | ---------------------------------------- |
| `--task`     | Task name                      | `ethos`, `jailbreak`                     |
| `--prompts`  | Path to prompt markdown file   | `prompts/ethos.md`                       |
| `--data_dir` | Directory containing task data | `data/ethos`                             |
| `--out`      | Output file path for results   | `expt7_datasets/treatment.ucb.ethos.out` |

### Optional Arguments

| Argument        | Description                   | Default        |
| --------------- | ----------------------------- | -------------- |
| `--evaluator`   | Evaluation strategy           | `ucb`          |
| `--max_threads` | Maximum number of threads     | System default |
| `--beam_size`   | Beam size for search          | -              |
| `--num_rounds`  | Number of optimization rounds | -              |

### View All Options

To see the complete list of available arguments:

```bash
python main.py --help
```

## Project Structure

```
.
├── main.py              # Main entry point
├── prompts/             # Prompt templates
│   └── ethos.md
├── data/                # Task datasets
│   └── ethos/
└── expt7_datasets/      # Experiment outputs
```

## Output

The program generates results including:

- 🎯 Optimized candidate prompts
- 📊 Associated performance scores
- 📈 Optimization progress metrics
- 💾 Detailed logs and statistics

## Examples

### Example 1: Ethos Task with UCB

```bash
python main.py \
  --task ethos \
  --prompts prompts/ethos.md \
  --data_dir data/ethos \
  --out results/ethos_ucb.out \
  --evaluator ucb
```

### Example 2: Jailbreak Task

```bash
python main.py \
  --task jailbreak \
  --prompts prompts/jailbreak.md \
  --data_dir data/jailbreak \
  --out results/jailbreak.out
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{protegi2023,
  title={Automatic Prompt Optimization with "Gradient Descent" and Beam Search},
  author={Your Name},
  booktitle={EMNLP},
  year={2023}
}
```

## License

[Add your license information here]

## Contact

[Add contact information or links to issues/discussions]

## Acknowledgments

[Add any acknowledgments here]

## Running Reflection-based Optimization

### Basic Usage with Reflection

```bash
python main.py \
  --task ethos \
  --prompts prompts/ethos.md \
  --data_dir data/ethos \
  --reflect_gradients \
  --reflect_candidates \
  --reflection_candidate_threshold 0.5 \
  --reflection_gradient_passes 1
```

### Reflection Parameters

| Argument                           | Description                            | Default |
| ---------------------------------- | -------------------------------------- | ------- |
| `--reflect_gradients`              | Apply reflection to textual gradients  | `False` |
| `--reflect_candidates`             | Filter prompts using reflection scores | `False` |
| `--reflection_candidate_threshold` | Minimum score to keep prompt           | `0.5`   |
| `--reflection_gradient_passes`     | Number of gradient reflection passes   | `1`     |
| `--reflection_temperature`         | Temperature for reflection calls       | `0.0`   |

### ⚠️ Overfitting Considerations

The reflection process can lead to overfitting issues:

1. **Performance Gap**

   - High scores on training data (>90%)
   - Significantly lower performance on test set
   - Gap increases with more reflection passes

2. **Common Issues**
   - Reflection becomes too specific to training examples
   - Reduced generalization on unseen cases
   - Over-optimization of reflection scores

### Best Practices

To minimize overfitting:

- Use lower reflection thresholds (0.3-0.4)
- Limit reflection passes to 1
- Implement cross-validation
- Regular test set performance monitoring
- Balance reflection intensity with generalization needs
