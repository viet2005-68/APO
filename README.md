# Automatic Prompt Optimization with "Gradient Descent" and Beam Search

This repository presents an extension to the paper  
**Automatic Prompt Optimization with "Gradient Descent" and Beam Search**  
([EMNLP 2023](https://arxiv.org/abs/2305.03495)).

Our work builds on the original **ProTeGi** framework by introducing an explicit **exemplar optimization stage**, aiming to further improve prompt effectiveness while controlling prompt length and redundancy.

---

## Background

The original ProTeGi framework provides a principled pipeline for automatic prompt optimization, combining:
- Gradient-based instruction refinement
- Beam search over prompt candidates
- Evaluation-driven optimization

While effective, ProTeGi does not explicitly optimize exemplars. The original ProTeGi framework focuses on optimizing the instruction prompt via gradient-based updates and beam search. Exemplar selection, when used, is treated as fixed or externally provided and is not part of the optimization objective.

---

## Our Contribution: Exemplar Optimization via Genetic Search

We augment the original pipeline with a **genetic algorithm–based exemplar selection module**.

The goal is to select a set of exemplars that:
- Maximizes downstream task performance
- Penalizes excessive prompt length
- Encourages diversity among exemplars to reduce redundancy and overfitting

Each individual in the genetic population represents a **candidate exemplar set**, which is evolved using selection and mutation operators.

---

## Results

### Qwen3-14B

| Task        | ProTeGi | Ours |
|-------------|---------|------|
| Liar        | 0.57    | **0.61** |
| Casual      | 0.63    | **0.68** |
| Clickbait   | **0.95** | 0.94 |
| Ethos       | 0.85    | **0.88** |
| Web of lies | 0.50    | **0.51** |
| **Average** | 0.70    | **0.73** |

---

### Qwen2.5-32B-Instruct-AWQ

| Task        | ProTeGi | Ours |
|-------------|---------|------|
| Liar        | 0.64    | **0.67** |
| Casual      | nan     | nan |
| Clickbait   | nan     | nan |
| Ethos       | nan     | nan |
| Web of lies | 0.66    | **0.72** |
| **Average** | 0.65    | **0.70** |

*All results are reported using accuracy.*

---

## Notes

- This repository is **not an official ProTeGi implementation**.
- The code is intended as a **research exploration** of exemplar optimization.

---

## Acknowledgements

This work is inspired by and builds upon:

> **Automatic Prompt Optimization with "Gradient Descent" and Beam Search**  
> EMNLP 2023  
> https://arxiv.org/abs/2305.03495

We thank the original authors for making their work publicly available.

---

## Citation

If you find this work useful, please consider citing the original ProTeGi paper.