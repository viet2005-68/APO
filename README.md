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

## Objective Function

We optimize the following objective:

$$
\text{score}
=
m\!\left(p^*, e_1, \ldots, e_k\right)
-
\lambda_{\text{len}} \, k
-
\lambda_{\text{div}} \, R_{\text{div}}(E)
$$



with the diversity regularization term:

$$
R_{\text{div}}(E)
=
\frac{1}{k(k-1)}
\sum_{i \ne j}
\operatorname{sim}(e_i, e_j)
$$

### Definitions

- $p^*$ : Optimized instruction prompt.
- $E = \{e_1, \ldots, e_k\}$ : Set of selected exemplars.
- $e_i$ : The $i$-th exemplar.
- $k = |E|$ : Number of exemplars in the prompt.

- $m(p^*, e_1, \ldots, e_k)$ :
  Task performance metric (e.g., accuracy, log-likelihood, reward)
  when using instruction $p^*$ together with exemplars $E$.

- $\lambda_{\mathrm{len}}$ :
  Length regularization coefficient penalizing large exemplar sets.

- $\lambda_{\mathrm{div}}$ :
  Diversity regularization coefficient.

- $\mathrm{sim}(e_i, e_j)$ :
  Similarity function between exemplars
  (e.g., cosine similarity of embeddings).

- $R_{\mathrm{div}}(E)$ :
  Average pairwise similarity among exemplars,
  encouraging diversity and reducing overfitting.


This formulation enables efficient exploration of the combinatorial exemplar space while balancing performance, compactness, and diversity.

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