# Extending Automatic Prompt Optimization with "Gradient Descent" and Beam Search: Exemplar Optimization for Prompting

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

<table>
<tr>
<td valign="top">

<h3>Qwen3-14B</h3>

<table>
  <tr>
    <th>Task</th>
    <th>ProTeGi</th>
    <th>Ours</th>
  </tr>
  <tr><td>Liar</td><td>0.57</td><td><b>0.61</b></td></tr>
  <tr><td>Ethos</td><td>0.85</td><td><b>0.88</b></td></tr>
  <tr><td>Casual judgement</td><td>0.63</td><td><b>0.68</b></td></tr>
  <tr><td>Web of lies</td><td>0.50</td><td><b>0.51</b></td></tr>
  <tr><td>Sports understanding</td><td><b>0.85</b></td><td>0.81</td></tr>
  <tr><td>Boolean expressions</td><td>0.85</td><td><b>0.91</b></td></tr>
  <tr><td><b>Average</b></td><td>0.71</td><td><b>0.73</b></td></tr>
</table>

</td>

<td style="width:20px;"></td>

<td valign="top">

<h3>Qwen2.5-32B-Instruct-AWQ</h3>

<table>
  <tr>
    <th>Task</th>
    <th>ProTeGi</th>
    <th>Ours</th>
  </tr>
  <tr><td>Liar</td><td>0.64</td><td><b>0.67</b></td></tr>
  <tr><td>Ethos</td><td>0.89</td><td><b>0.90</b></td></tr>
  <tr><td>Casual judgement</td><td><b>0.69</b></td><td><b>0.69</b></td></tr>
  <tr><td>Web of lies</td><td>0.66</td><td><b>0.72</b></td></tr>
  <tr><td>Sports understanding</td><td>0.80</td><td><b>0.83</b></td></tr>
  <tr><td>Boolean expressions</td><td>0.83</td><td><b>0.90</b></td></tr>
  <tr><td><b>Average</b></td><td>0.75</td><td><b>0.78</b></td></tr>
</table>

</td>
</tr>
</table>

<p><i>*All results are reported using accuracy.</i></p>

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