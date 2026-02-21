# Isotonic Reward Distillation: Design & Method

## Core Concept
The Ordinal Reward Models (ORM) used in this repository excel at ranking but produce scores on uncalibrated scales (e.g., Bradley-Terry logits). For downstream Reinforcement Learning from Human Feedback (RLHF) like PPO, a stable and interpretable scalar reward is required.

**Isotonic Distillation** solves this by mapping the ORM output to a "Golden" scalar reward distribution while strictly preserving the model's ranking ability.

## Implementation: Two Models for Scalar Rewards

Through this pipeline, we obtain two ways to generate scalar rewards:

### 1. Model A: Calibrated Teacher (ORM + Isotonic Mapping)
- **Method:** `Teacher_MLP(x)` -> `Isotonic_Regression.transform(score)`
- **Pros:** Maximum fidelity to the teacher's ranking.
- **Cons:** Slightly more complex inference (two steps: neural network + mapping function).

### 2. Model B: Distilled Student (Single MLP)
- **Method:** `Student_MLP(x)`
- **Pros:** Fastest inference, standard PPO compatibility (just a single `.ckpt` file).
- **Cons:** May have a slight "distillation loss" relative to Model A.

---

## Future Research: Beyond Outcome Rewards

Once you have a calibrated scalar reward, several advanced techniques become possible:

### A. Process-Supervised Reward Models (PRM)
Instead of scoring a full response, you can apply this calibrated scalar model to step-by-step reasoning. Significant "reward drops" between steps indicate where a logic chain failed.

### B. Uncertainty-Aware Reward Shaping
By using an ensemble of distilled models, you can compute the variance. High variance (model disagreement) can be used to penalize PPO updates in "out-of-distribution" regions of the prompt space.

### C. Iterative Alignment (Re-Ranking & Self-Correction)
You can use the Distilled RM to relabel existing preference datasets. If the Distilled RM finds a "rejected" response actually has a higher calibrated score than the "preferred" one, it can filter noisy data for the next round of DPO/RLHF.

### D. Steerable Rewards
With a calibrated scale, you can optimize for specific quantiles of the reward distribution (e.g., CVaR) to prioritize avoiding low-quality/toxic outputs over simply achieving the highest average score.
