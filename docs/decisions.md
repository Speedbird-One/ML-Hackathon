# Decision log

Newest entries at the bottom. Format: date, decision, alternatives considered, reasoning, evidence (fill in when available).

---

## 2026-11-18: Problem framing and candidate model families

**Framing (planned, to be validated by EDA).** For each GPS point, generate candidate road arcs nearby.
A classifier scores each candidate (true arc vs not). The HMM/Viterbi decoder uses these scores as emission
probabilities and road-network connectivity for transitions. The classifier therefore has to output usable
probabilities, so models are judged on log loss/calibration as well as F1 and top-1 accuracy.

**Model families to try** (the guidelines require at least five distinct models). Reasoning below is the
*pre-experiment* rationale; results and revised conclusions go in later entries.

| # | Family | Model | Why try it | Expected weakness |
|---|---|---|---|---|
| 1 | Linear / probabilistic | Logistic Regression (scaled) | Fast, well-calibrated baseline; shows how much simple features (distance, heading difference) explain | Cannot model interactions, e.g. distance mattering less at low speed |
| 2 | Probabilistic generative | Gaussian Naive Bayes | Near-free to train; contrasts with logistic regression | Independence assumption is violated (distance and along-track position are correlated); likely overconfident probabilities |
| 3 | Bagging ensemble | Random Forest / ExtraTrees | Nonlinear interactions, robust to noisy features, stable across folds | Less sharp probabilities; large artifacts, so cap depth and tree count for repo size |
| 4 | Boosting | LightGBM (or HistGradientBoosting) | Strong default on large tabular data, fast, good probabilities, handles imbalance; likely HPO candidate | May overfit trip-specific patterns; grouped CV should expose this |
| 5 | Neural network | MLP | Learns nonlinear representations; produces epoch-wise loss curves (the report asks for these only for models with histories) | Needs scaling and regularisation; may not beat boosting on tabular data |
| 6 | Instance-based / kernel | KNN on a subsample, or linear SVM / SGD | Adds a further family | RBF SVM and full KNN do not scale to ~1.5M candidate rows (estimate, not measured), so subsampling is needed and must be reported |

Optional: single Decision Tree (to show overfitting vs ensembles), XGBoost as a second boosting variant.

**Non-ML baseline.** Nearest-arc by distance with no classifier. Not counted among the five models; it shows
what ML adds.

**Not chosen:** LightGBM `lambdarank` (ranking objective). Scores are not probabilities, which fits the HMM
emission model less cleanly. Revisit only if time allows.

**Selection rule.** The final model is chosen from measured results (grouped CV on the 80 training trips plus
the 10 student-test trips), not in advance. LightGBM is a hypothesis, not a decision. The primary selection
metric will be recorded in a later entry once the candidate miss rate and class balance are known.

**Evidence:** none yet. Update this entry or add a follow-up after the first experiments.
