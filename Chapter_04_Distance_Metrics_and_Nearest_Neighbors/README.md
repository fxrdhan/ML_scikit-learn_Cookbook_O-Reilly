# Chapter 4 — Building Models with Distance Metrics and Nearest Neighbors

**Notebook:** [`Chapter_04_Distance_Metrics_and_Nearest_Neighbors.ipynb`](Chapter_04_Distance_Metrics_and_Nearest_Neighbors.ipynb)
<a href="https://colab.research.google.com/github/fxrdhan/ML_scikit-learn_Cookbook_O-Reilly/blob/main/Chapter_04_Distance_Metrics_and_Nearest_Neighbors/Chapter_04_Distance_Metrics_and_Nearest_Neighbors.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Summary

Chapter 4 turns the everyday idea of similarity into models. It introduces **distance metrics** — Euclidean, Manhattan and their generalization, Minkowski. It then builds the **k-nearest neighbors (KNN)** classifier, which predicts the majority label among the $k$ training points closest to a new point (or their average, for regression). The recipes:

- compare metrics on two synthetic datasets (noisy circles and a checkerboard);
- tune $k$, the neighbour weighting and the metric with `GridSearchCV` (with `RandomizedSearchCV` as the alternative for large spaces);
- evaluate the tuned model with cross-validation scores, learning curves, a confusion matrix and a classification report (precision, recall, F1, macro and weighted averages), with advice for imbalanced data.

Three exercises then build, tune and evaluate KNN classifiers on further datasets.

The notebook reproduces every listing of the chapter and the official solutions of the three exercises. It adds a theoretical deep-dive for every recipe, and experiments that test the book's claims and quantify what the book shows on single splits.

## Recipes

| # | Recipe | Book code | What the notebook demonstrates |
|---|---|---|---|
| 1 | Introduction to distance metrics | — | The metric axioms, the Minkowski family and its unit balls, why $p < 1$ is not a metric, how $L_1$, $L_2$ and $L_\infty$ weigh one large difference, cosine, Hamming and Mahalanobis distances |
| 2 | Understanding KNNs | pp. 71–72 | The KNN rule and the Cover–Hart bound, decision boundaries for $k = 1, 15, 50$, choosing $k$ with repeated cross-validation, KNN regression |
| 3 | Distance metrics overview | pp. 75–78 | Why "minkowski" equals "euclidean" here, a fair comparison over 50 datasets, distance ties on the checkerboard grid, metrics on real data |
| 4 | Hyperparameter tuning in KNN | pp. 81–82 | Ties in the grid, randomized search, nested cross-validation and the optimism of `best_score_` |
| 5 | Evaluating KNN performance | pp. 84–87 | Every number of the classification report computed from the confusion matrix, reading the learning curve, KNN on imbalanced data with threshold moving |
| 6 | Practical exercises with KNN models | p. 90 (official solutions) | Scaling in exercise 1 judged by one split vs. cross-validation, radius-based and centroid-based neighbour classifiers |

## Highlights from the notebook

- **Both printed outputs reproduce exactly** on scikit-learn 1.9.1 (the book targets 1.5):
  - `Accuracy: 0.9333333333333333` for 3-NN on Iris (p. 72);
  - grid search best `{'metric': 'euclidean', 'n_neighbors': 9, 'weights': 'uniform'}` with a CV score of `0.9916666666666668` (p. 82).
- **How the metrics weigh differences:** ten coordinates differing by 1 versus one coordinate differing by 10:

  | Metric | Ten differences of 1 | One difference of 10 |
  |---|---|---|
  | Manhattan | 10 | 10 |
  | Euclidean | 3.16 | 10 |
  | Chebyshev | 1 | 10 |

  This is the precise sense in which $L_1$ is "less sensitive to outliers".
- **"minkowski" is Euclidean** in the book's comparison (default `p=2`), so its column always matches the Euclidean one.
- **No metric wins on the noisy circles:** over 50 random datasets, Euclidean and Manhattan both average 0.823 accuracy, and Manhattan wins 25 times, loses 22 and ties 3. The book's single comparison measured luck.
- **The checkerboard is decided by ties:**
  - for about half of the test points, the 3rd and 4th neighbours are equally far;
  - simply shuffling the order of the same training rows moves the test accuracy from 0.74 to 0.85 (Euclidean), more than the gap between the metrics.
- **Choosing $k$ on Iris:** repeated cross-validation shows a broad plateau (0.96–0.97 for $k$ = 3 to 15), and a decline as $k$ approaches the class size (0.88 at $k$ = 75).
- **Grid search on a small dataset:** the winner ties with another candidate, and the runners-up are one misclassified flower behind. On all 150 flowers, nested cross-validation estimates the tuned model at 0.98.
- **Imbalanced data** (5% positives):

  | Model | Accuracy | Minority recall |
  |---|---|---|
  | always predict the majority | 0.950 | 0 |
  | KNN, default threshold | 0.961 | 0.22 |
  | KNN, vote threshold 0.2 | 0.962 | 0.52 |

  Lowering the vote threshold from 0.5 to 0.2 also raises the balanced accuracy from 0.61 to 0.75.
- **One split can mislead (exercise 1):** on the solution's split, the unscaled model scores higher (0.956 vs. 0.947), but 5-fold cross-validation shows that scaling helps by more than three points (0.965 vs. 0.932).

## Notes on the book vs. the current library

- All listings and the three official exercise solutions run unchanged on scikit-learn 1.9.1. The evaluation recipe needs **seaborn** for its heatmap (added to `requirements.txt`).
- The notebook is deterministic, except the object id and random CSS id of the styled classification report.
- Figure 4.2 cannot be reproduced number for number, because `make_circles` has no `random_state`. The notebook fixes the global seed instead.
- The printed evaluation listing (p. 84) uses `cross_val_score`, `classification_report` and pandas without importing them; the notebook follows the official notebook, which imports them.
- Precisions to the chapter text:
  - The opening list mentions a recipe "Model implementation in scikit-learn" that has no section of its own.
  - `iris.data` and `iris.target` are attributes, not methods.
  - The first model sets `n_neighbors=3`; the default is 5.
  - Cross-validation averages over deterministic folds rather than "adding randomization".
  - The checkerboard listing overwrites the iris labels `y` with a grid axis.
  - Re-running `cross_val_score` on the grid search's folds repeats `best_score_` rather than adding new evidence.
- Every dataset ships with scikit-learn or is generated with a fixed seed, so the notebook runs offline.
