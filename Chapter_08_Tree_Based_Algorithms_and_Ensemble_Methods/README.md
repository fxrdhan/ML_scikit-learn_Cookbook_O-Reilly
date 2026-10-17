# Chapter 8 — Tree-Based Algorithms and Ensemble Methods

**Notebook:** [`Chapter_08_Tree_Based_Algorithms_and_Ensemble_Methods.ipynb`](Chapter_08_Tree_Based_Algorithms_and_Ensemble_Methods.ipynb)
<a href="https://colab.research.google.com/github/fxrdhan/ML_scikit-learn_Cookbook_O-Reilly/blob/main/Chapter_08_Tree_Based_Algorithms_and_Ensemble_Methods/Chapter_08_Tree_Based_Algorithms_and_Ensemble_Methods.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Summary

Chapter 8 covers **decision trees**, which split the data recursively on single features to reduce impurity, and the **ensembles** built from them:

- **random forests and bagging** — many trees grown in parallel on bootstrap samples, then voted;
- **gradient boosting** — shallow trees added one after another, each correcting the current model;
- **stacking** — base models combined through a meta-model.

The recipes fit and plot a tree, read a forest's feature importances, fit a GBM, tune it with a grid search and a heatmap, and compare the three ensemble strategies. Three exercises then use the Wine, Breast Cancer and Digits datasets.

The notebook reproduces every listing of the chapter and the official solutions of the three exercises. It adds a theoretical deep-dive for every recipe (Gini impurity, greedy splitting, the variance of averaged trees, boosting as gradient descent, out-of-fold stacking), and experiments that test the book's explanations.

## Recipes

| # | Recipe | Book code | What the notebook demonstrates |
|---|---|---|---|
| 1 | Introduction to decision trees | pp. 188–190 | Gini impurity and the first split computed by hand, tree depth versus training, CV and test accuracy |
| 2 | Random forests and bagging | pp. 194–197 | The forest versus its own 100 trees, the out-of-bag estimate, impurity versus permutation importance, decision regions |
| 3 | Gradient boosting machines | pp. 198–200 | Test accuracy after each of the 1,000 stages, early stopping |
| 4 | Hyperparameter tuning for trees and ensembles | pp. 202–206 | Twenty tied candidates, what the heatmap averages over, a randomized search |
| 5 | Comparing ensemble methods | pp. 207–210 | Inside the stacking model, seven tree methods on three datasets with repeated cross-validation |
| 6 | Practical exercises with tree-based models | pp. 211–212 (official solutions) | A forest on the Wine data, why gradient boosting on Digits fits 1,000 trees |

## Highlights from the notebook

- **Every printed result reproduces** on scikit-learn 1.9.1:
  - 0.867 for the decision tree and 0.911 for the random forest;
  - the GBM equal to the tree;
  - 0.91 / 0.87 / 0.87 for random forest, gradient boosting and stacking.
- **Gini, by hand:** the root's impurity is 0.665, close to the three-class maximum $1 - 1/K = 0.667$, not 1.0 as the book suggests. The first split (`petal width <= 0.8`) lowers it by 0.318.
- **The forest versus its trees:** the 100 trees score 0.901 on average (0.80 to 0.98), their vote 0.911, and the out-of-bag estimate is 0.962.
  - Impurity importances split the credit between the correlated petal features (0.45 and 0.44).
  - Permutation importance shows that the forest relies mostly on petal width (0.45 against 0.22).
- **Why the GBM equals the tree:**
  - not overfitting: its test accuracy is already 0.867 after the first stage, while the training loss falls to zero;
  - both models miss exactly the same six overlapping flowers;
  - early stopping keeps 44 of the 1,000 stages with the same accuracy.
- **Tuning:** 20 of the 27 grid candidates tie at 0.981, so the winner is decided by the grid's order. The book's heatmap averages each cell over three learning rates. A randomized search reaches the same score.
- **Ensembles on three datasets** (repeated 5-fold CV):

  | Model | Iris | Wine | Breast Cancer |
  |---|---|---|---|
  | decision tree | 0.947 | 0.919 | 0.927 |
  | random forest | 0.950 | 0.978 | 0.962 |
  | extra trees | **0.953** | **0.986** | 0.967 |
  | hist gradient boosting | 0.950 | 0.977 | **0.968** |
  | stacking (RF + GB) | 0.947 | 0.966 | 0.966 |

  On Iris all methods tie; on the other datasets every ensemble clearly beats one tree, and stacking is never the best.
- **Exercise 3:** gradient boosting on Digits fits 1,000 trees (100 stages times 10 classes), which is why it is so much slower than the forest. `HistGradientBoostingClassifier` matches its accuracy (0.967) in about two seconds.

## Notes on the book vs. the current library

- All listings and the three official exercise solutions run unchanged on scikit-learn 1.9.1.
- The random forest, GBM and tuning listings use `classification_report` without importing it (it comes from recipe 1). The notebook runs them in order, as the book does.
- In the listing's `plot_tree` call, "samples" and "value" are counts. The text describes proportions, which the official notebook shows with `proportion=True`.
- The notebook is deterministic, except one cell that reports a measured fitting time, and the random HTML ids of the styled reports.
- Precisions to the chapter text:
  - Gini impurity is at most $1 - 1/K$;
  - the GBM's tie with the single tree is not overfitting;
  - the heatmap cells are averages over learning rates;
  - `RandomizedSearchCV` draws independent candidates rather than adjusting them between iterations;
  - the forest's lead on Iris is two flowers and disappears under cross-validation.
- Every dataset ships with scikit-learn, so the notebook runs offline (about three minutes). The tuning recipe uses **seaborn** for its heatmap.
