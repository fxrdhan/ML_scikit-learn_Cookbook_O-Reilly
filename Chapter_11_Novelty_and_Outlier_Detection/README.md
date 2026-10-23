# Chapter 11 — Novelty and Outlier Detection

**Notebook:** [`Chapter_11_Novelty_and_Outlier_Detection.ipynb`](Chapter_11_Novelty_and_Outlier_Detection.ipynb)
<a href="https://colab.research.google.com/github/fxrdhan/ML_scikit-learn_Cookbook_O-Reilly/blob/main/Chapter_11_Novelty_and_Outlier_Detection/Chapter_11_Novelty_and_Outlier_Detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Summary

Chapter 11 is about finding the observations that do not fit: a faulty sensor, a fraudulent transaction, a machine about to fail. It covers:

- the difference between **outliers** (abnormal points that are in the training data) and **novelties** (abnormal points that appear after training), and the `contamination` parameter;
- **Isolation Forest** — points that are easy to isolate by random cuts are anomalies;
- the **One-Class SVM** — a smooth boundary learned around the normal data (novelty detection);
- the **Local Outlier Factor (LOF)** — a point is abnormal when its neighbourhood is sparser than the neighbourhoods of its neighbours;
- **evaluation** with the confusion matrix, precision, recall, F1 and the ROC-AUC;
- **handling** the detected points: removal, replacement, capping (winsorization) and flagging;
- **choosing** a technique from the density, the size and the dimension of the data.

Three exercises then use synthetic transaction, network and sensor data.

The notebook reproduces every listing of the chapter and the official solutions of the three exercises. It adds a theoretical deep-dive for every recipe (the isolation score, the nu property of the One-Class SVM, the LOF definitions, ROC and precision-recall curves, winsorization and robust statistics, the cost of each method), and experiments that test the book's explanations.

## Recipes

| # | Recipe | Book code | What the notebook demonstrates |
|---|---|---|---|
| 1 | Introduction to outlier and novelty detection | pp. 280–284 | Which planted outliers can be detected at all, the quota set by `contamination`, the API of outlier and novelty detectors, an anomaly inside every marginal range, masking |
| 2 | Understanding Isolation Forest | pp. 284–288 | The score rebuilt from the forest's own trees, path lengths, `contamination` as a threshold, number of trees, invariance to scaling, a score map |
| 3 | One-Class SVM for novelty detection | pp. 289–293 | Why the book's result looks wrong, the meaning of `nu` and `gamma`, overfitting of the boundary, feature scaling |
| 4 | Detecting outliers with LOF | pp. 294–297 | LOF by hand, the choice of `n_neighbors`, local against global outliers, `novelty=True` |
| 5 | Evaluating outlier detection models | pp. 298–302 | The sign of the ROC-AUC, the accuracy trap, ROC and precision-recall curves, the ceiling set by the data, run-to-run variation, label-free stability |
| 6 | Handling detected outliers | pp. 303–306 | What winsorizing really changes, each strategy tested on a regression, errors against genuine extremes, handling inside a `Pipeline` |
| 7 | Choosing the right detection technique | pp. 306–309 | The book's comparison table tested row by row: dimension, size, density, clean against contaminated training data, combining detectors |
| 8 | Practical exercises in novelty and outlier detection | pp. 309–310 (official solutions) | Invariance of Isolation Forest to scaling, LOF against Isolation Forest, the effect of `nu` and `gamma` |

## Highlights from the notebook

- **The printed output of recipe 5 reproduces exactly:** a confusion matrix of 297, 3, 3 and 27, precision, recall and F1 of 0.900 for the outliers, an accuracy of 0.982 and a ROC-AUC of **0.0051**. That last number is a mistake of orientation: `decision_function` is higher for inliers, so the score must be negated. With `-decision_function` the AUC is **0.9949**.
- **The One-Class SVM recipe is not a tuning problem.** The "normal" test points are generated around a different centre than the training data (5.6 standard deviations away), so the model is right to reject all 100 of them. On a test set drawn from the training distribution it flags 6 of 100, and the ROC-AUC goes from 0.596 to 0.977. The figure's legend also uses red for inliers, the opposite of the other figures.
- **Isolation Forest:**
  - the score of `score_samples` is rebuilt from the leaf depths of the fitted trees to $10^{-16}$;
  - the point closest to the centre needs 16.7 random cuts to be isolated, the farthest planted outlier 4.3;
  - standardizing the data changes nothing (identical scores);
  - it has no `feature_importances_`, contrary to the book;
  - its score map is made of axis-aligned bands, and a point in an empty corner ranks 11th for it against 1st for LOF.
- **`contamination` is a quota.** On 300 clean points, `contamination=0.05` flags exactly 15; `'auto'` flags 47 for Isolation Forest.
- **One-Class SVM:** `nu` bounds the flagged share and the share of support vectors, and the sum of the dual coefficients is exactly $\nu n$. A large `gamma` memorizes the training data: with `gamma=30` the decision values of the training points collapse to zero and 66% of fresh normal points are flagged. `gamma='scale'` divides by the variance of all entries, so it depends on where the data sits and on the units: with one feature multiplied by 20, a shift of 4 standard deviations along the other feature is found in 5% of the cases, against 96% after standardizing.
- **LOF:**
  - by hand it matches scikit-learn to $10^{-9}$;
  - it is scale-free (a LOF of about 1.04 in a dense and in a loose cluster whose densities differ by a factor 3);
  - with `novelty=True` it does predict on new data, contrary to the book;
  - on the book's own data Isolation Forest is better (ROC-AUC 0.840 against 0.767), but on data with local outliers LOF reaches 0.998 against 0.907.
- **A ceiling set by the data:** five of the twenty planted points of recipe 1 fall inside the normal cloud and cannot be found by any detector. In recipe 5 Isolation Forest reaches an AUC of 0.9949 against 0.9942 for an oracle that knows the generator. Over 50 new draws, the F1 is 0.910 with a standard deviation of 0.038 (from 0.80 to 1.00), so one run says little.
- **Handling outliers:**
  - the book's 5th/95th percentile winsorization changes 19% of the rows even when there is no outlier at all;
  - in a regression with corrupted rows, removal cuts the coefficient error from 2.77 to 0.28, but with genuine extreme points it makes it about 1.6 times larger, and winsorizing only the features is by far the worst choice with genuine extremes (1.85 against 0.03 for doing nothing);
  - a transformer cannot drop rows inside a `Pipeline`.
- **The comparison table of recipe 7, tested:**
  - with only 2 informative features out of 100, Isolation Forest falls to an AUC of 0.61 against 0.82 for LOF, but when every feature is informative both reach 1.000;
  - the time of the One-Class SVM is multiplied by 3 or more each time the number of samples doubles, while Isolation Forest hardly changes, and LOF in 2 features is the fastest of all;
  - LOF wins on local outliers (0.990 against 0.923), and a rank average of the two lies in between.
- **Exercises:** the `StandardScaler` of Exercise 1 does not change the Isolation Forest predictions; on Exercise 2 LOF and Isolation Forest are tied (0.970 and 0.965); in Exercise 3 `gamma='scale'` is 1.86 against 0.307 in recipe 3, because the data is centred.

## Notes on the book vs. the current library

- All listings and the three official exercise solutions run on scikit-learn 1.9.1.
- Precisions to the chapter text:
  - `roc_auc_score(y_true, model.decision_function(X))` needs `-model.decision_function(X)`: the book prints 0.0051;
  - the "normal" test points of recipe 3 come from another centre than the training data, and its figure swaps the colours of inliers and novelties;
  - LOF "cannot predict on unseen data" only in the default mode; `novelty=True` enables `predict`, `decision_function` and `score_samples`;
  - `IsolationForest` has no `feature_importances_`;
  - for Isolation Forest, scaling the features changes nothing;
  - the comparison table of recipe 7 describes tendencies that depend on the data (the notebook measures them);
  - winsorizing at the 5th and 95th percentiles changes about 10% of the values of every feature, including normal ones;
  - several listings call `np.random.uniform` without a seed, so their numbers depend on the order in which the cells run.
- The notebook is deterministic, except one cell that reports measured fit times. The extras never touch NumPy's global random generator, so they cannot change the book's numbers.
- Every dataset is generated by NumPy and scikit-learn, so the notebook runs offline, in about two and a half minutes.
