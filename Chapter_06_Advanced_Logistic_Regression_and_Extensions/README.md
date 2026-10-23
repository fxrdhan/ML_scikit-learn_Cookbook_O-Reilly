# Chapter 6 — Advanced Logistic Regression and Extensions

**Notebook:** [`Chapter_06_Advanced_Logistic_Regression_and_Extensions.ipynb`](Chapter_06_Advanced_Logistic_Regression_and_Extensions.ipynb)
<a href="https://colab.research.google.com/github/fxrdhan/ML_scikit-learn_Cookbook_O-Reilly/blob/main/Chapter_06_Advanced_Logistic_Regression_and_Extensions/Chapter_06_Advanced_Logistic_Regression_and_Extensions.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Summary

Chapter 6 uses regression to predict **classes**. Logistic regression passes a linear combination of the features through the sigmoid function to estimate the probability of a class, and is fitted by maximum likelihood. The recipes:

- fit a binary model on the Breast Cancer data and read its confusion matrix and classification report;
- extend it to three classes (Iris) with **one-vs-rest** and **multinomial (softmax)** logistic regression;
- control overfitting with **$L_2$ (ridge) and $L_1$ (Lasso) penalties** and the inverse regularization strength `C`;
- handle **multilabel** data, where a sample can carry several labels, and visualize how the labels co-occur;
- evaluate classifiers with **precision, recall, F1, the ROC curve and its AUC**.

Three exercises then build a regularized model, a multinomial model and a two-dimensional decision boundary.

The notebook reproduces every listing of the chapter and the official solutions of the three exercises. It adds a theoretical deep-dive for every recipe, and experiments that check the book's claims, including two listings that fail on scikit-learn 1.9.

## Recipes

| # | Recipe | Book code | What the notebook demonstrates |
|---|---|---|---|
| 1 | Overview of logistic regression | pp. 125–128 | The sigmoid and the log-odds checked on the fitted model, the log-loss minimized by Newton's method by hand, odds ratios, why `lbfgs` needs `max_iter=10000` on raw features |
| 2 | Multiclass classification techniques | pp. 130–135 | The replacements for two book lines that fail on scikit-learn 1.9, the softmax by hand, decision boundaries, OvR versus multinomial over 100 folds, what the pairplot really shows |
| 3 | Regularization in logistic regression | pp. 136–142 | `penalty=` versus the new `l1_ratio`, which features Lasso keeps, readable coefficient paths, `C` tuned by cross-validation |
| 4 | Multilabel classification concepts | pp. 143–149 | How the dataset is generated, subset accuracy, Hamming loss and the effect of `zero_division`, label dependence and classifier chains |
| 5 | Model evaluation metrics | pp. 150–154 | The metrics with *malignant* as the positive class, the AUC as a ranking probability, choosing a threshold, the spread over 200 random splits |
| 6 | Practical exercises with advanced logistic regression | p. 155 (official solutions) | Why the unscaled Lasso keeps the area and perimeter features, the probability surface behind exercise 3's boundary |

## Highlights from the notebook

- **Every printed output reproduces** on scikit-learn 1.9.1:
  - accuracy 0.95 in recipe 1;
  - the label statistics of recipe 4;
  - accuracy 0.94, precision 0.94, recall 0.95, F1 0.95 and ROC AUC 0.99 in recipe 5;
  - 0.95 and 0.93 in the exercises.
- **Logistic regression demystified:** eleven Newton steps on the penalized log-loss reproduce scikit-learn's coefficients to within $10^{-6}$. On the raw features `lbfgs` needs 1,894 iterations, against 19 on standardized ones. The standardized model is also more accurate (0.982 against 0.947), because the default $L_2$ penalty is only fair on a common scale.
- **OvR versus multinomial on Iris:**
  - on the book's split OvR wins by one flower (43 against 42 of 45);
  - over 100 cross-validation folds the multinomial model is slightly ahead (0.964 against 0.958);
  - the multinomial model's log-loss is about half of OvR's (0.17 against 0.32).
- **The pairplot** shows separation, not kurtosis. Over all flowers the petal features have a *negative* excess kurtosis (−1.40 and −1.34, a bimodal distribution). What makes them valuable is an ANOVA $F$ of 1,180 and 960, against 119 and 49 for the sepal features.
- **Regularization** (`C = 0.1`, standardized features):

  | Model | Accuracy | Malignant tumours missed | Benign tumours flagged | Non-zero coefficients |
  |---|---|---|---|---|
  | ridge ($L_2$) | 0.982 | 2 | 1 | 30 |
  | Lasso ($L_1$) | 0.965 | 1 | 5 | 7 |

  Lasso keeps one of the six nearly identical radius, perimeter and area features. Along its path, *worst perimeter* enters first and later gives way to *worst radius*.
- **Multilabel:**
  - the book's `accuracy` is a subset accuracy of only 0.34, and the Hamming loss is 0.21;
  - `zero_division=1` raises the samples-averaged recall from 0.61 to 0.78;
  - classifier chains raise the subset accuracy to 0.41–0.42 by using the dependence between labels.
- **Which class is positive matters:** in recipe 5, the printed precision and recall refer to the *benign* class. For the malignant class the recall is 0.910: 6 of 67 cancers are missed.
  - Lowering the threshold on $P(\text{malignant})$ to 0.2, chosen with cross-validation on the training set, catches 65 of the 67 at the cost of 9 false alarms instead of 5.
  - Over 200 random splits, the same model averages an accuracy of 0.949 (standard deviation 0.015).
- **Exercise 1 without scaling** lets the $L_1$ penalty prefer large-unit features: 5 of the 7 kept features are areas or perimeters. Ten-fold cross-validation ranks the variants: Lasso raw 0.930, ridge raw 0.944, Lasso standardized 0.972, ridge standardized 0.981.

## Notes on the book vs. the current library

- **Two listings fail on scikit-learn 1.9** (recipe 2):
  - `LogisticRegression(solver='liblinear')` on three classes raises a `ValueError` since 1.8. It is replaced by `OneVsRestClassifier(LogisticRegression(solver='liblinear'))`, which is what the old internal one-vs-rest did.
  - `multi_class='multinomial'` raises a `TypeError` since 1.8. It is dropped, as in the official notebook; the multinomial model is the default.
- **`penalty='l1'` / `penalty='l2'`** (recipe 3 and exercise 1) still work but are deprecated since 1.8 and removed in 1.10. The replacement is `l1_ratio=1` / `l1_ratio=0`, and the notebook checks that the models are identical. The two deprecation warnings are silenced in the setup cell.
- The styled classification reports carry random HTML ids, so their raw output changes from run to run; every number is reproducible.
- Small fixes to the listings:
  - imports that the printed code omits (`plt`, `sns`, and `accuracy_score` in the exercise solutions);
  - the official version of the co-occurrence function, because the printed one misses a comma.
- Precisions to the chapter text:
  - The exponent of the logistic formula should end in $\beta_n X_n$.
  - In recipe 2:
    - the OvR "confusion matrix" figure is a classification report;
    - "macro accuracy" means the macro average;
    - Iris has three classes.
  - In recipe 3, "ridge outperforms in precision while Lasso outperforms in recall" holds for the malignant class only.
  - In recipe 4:
    - the relative co-occurrence matrix gives $P(j \mid i)$ in row $i$;
    - "binary relevance" is the strategy the recipe already uses.
  - The chapter's list of recipes names an "Implementation in scikit-learn" recipe that has no section of its own.
- Every dataset ships with scikit-learn or is generated with a fixed seed, so the notebook runs offline. The book's code uses **seaborn** for its heatmaps and the pairplot (listed in `requirements.txt`).
