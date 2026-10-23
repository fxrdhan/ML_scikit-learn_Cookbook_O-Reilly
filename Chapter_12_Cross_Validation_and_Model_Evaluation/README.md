# Chapter 12 — Cross-Validation and Model Evaluation Techniques

**Notebook:** [`Chapter_12_Cross_Validation_and_Model_Evaluation.ipynb`](Chapter_12_Cross_Validation_and_Model_Evaluation.ipynb)
<a href="https://colab.research.google.com/github/fxrdhan/ML_scikit-learn_Cookbook_O-Reilly/blob/main/Chapter_12_Cross_Validation_and_Model_Evaluation/Chapter_12_Cross_Validation_and_Model_Evaluation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Summary

Chapter 12 is about estimating how a model will perform on data it has not seen, and about using that estimate to choose between models without fooling yourself. It covers:

- **k-fold and stratified k-fold cross-validation**: every observation is tested once and used for training $k-1$ times;
- **leave-one-out** and **nested** cross-validation: the first for very small samples, the second to report a performance that was not also optimized;
- **running** cross-validation with `cross_val_score`, `cross_validate` and `GridSearchCV`, and reading precision, recall and F1 next to the accuracy;
- **model selection** with grid search and randomized search (log-uniform distributions);
- **generalization diagnostics**: learning curves (score against the size of the training set) and validation curves (score against one hyperparameter).

Three exercises then repeat the workflow on new data.

The notebook reproduces every listing of the chapter and the official solutions of the three exercises. It adds a theoretical deep-dive for every recipe (the bias and variance of the k-fold estimate, the closed form of leave-one-out, selection bias and nested cross-validation, the splitters for groups and for time, the scoring API and the metrics, random search, the winner's curse, the bias-variance decomposition), and experiments that test the book's explanations.

## Recipes

| # | Recipe | Book code | What the notebook demonstrates |
|---|---|---|---|
| 1 | Introduction to cross-validation | pp. 314–316 | The book's two accuracies against 200 shuffles, the error bar of a CV estimate, the bias and variance as a function of $k$ against a known truth, when stratification matters |
| 2 | Advanced cross-validation methods | pp. 316–318 | Leave-one-out by the PRESS formula in one fit, `RidgeCV`, the selection bias that nested CV removes, the anatomy of each splitter, leakage through groups and through time |
| 3 | Implementing cross-validation in scikit-learn | pp. 319–321 | What `cv=5` really builds, baselines, nine metrics, the dictionary of `cross_validate`, `cross_val_predict`, leakage through preprocessing |
| 4 | Model selection techniques | pp. 321–324 | The number of fits of each search, `cv_results_` and the one-standard-error rule, the metric that picks the winner, random against grid under a budget, the winner's curse |
| 5 | Evaluating model generalizability | pp. 324–328 | The numbers behind both figures, the effect of `shuffle`, curves on more data, high bias against high variance, a bias-variance simulation, a shift of the data that CV cannot see |
| 6 | Practical exercises in cross-validation and evaluation | pp. 328–329 (official solutions) | Repeated CV, nested CV of a whole grid search, the error bar of a test accuracy, learning curves of two models |

## Highlights from the notebook

- **The book's numbers reproduce exactly:** 0.756 for `KFold` and 0.766 for `StratifiedKFold`, an LOOCV mean squared error of 3327.655 and a nested score of 0.480, the metrics of recipe 3, the best `C` of both searches (0.1 and 0.1744) with a test accuracy of 0.940, and both curves of recipe 5.
- **The one-point gap of recipe 1 is noise.** The classes are balanced, so stratification has nothing to fix. Over 200 shuffles the two splitters have the same mean (0.7700 and 0.7709) and a spread of about 0.006 from one shuffle to the next; the stratified score is higher for 52.5% of the seeds. A repeated stratified 5-fold has a spread about three times smaller.
- **The error bar is wider than the fold-to-fold standard deviation suggests.** The five fold accuracies of the book have a standard deviation of 0.071 for `KFold` and 0.013 for `StratifiedKFold` on the same model and data. The binomial standard error of an accuracy of 0.756 on 500 points is 0.019.
- **How many folds?** Against a known truth, $k=2$ is the only pessimistic value (about 1.2 points below, two standard errors), and the error is smallest at $k=5$ (RMSE 0.041); leave-one-out takes 100 fits instead of 5 on 100 points and is no more accurate (0.0425). The experiment uses one model and one sample size.
- **Stratification matters when a class is rare.** On 100 points with 10 positives, a plain k-fold leaves 9.9% of the test folds without any positive, and 45% of the shuffles contain such a fold.
- **Leave-one-out without the loop.** The PRESS formula gives the 442 fits of the book in one fit (3327.655), and `RidgeCV` shows that the book's `alpha=1.0` is not the best value (the error at `alpha=0.01` is 10% lower).
- **Selection bias, measured.** With labels that carry no information, the `best_score_` of a 25-candidate grid is 0.608 on average (11 points above chance), the nested score is 0.525 and the tuned model scores 0.498 on fresh data.
- **Structure of the data.** With 20 rows per subject, a shuffled `KFold` reports 0.973 where the honest value is 0.528 (`GroupKFold`). On a time series, a shuffled `KFold` gives an $R^2$ of 0.934 against $-0.113$ for `TimeSeriesSplit`.
- **Leakage.** Selecting 20 of 1,000 noise features before cross-validating gives an accuracy of 0.817 on random labels; inside a `Pipeline` it is 0.519. Standardizing outside the CV barely matters (0.925 against 0.930).
- **Recipe 3:** `cv=5` with a classifier already builds a `StratifiedKFold` (30 positives in each test fold), contrary to the text. A constant predictor reaches an accuracy of 0.700 on the 70/30 data, and the best `C` depends on the metric (0.1 for accuracy, 0.001 for the ROC AUC).
- **Recipe 4:**
  - the randomized search of the book makes 50 fits, the grid 30;
  - the grid's winner classifies 140 of the 150 test samples correctly and the random winner 141, so the "better" search is one sample ahead;
  - with 36 grid candidates against 12 random ones (a third of the budget), the random search is 0.6 points below the grid in test accuracy on average, a difference within the noise (standard error 0.6 points; 7 wins, 2 ties and 6 losses over 15 data sets);
  - on 100 training samples the `best_score_` is 0.032 too optimistic on average.
- **Recipe 5:**
  - the learning curve cannot go beyond 800 samples (4/5 of 1,000), where the gap is already 0.006;
  - `random_state` is ignored unless `shuffle=True`;
  - the validation curve has no overfitting at large `C`, only a plateau (0.892);
  - on 6,000 samples the validation score levels off at 0.946 from 1,600 samples on;
  - a logistic regression on the moons data is high bias (gap $-0.001$), an unpruned tree high variance (gap 0.159);
  - the bias-variance simulation has its minimum at degree 3 (expected error 0.111);
  - when four features shift by 2 standard deviations, the accuracy falls from 0.850 to 0.658 while the CV estimate stays at 0.843.
- **Exercises:** the 0.896 of Exercise 1 could as well have been 0.88 or 0.90 (standard deviation 0.0063 over repeats); in Exercise 2 three grid candidates tie for the first rank and the nested CV gives 0.912; in Exercise 3 the SVC has a validation score one point higher than the logistic regression and a larger gap (0.040 against 0.012).

## Notes on the book vs. the current library

- All listings and the three official exercise solutions run on scikit-learn 1.9.1.
- Precisions to the chapter text:
  - `cross_val_score(model, X, y, cv=5)` with a classifier is stratified: the book's remark that recipe 3 does not use stratification does not hold;
  - the difference between 0.756 and 0.766 in recipe 1 comes from the random partition drawn by each splitter, not from stratification;
  - the target of `load_diabetes()` is quantitative (25 to 346), the "Nested CV mean score: 0.480" is an $R^2$ and not an MSE, and the "0.593 MSE" of the text appears nowhere in the code or its output;
  - the randomized search of recipe 4 uses more fits than its grid, and the text evaluates only the randomized winner on the test set;
  - `learning_curve(random_state=...)` has no effect unless `shuffle=True`;
  - the chapter introduction lists a "Model evaluation metrics overview" recipe that the book does not contain (the notebook covers it in the theory of recipe 3);
  - the solution of Exercise 3 reuses `X`, `y` and `LogisticRegression` from earlier cells.
- The notebook is deterministic, except one cell that reports measured fit times. Every extra uses its own random generators and passes `random_state` to every estimator that has one, so the extras cannot change the book's numbers.
- Every dataset is generated by NumPy and scikit-learn (or ships with it), so the notebook runs offline, in about two minutes.
