# Chapter 1 — Common Conventions and API Elements of scikit-learn

📓 **Notebook:** [`Chapter_01_Common_Conventions_and_API_Elements.ipynb`](Chapter_01_Common_Conventions_and_API_Elements.ipynb)
<a href="https://colab.research.google.com/github/fxrdhan/ML_scikit-learn_Cookbook_O-Reilly/blob/main/Chapter_01_Common_Conventions_and_API_Elements/Chapter_01_Common_Conventions_and_API_Elements.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Summary

Chapter 1 explains the conventions that every scikit-learn object follows, so that the rest of the library becomes predictable. **Estimators** learn from data with `fit()` and predict with `predict()`; **transformers** learn a transformation with `fit()` and apply it with `transform()` (`fit_transform()` does both, but only on training data); **pipelines** chain transformers and a final estimator into one object. Fitted models expose what they learned through attributes ending in `_` (`coef_`, `intercept_`) and evaluate themselves with `score()` (accuracy for classifiers, R² for regressors). Hyperparameters are plain constructor arguments managed with `get_params()`/`set_params()` and tuned with `GridSearchCV()`, `RandomizedSearchCV()` or successive-halving search. Finally, **metadata** — estimator tags and metadata routing — describes what each estimator can do and sends extra information such as sample weights or group labels to exactly the steps that need it.

The book's chapter is mostly conceptual (six short listings). The notebook reproduces every listing and adds a theoretical deep-dive and runnable experiments for every recipe, including the ones the book explains only in prose.

## Recipes

| # | Recipe | Book code | What the notebook demonstrates |
|---|---|---|---|
| 1 | Introduction to scikit-learn's design philosophy | — | The five API principles of Buitinck et al. (2013), the estimator/predictor/transformer interfaces, five algorithms trained with one identical loop |
| 2 | Understanding estimators | p. 3, p. 4 | OLS solved by hand ($\hat y = 0.95x + 0.05$), the K-means objective and Lloyd's algorithm, a tie between two optimal partitions in the book's K-means example |
| 3 | Transformers and the `transform()` method | p. 5, p. 6 | Z-score math, when scaling matters, the learned state of a transformer, why new data must never be re-fitted, Figure 1.1 re-drawn |
| 4 | Handling custom estimators and transformers | — | A custom `QuantileClipper` transformer and `MinkowskiCentroidClassifier` that pass `check_estimator()`; why mixins go before `BaseEstimator` |
| 5 | Pipelines and workflow automation | — | `ColumnTransformer` + `Pipeline` on mixed-type data, the wrong vs. right way to cross-validate, model persistence |
| 6 | Common attributes and methods | p. 9 | R² and adjusted R² derived by hand, a simulation of R² with pure-noise features |
| 7 | Hyperparameter tuning with search methods | p. 10 | `get_params`/`set_params`/`clone`, grid vs. random vs. successive-halving search on the breast-cancer dataset |
| 8 | Working with metadata: Tags and more | — | Estimator tags that change behaviour, routing balanced sample weights to the classifier only, routing groups to `GroupKFold` |
| 9 | Best practices for API usage | — | An end-to-end workflow that uses every convention of the chapter |

## Highlights from the notebook

- **All six book listings reproduce the printed outputs** on scikit-learn 1.9.1 (the book targets 1.5): predictions `[5.75 6.7]`, K-means labels `[0 0 0 1 1]`, the standardized matrix, `coef_ = [0.95]`, `intercept_ = 0.04999999999999938`, R² = 0.9809782608695652, and the full `get_params()` dictionary.
- **A hidden tie in the K-means example:** the partitions {1,2,3}|{4,5} and {1,2}|{3,4,5} both have inertia 2.5, so the printed labels depend on the random initialization.
- **Custom estimators done right:** both custom classes pass scikit-learn's API test suite (45/46 and 54/55 checks passed, one optional Array-API check skipped) and work inside `Pipeline` and `GridSearchCV`.
- **Data leakage made visible:** selecting features from 5,000 pure-noise columns *before* cross-validation reports 90% accuracy on coin-flip labels; doing it inside a pipeline gives a realistic 57%.
- **R² vs. adjusted R²:** adding 40 noise features pushes training R² from 0.57 to 0.93, while adjusted R² stays near the true 0.57 and test R² drops to −2.22.
- **Search strategies:** successive halving found the same best random forest as an exhaustive grid search using about one third of the data budget.
- **Metadata routing:** sending balanced sample weights only to the classifier raised minority-class recall from 0.43 to 0.87.

## Notes on the book vs. the current library

- Estimator tags became a public dataclass API in scikit-learn 1.6 (`sklearn.utils.get_tags`, `__sklearn_tags__`), replacing the private `_get_tags()`/`_more_tags()` of 1.5; `validate_data()` also became public in 1.6.
- Small precisions to the text: `get_params()` reports the *current* configuration (not the hyperparameters "that provide the best fit"); in-sample R² *never decreases* when variables are added; mixins are implemented through multiple inheritance; the pipeline material announced for "Chapter 14" appears in Chapter 2 of this edition.
