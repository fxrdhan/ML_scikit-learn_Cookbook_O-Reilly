# Chapter 2 — Pre-Model Workflow and Data Preprocessing

**Notebook:** [`Chapter_02_Pre_Model_Workflow_and_Data_Preprocessing.ipynb`](Chapter_02_Pre_Model_Workflow_and_Data_Preprocessing.ipynb)
<a href="https://colab.research.google.com/github/fxrdhan/ML_scikit-learn_Cookbook_O-Reilly/blob/main/Chapter_02_Pre_Model_Workflow_and_Data_Preprocessing/Chapter_02_Pre_Model_Workflow_and_Data_Preprocessing.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Summary

Chapter 2 covers everything that happens to the data before a model sees it. It opens with "garbage in, garbage out": missing values, outliers, categorical variables, features on very different scales and data leakage limit any model, whatever the algorithm. Each of the following recipes then introduces one family of preprocessing tools:

- **imputing** missing values with `SimpleImputer` (mean/median/most frequent), `KNNImputer` (average of the nearest rows) and `IterativeImputer` (each feature regressed on the others);
- **scaling** numeric features with `StandardScaler` (z-scores), `MinMaxScaler` (range $[0, 1]$) and `Normalizer` (each row scaled to unit length);
- **encoding** categorical variables with `OneHotEncoder` and `LabelEncoder`, and handling mixed numeric/categorical data with `ColumnTransformer`.

The steps are then chained into a **`Pipeline`**. The data are split *before* any transformation, so that every learned statistic comes from the training data only, and the pipeline can be displayed as a diagram. The chapter ends with **feature engineering**: creating features with `PolynomialFeatures` and `KBinsDiscretizer`, and selecting them with recursive feature elimination (`RFE`) and `SelectFromModel`. A practical exercise combines everything in an imputer + scaler + random-forest pipeline on the California Housing data.

The notebook reproduces every listing of the chapter and the official solution of the exercise. It adds a theoretical deep-dive for every recipe, and runnable experiments on real and synthetic data that put numbers on what the book explains in prose.

## Recipes

| # | Recipe | Book code | What the notebook demonstrates |
|---|---|---|---|
| 1 | The impact of raw data on model performance | — | The bias–variance decomposition. Label noise and irrelevant features measured on real models. One outlier against least squares, z-score and Tukey (IQR) detection, and the Huber loss |
| 2 | Handling missing data | pp. 16–19 | MCAR / MAR / MNAR mechanisms, how each imputer computes its values, imputation accuracy against a known ground truth, variance shrinkage under mean imputation |
| 3 | Scaling techniques | pp. 22–24 | The formulas checked by hand, the column-wise vs. row-wise difference, robustness to an outlier (`RobustScaler`), which models need scaling and why trees do not |
| 4 | Encoding categorical variables | pp. 26–31 | Nominal vs. ordinal encoding (`OrdinalEncoder` with an explicit order), unseen categories, the dummy-variable trap, automatic feature names |
| 5 | Introduction to pipelines in scikit-learn | pp. 33–36 | What a pipeline guarantees, a hidden target leak in the book's example, the scaler chosen by `GridSearchCV` as a hyperparameter |
| 6 | Feature engineering | pp. 37–41 | Polynomial expansion and its column count, three binning strategies, filter / wrapper / embedded selection checked against a known ground truth (`RFECV`, Lasso, random forest) |
| 7 | Practical exercise on data preprocessing | p. 42 (official solution) | The California Housing data and regression metrics (R², RMSE, MAE), baselines, a scale-invariance check of the random forest, the capped target |

## Highlights from the notebook

- **The book's printed values reproduce** on scikit-learn 1.9.1 (the book targets 1.5): the mean 52.558605 imputed for `Feature1`, and the two features that RFE ranks first (`Position_Manager`, `Position_Junior`).
- **Garbage in, garbage out, measured:**
  - Flipping 45% of the training labels lowers the test accuracy of a logistic regression on the breast-cancer data from 0.959 to 0.661.
  - Appending 100 pure-noise features lowers k-NN cross-validated accuracy on the wine data from 0.961 to 0.820.
- **One outlier among 31 points** cuts the least-squares slope from 2.03 to 1.18. Both the z-score (−4.38) and Tukey's fences flag it. The Huber regressor recovers a slope of 2.01 without deleting anything.
- **Which imputer is most accurate?** With 20% of the values of the diabetes data hidden completely at random:

  | Imputer | RMSE on the hidden cells | Downstream CV R² (Ridge) |
  |---|---|---|
  | `SimpleImputer` (mean) | 0.0473 | 0.4264 |
  | `SimpleImputer` (median) | 0.0509 | 0.4234 |
  | `KNNImputer` (k = 5) | 0.0395 | 0.4392 |
  | `IterativeImputer` | 0.0362 | 0.4475 |
  | reference: complete data | 0 | 0.4896 |

  Mean imputation also shrinks the standard deviation of `s2` from 0.0476 to 0.0428, close to the theoretical 0.0426 (a factor of $\sqrt{0.8}$).
- **Scaling with an outlier:** `MinMaxScaler` squeezes 99 normal values into a band of width 0.052, while `RobustScaler` keeps them spread over 3.4 units.
- **Who needs scaling:** on the wine data, k-NN rises from 0.663 (unscaled) to 0.955–0.972 with any column-wise scaler. A decision tree scores exactly 0.927 with no scaling and with every column-wise scaler.
- **A hidden target leak:** in the book's pipeline example, the target `Position_Senior` equals `1 - Position_Junior - Position_Manager`, so a linear model fits it perfectly (training R² = 1.0). The same leak explains why `SelectFromModel` keeps exactly those two features.
- **Feature selection checked against the truth:** on synthetic data with four informative features out of twelve:
  - `RFE` and `RFECV` recover exactly the four.
  - Lasso-based `SelectFromModel` adds one false positive.
  - Random-forest importances miss one.
- **Polynomial features:** degree 2 on the book's 9 inputs gives $\binom{11}{2} = 55$ columns. On a curved relationship, adding $x^2$ lifts the CV R² of the same linear regression from 0.594 to 0.937. On 0/1 columns, squares and products of mutually exclusive dummies add nothing new.
- **The exercise in context:** the book's pipeline reaches R² = 0.803 on 4,128 held-out districts, a typical error (MAE) of about \$33,000. On the same split (errors in units of \$100,000):

  | Model | R² | RMSE | MAE |
  |---|---|---|---|
  | baseline: always predict the training mean | about 0 | 1.170 | 0.921 |
  | linear regression (imputer + scaler) | 0.599 | 0.741 | 0.535 |
  | random forest (book solution) | 0.803 | 0.520 | 0.334 |
  | `HistGradientBoostingRegressor` without any preprocessing | 0.835 | 0.475 | 0.312 |

- **Imputer and scaler change nothing the forest learns:** with the same seed, the forest grows exactly the same 50 trees with or without them (R² 0.8019 vs. 0.8021). Predictions still differ slightly for 1,404 of the 4,128 districts (by at most 0.168), because some new values land exactly on a split threshold, where floating-point rounding decides the branch.

## Notes on the book vs. the current library

- All listings run unchanged on scikit-learn 1.9.1. RFE ranks 3 to 9 differ from the printed ones because those coefficients are zero up to floating-point round-off (about $10^{-16}$).
- Precisions to the chapter text:
  - `KNNImputer` averages the neighbours' values; it does not take a "majority label".
  - `IterativeImputer` regresses each feature on all the others and, like `KNNImputer`, needs numeric input.
  - The "> 5 %" guideline for simple imputation presumably means "< 5 %".
  - Column-wise scaling is unnecessary for trees rather than detrimental.
  - `LabelEncoder` is meant for targets; `OrdinalEncoder` (with an explicit category order) is the tool for features.
  - The California data have 8 features plus the target, which is the median house value of a block group in units of \$100,000 (the book describes 9 features and an average price).
- The exercise downloads the California Housing data with `fetch_california_housing()` on the first run and caches it in `~/scikit_learn_data`. Every other experiment uses data bundled with scikit-learn or generated with fixed seeds.
