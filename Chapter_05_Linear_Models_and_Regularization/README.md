# Chapter 5 — Linear Models and Regularization

**Notebook:** [`Chapter_05_Linear_Models_and_Regularization.ipynb`](Chapter_05_Linear_Models_and_Regularization.ipynb)
<a href="https://colab.research.google.com/github/fxrdhan/ML_scikit-learn_Cookbook_O-Reilly/blob/main/Chapter_05_Linear_Models_and_Regularization/Chapter_05_Linear_Models_and_Regularization.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Summary

Chapter 5 starts from **ordinary least squares** — the straight line of school mathematics — and builds up the toolbox that makes linear models robust and flexible:

- **ridge regression** adds an $L_2$ penalty that shrinks every coefficient;
- **Lasso** adds an $L_1$ penalty that also sets some coefficients exactly to zero (feature selection);
- **ElasticNet** mixes both penalties, and its effect is shown in a coefficient path plot.

A conceptual recipe explains regularization as a balance between model complexity and fit, through the bias–variance trade-off. The last recipe bends linear models around non-linear data with **polynomial features** and **cubic splines**. Three exercises then apply ridge, Lasso and ElasticNet to small synthetic datasets.

The notebook reproduces every listing of the chapter and the official solutions of the three exercises. It adds a theoretical deep-dive for every recipe, and experiments that diagnose the book's results, including one result that changes with scikit-learn 1.9.

## Recipes

| # | Recipe | Book code | What the notebook demonstrates |
|---|---|---|---|
| 1 | Introduction to linear models | pp. 94–98 | OLS by the normal equations, why the book's $R^2$ is near zero (the target was generated before most informative columns were overwritten), variance inflation and unstable coefficients |
| 2 | Ridge and Lasso regression | pp. 100–103 | scikit-learn's exact objectives (Lasso's $1/(2n)$ factor), shrinkage versus soft-thresholding, `RidgeCV` and `LassoCV` with standardized features |
| 3 | ElasticNet and regularization | pp. 105–108 | The ElasticNet objective in the book's notation, a readable coefficient path plot on the diabetes data, the grouping effect on correlated features |
| 4 | Regularization theory and practice | — | Why the $L_1$ constraint produces zeros (geometry), the bias–variance decomposition of ridge measured over 200 training sets, the Bayesian view |
| 5 | Regression and regularization (polynomials and splines) | pp. 112–119 | A scikit-learn 1.9 numerical cut-off that breaks the degree-5 fit, and choosing the degree and number of knots by cross-validation |
| 6 | Practical exercises with regularization techniques | pp. 120–122 (official solutions) | Why a straight ridge line cannot fit a parabola, Lasso's shrinkage of a slope by formula, the four models compared on the same data |

## Highlights from the notebook

- **Why recipe 1's model explains almost nothing:**
  - `make_regression` placed 7 of its 10 informative features in columns 50–99, which the "multicollinearity" loop then overwrote with copies of columns 0–49. About 77% of the signal's variance became unexplainable.
  - OLS on the original columns reaches $R^2 = 0.99$; on the book's columns $R^2 \approx 0$.
  - The added "non-linear" terms are more than two thousand times smaller than the target's spread.
- **Multicollinearity measured:** `feature_0` and its copy `feature_50` correlate at 0.995 (VIF 121). Their bootstrap coefficients swing by about ±69,000 each, while their sum varies about eight times less.
- **Tuned regularization** (standardized features), test $R^2$:

  | Model | Test $R^2$ |
  |---|---|
  | OLS | 0.001 |
  | ridge, `alpha` by cross-validation | 0.143 |
  | Lasso, `alpha` by cross-validation (keeps 13 of 100 columns, including the 3 surviving informative ones) | 0.209 |

  The book's `ElasticNet(alpha=1, l1_ratio=0.5)` also reaches 0.14, confirming the book's claim that it beats the other three models.
- **The grouping effect:** for two near-identical features that matter equally, OLS and Lasso split the weight unevenly (for example 1.15 / 0.62), while ridge and ElasticNet split it evenly (0.89 / 0.89).
- **Bias and variance of ridge** (degree-12 polynomial, 25 points, 200 training sets): the expected test error falls from 20.3 (no penalty) to 0.18 at `alpha` = 0.1, then rises again as bias takes over.
- **A version difference** — the degree-5 polynomial:
  - scikit-learn 1.9 uses `LinearRegression`'s `tol` (default `1e-6`) as the singular-value cut-off on dense data. The raw powers of $x \in [-50, 50]$ (scales up to $3 \times 10^8$) then lose a direction: rank 4 instead of 5, training $R^2$ 0.70.
  - Standardizing $x$ first (or `tol=1e-10`) restores test $R^2 = 0.98$, so degree 5 is the best of the book's five, as the book found.
- **Choosing the complexity by cross-validation:**
  - polynomials need about degree 7–9 to follow the two sine waves;
  - cubic splines are best with about 10 knots, and their training MSE keeps falling to 80 knots while the CV error rises.
- **The exercises:**
  - a straight ridge line on a parabola scores test $R^2 = -0.04$, while ridge on $[x, x^2]$ scores 0.89;
  - Lasso shrinks the slope by exactly $\alpha / \operatorname{Var}(x)$ (2.86 to 2.45).

## Notes on the book vs. the current library

- All listings and the three official exercise solutions run on scikit-learn 1.9.1. The degree-5 polynomial result differs from the book because of the `tol` change described above; the notebook diagnoses it and shows the fix.
- The first dataset adds unseeded noise, so its numbers differ between runs (the notebook fixes the global seed).
- Several listings fit on arrays and predict from DataFrames (or the reverse). The resulting feature-name warnings are harmless and are silenced in the setup cell.
- The official exercise notebook stores outputs that its published code does not reproduce (for example a test $R^2$ of 0.458 instead of −0.036 in exercise 1). The stored outputs presumably come from earlier versions of the cells.
- Precisions to the chapter text:
  - scikit-learn's Lasso and ElasticNet losses carry a $1/(2n)$ factor that the book's formulas omit.
  - Ridge never sets coefficients exactly to zero, and "encouraging sparsity" is specific to $L_1$.
  - The path-plot listing uses the line styles `'solid'` and `'-'`, which are identical, so two `l1_ratio` values look the same.
  - The non-linear dataset of recipe 5 is not built with `make_regression()` and has no exponential term.
  - The "spline interpolation" is spline regression, and its metrics are measured on the training data.
- Recipe 5 uses **seaborn** for one plot (listed in `requirements.txt`). Every dataset is generated with fixed seeds or ships with scikit-learn, so the notebook runs offline.
