# Chapter 3 — Dimensionality Reduction Techniques

**Notebook:** [`Chapter_03_Dimensionality_Reduction_Techniques.ipynb`](Chapter_03_Dimensionality_Reduction_Techniques.ipynb)
<a href="https://colab.research.google.com/github/fxrdhan/ML_scikit-learn_Cookbook_O-Reilly/blob/main/Chapter_03_Dimensionality_Reduction_Techniques/Chapter_03_Dimensionality_Reduction_Techniques.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Summary

Chapter 3 is about choosing the right "ingredients" of a model: reducing the number of features while keeping the information that matters. It explains why this helps, and it separates **feature selection** (keeping a subset of the original columns) from **feature extraction** (building new ones). The reasons are:

- simpler data;
- lower computational cost;
- less overfitting;
- pictures of data that live in many dimensions.

The chapter then covers the three main extraction techniques in scikit-learn:

- **PCA** (`PCA`) projects the data onto the directions of maximum variance. It is unsupervised and needs standardized features.
- **LDA** (`LinearDiscriminantAnalysis`) uses the labels to find the directions that best separate the classes, with at most $C - 1$ components for $C$ classes.
- **t-SNE** (`TSNE`) draws high-dimensional data in two dimensions while preserving local neighbourhoods. It is a visualization tool.

The last recipes give guidelines for choosing between the techniques and discuss their trade-offs for model performance. Two exercises close the chapter: PCA before a logistic regression, and a t-SNE map of handwritten digits next to K-means clusters.

The notebook reproduces every listing of the chapter, the repository-only code of Figure 3.4, and the official solutions of both exercises. It adds a theoretical deep-dive for every recipe, and experiments that check the book's claims and put numbers on the ideas it explains in prose.

## Recipes

| # | Recipe | Book code | What the notebook demonstrates |
|---|---|---|---|
| 1 | Introduction to dimensionality reduction | — | The curse of dimensionality: as dimensions grow, distances concentrate and neighbourhoods stop being local |
| 2 | Transforming datasets with PCA | pp. 48–54 | PCA derived as an eigenproblem and checked against scikit-learn, a proper biplot, why scaling matters, how many components to keep, PCA as image compression |
| 3 | Maximizing class separability with LDA | pp. 55–59 | Fisher's criterion computed by hand, the $C - 1$ limit, why LDA does not need scaling, how LDA "separates" random labels on training data |
| 4 | t-SNE and data visualization | pp. 60–62 | The t-SNE objective (perplexity, Student-t map, KL divergence), three perplexities compared, trustworthiness of PCA, LDA and t-SNE maps, why t-SNE has no `transform` |
| 5 | Selecting the right technique | — | Cost and neighbourhood quality of each technique measured, PCA followed by t-SNE, other scikit-learn tools |
| 6 | Impact on model performance | — | Accuracy against the number of components, and a dataset where the class signal hides in a low-variance direction |
| 7 | Practical exercises in dimensionality reduction | p. 67 (official solutions) | Exercise 1 on Iris (as the text says) and on the digits (as the solution does) with cross-validation; the adjusted Rand index for exercise 2 |

## Highlights from the notebook

- **The book's printed value reproduces** on scikit-learn 1.9.1 (the book targets 1.5): the first two principal components of the standardized wine data explain **55.41%** of the variance (36.20% + 19.21%). The t-SNE map shows what the book describes: an isolated group of 0s in the bottom-left corner, and 9s near the 7s and 4s.
- **The curse of dimensionality, measured:** for 200 random points, the farthest pair is several hundred times farther apart than the nearest pair in 2 dimensions, but only 5% farther in 10,000 dimensions.
- **PCA by hand:** the eigenvalues of the covariance matrix and the squared singular values of the data (divided by $n - 1$) match `explained_variance_` exactly, and the eigenvectors match `components_` up to sign.
- **What the book's PCA arrows show:** they plot the weights of alcohol and malic acid on each component, not the components' directions. The notebook draws a full biplot instead.
- **Figure 3.4 is a rotation:** for two standardized features, PCA always turns the data by 45° (and possibly mirrors it). Every pairwise distance is unchanged, so class separation cannot improve.
- **Scaling matters for PCA, not for LDA:**
  - Without standardization, PC1 is essentially proline and explains 99.8% of the variance.
  - LDA gives the same projection with or without `StandardScaler`, up to round-off ($10^{-14}$).
- **Supervised projections can fool you:** with 50 noise features and 60 samples, LDA separates three *random* classes perfectly on its training data, but cross-validated accuracy is at chance (0.30).
- **Honest comparison on the wine data** (k-NN, projection fitted inside each fold):

  | Space | CV accuracy |
  |---|---|
  | LDA, 2 dimensions | 0.989 |
  | PCA, 2 dimensions | 0.966 |
  | all 13 standardized features | 0.961 |

- **Trustworthiness of 2-D maps of the digits** (share of map neighbours that are true neighbours):

  | Map | Trustworthiness |
  |---|---|
  | t-SNE | 0.986 |
  | PCA | 0.816 |
  | LDA | 0.765 |

  Running t-SNE on 30 principal components gives the same quality (0.986).
- **Accuracy against the number of components** (digits, logistic regression):
  - two components reach 0.54;
  - thirty reach 0.95;
  - all 61 non-constant directions reach 0.97.

  For k-NN, 30 components are slightly *better* than all of them.
- **Variance is not relevance:** when the class difference lies in a low-variance direction, PCA with one component gives 0.535 accuracy, while LDA with one discriminant gives 0.965.
- **The exercises:**
  - Exercise 1 on the digits: 0.975 without PCA, 0.964 with 39 components. On Iris, which the text names, PCA costs more: 0.95 against 0.91 in 10-fold cross-validation.
  - Exercise 2: K-means agrees with the true digits at an adjusted Rand index of 0.66 on raw pixels, 0.49 on standardized pixels, and 0.87 on the t-SNE map.

## Notes on the book vs. the current library

- All listings and both official exercise solutions run unchanged on scikit-learn 1.9.1. Since 1.5, `TSNE`'s `n_iter` is called `max_iter`, and `PCA` has a `"covariance_eigh"` solver; the book's code uses neither.
- The notebook is deterministic, except the cell that measures run times.
- Precisions to the chapter text:
  - The first two principal components capture more variance than any other pair of directions, but only 55% of the total on the wine data.
  - LDA's "common covariance matrix" assumption means that all classes share one covariance matrix, not that the features have similar variances. Standardizing does not change LDA's projection.
  - PCA components and LDA discriminants are both linear combinations of all features; neither is inherently more interpretable.
  - `load_digits` is the UCI optical digits dataset (1,797 images of $8 \times 8$ pixels), not MNIST.
  - t-SNE does not use the labels: it reveals neighbourhoods, which here coincide with the digits.
  - Exercise 1 is described on the Iris dataset but solved on the digits in the official notebook. The exercise text promises an analysis of "subsequent classification tasks" that the solution does not perform.
- Every dataset ships with scikit-learn (`load_wine`, `load_digits`, `load_iris`) or is generated with a fixed seed, so the notebook runs offline.
