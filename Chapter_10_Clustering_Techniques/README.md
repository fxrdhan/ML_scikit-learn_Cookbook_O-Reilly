# Chapter 10 — Clustering Techniques

**Notebook:** [`Chapter_10_Clustering_Techniques.ipynb`](Chapter_10_Clustering_Techniques.ipynb)
<a href="https://colab.research.google.com/github/fxrdhan/ML_scikit-learn_Cookbook_O-Reilly/blob/main/Chapter_10_Clustering_Techniques/Chapter_10_Clustering_Techniques.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Summary

Chapter 10 turns to **unsupervised learning**: grouping data that has no labels. It covers:

- **K-means** — centroids, Lloyd's algorithm, and the **elbow method** for the number of clusters;
- **hierarchical (agglomerative) clustering** — nested clusters, linkages and the **dendrogram**;
- **DBSCAN** — clusters as dense regions, with noise points, found without choosing the number of clusters;
- **evaluation metrics** — the silhouette score, the Davies-Bouldin index and the adjusted Rand index;
- **choosing an algorithm** from the shape, noise, size and dimension of the data;
- **spectral clustering** and **Gaussian mixture models**, for curved clusters and for probabilistic assignments.

Three exercises then use the Iris, moons and Wine data.

The notebook reproduces every listing of the chapter and the official solutions of the three exercises. It adds a theoretical deep-dive for every recipe (Lloyd's algorithm and k-means++, linkages and Ward's cost, density-reachability, the three internal metrics and the ARI, the graph Laplacian, EM and the BIC), and experiments that test the book's explanations.

## Recipes

| # | Recipe | Book code | What the notebook demonstrates |
|---|---|---|---|
| 1 | Introduction to clustering | pp. 254–256 | The geometry of the blobs, why scaling matters, clusters found in pure noise |
| 2 | K-means clustering | pp. 256–260 | Lloyd's algorithm by hand, 100 seeds of two initializations, the elbow against the silhouette, where K-means breaks |
| 3 | Hierarchical clustering | pp. 260–263 | Ward's method by hand, reading the merge heights, four linkages on three shapes, bisecting K-means |
| 4 | Density-based clustering with DBSCAN | pp. 264–267 | Core, border and noise points, the k-distance plot decoded, the window of `eps`, varying density and HDBSCAN |
| 5 | Cluster evaluation metrics | pp. 268–270 | The metrics by hand, metrics against the number of clusters, internal metrics on curved clusters, chance correction |
| 6 | Choosing the right clustering algorithm | pp. 270–273 | Six algorithms on three shapes, fit time against the number of samples, irrelevant features and PCA |
| 7 | Advanced clustering techniques | pp. 274–277 | Spectral clustering and EM by hand, the affinity graph, BIC, soft assignments and their confidence |
| 8 | Practical exercises with clustering models | pp. 277–278 (official solutions) | Clusters against species, the window of `eps`, PCA and a mixture on Wine |

## Highlights from the notebook

- **The printed output of recipe 5 reproduces exactly:** silhouette 0.534, Davies-Bouldin 0.702 and adjusted Rand index 0.814.
- **The blobs are not four clean clusters.** Two of the four generating blobs are only 2.4 standard deviations apart. The silhouette is highest at 2 clusters (0.880), the Calinski-Harabasz index at 3, and only the ARI, which needs the labels, peaks at 4 (0.801). The book reads the elbow chart as "two to three clusters", which agrees.
- **K-means:**
  - the by-hand Lloyd loop gives the library's inertia and clusters, and the inertia never increases;
  - with a single start, random initialization reaches the best solution in 33% of 100 runs and k-means++ in 77%;
  - K-means breaks on stretched clusters (ARI 0.803 against 1.000 for a Gaussian mixture), and on a small tight cluster beside a large loose one (0.118 against 0.987).
- **Hierarchical clustering:**
  - Ward's costs equal the increase of the sum of squares, and the merge heights equal SciPy's;
  - the dendrogram's last heights (33.95, 5.21, 2.25) support 2 clusters strongly and 4 only weakly;
  - single linkage solves the circles (ARI 1.00) and fails on the moons (0.00, by chaining);
  - `BisectingKMeans` is a divisive method (its partitions are nested for every dataset tested), contrary to the book's statement, and plain K-means is not.
- **DBSCAN:**
  - in the book's result 290 points are core, 6 are border and 4 are noise;
  - the k-distance plot counts the point itself, and the points under `eps` are exactly the core points;
  - with `min_samples=5` only `eps=0.2` of the grid scores above 0.9;
  - with `min_samples=1` it equals a single-linkage cut.
- **Metrics:** on the moons, all three internal metrics prefer K-means (silhouette 0.485) to the true labelling (0.377), although the ARI says 0.507 against 1.000. Random labels get a Rand index of 0.625 and an ARI of -0.001.
- **Choosing an algorithm:**

  | ARI with the true labels | K-means | Ward | single linkage | DBSCAN | spectral | Gaussian mixture |
  |---|---|---|---|---|---|---|
  | blobs | 0.64 | 0.55 | 0.57 | 0.57 | 0.60 | **0.70** |
  | moons | 0.51 | 0.47 | 0.00 | **0.95** | 0.62 | 0.52 |
  | circles | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | 0.00 |

  Ward and dense spectral clustering grow fastest with the number of samples. Two noise features cut the ARI of K-means from 0.801 to 0.334, and a PCA first helps only when the noise dominates.
- **Spectral clustering and mixtures:** the neighbour graph of the moons has 2 components, and the Laplacian has exactly two zero eigenvalues; 3 neighbours break the graph into 19 components, and 50 link the two moons. The BIC picks 4 components for a tied covariance (the generating model) and 3 for the others. The mixture on the moons claims a confidence of 0.964 and is right for 86.3% of the points.
- **Exercises:** standardizing makes Iris worse (ARI 0.730 to 0.620); DBSCAN beats K-means on the moons (0.948 against 0.507) inside a window of about 0.25 to 0.35 for `eps`; two PCA components and a mixture recover 173 of 178 wines.

## Notes on the book vs. the current library

- All listings and the three official exercise solutions run on scikit-learn 1.9.1.
- Precisions to the chapter text:
  - K-means does not "randomly create centroids" by default: scikit-learn uses k-means++ and, since 1.4, `n_init='auto'`;
  - scikit-learn does have a divisive method, `BisectingKMeans`;
  - the "5th nearest neighbour" of the k-distance plot is the fourth other point, because the point is its own first neighbour;
  - the recipe 6 listing only draws the three datasets; no algorithm is applied to them (the notebook does);
  - spectral clustering "works well on large datasets" only with a sparse neighbour graph, not with the default dense `rbf` affinity;
  - internal metrics favour compact clusters, so they cannot compare algorithms of different kinds.
- The notebook is deterministic, except one cell that reports measured fit times.
- Every dataset is generated by scikit-learn or ships with it, so the notebook runs offline, in about a minute.
