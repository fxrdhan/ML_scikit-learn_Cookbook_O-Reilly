# ML_scikit-learn_Cookbook_O-Reilly

**Code reproduction and theoretical deep-dive of _scikit-learn Cookbook, Third Edition_**

This repository reproduces, chapter by chapter, the code of **_scikit-learn Cookbook, Third Edition — Over 80 recipes for machine learning in Python with scikit-learn_** by **John Sukup** (Packt Publishing, 2025). Each chapter has its own Jupyter notebook that:

1. **reproduces** every code listing of the chapter and runs it against a current scikit-learn release,
2. **summarizes** the chapter and each of its recipes,
3. **explains the theory** behind every concept (math, algorithms and design reasoning), and
4. adds **extra experiments** that turn concepts the book explains only in prose into runnable code.

> Assignment: *Tugas 2 — Enrichment for Machine Learning classes: "Code Reproduction + Theoretical Deep-Dive from scikit-learn Cookbook"*.

---

## 📚 Chapters

| # | Chapter | Status | Notebook |
|---|---|---|---|
| 01 | Common Conventions and API Elements of scikit-learn | ✅ Done | [Chapter 1 notebook](Chapter_01_Common_Conventions_and_API_Elements/Chapter_01_Common_Conventions_and_API_Elements.ipynb) · [summary](Chapter_01_Common_Conventions_and_API_Elements/README.md) |
| 02 | Pre-Model Workflow and Data Preprocessing | ⏳ Planned | — |
| 03 | Dimensionality Reduction Techniques | ⏳ Planned | — |
| 04 | Building Models with Distance Metrics and Nearest Neighbors | ⏳ Planned | — |
| 05 | Linear Models and Regularization | ⏳ Planned | — |
| 06 | Advanced Logistic Regression and Extensions | ⏳ Planned | — |
| 07 | Support Vector Machines and Kernel Methods | ⏳ Planned | — |
| 08 | Tree-Based Algorithms and Ensemble Methods | ⏳ Planned | — |
| 09 | Text Processing and Multiclass Classification | ⏳ Planned | — |
| 10 | Clustering Techniques | ⏳ Planned | — |
| 11 | Novelty and Outlier Detection | ⏳ Planned | — |
| 12 | Cross-Validation and Model Evaluation Techniques | ⏳ Planned | — |
| 13 | Deploying scikit-learn Models in Production | ⏳ Planned | — |

---

## 🧭 Chapter-by-chapter overview

The book moves from the conventions of the library, through data preparation and the main families of algorithms, to evaluation and deployment. A short explanation of each chapter:

### Chapter 1 — Common Conventions and API Elements of scikit-learn
The "grammar" of scikit-learn. Every object follows the same design: **estimators** learn with `fit()` and predict with `predict()`, **transformers** learn and apply a transformation with `fit()`/`transform()`, and **pipelines** chain them into one object. The chapter also covers custom estimators built from `BaseEstimator` and mixins, common attributes and methods (`coef_`, `intercept_`, `score()`), hyperparameter management (`get_params()`, `set_params()`, `GridSearchCV`, `RandomizedSearchCV`), metadata (estimator tags and metadata routing) and best practices for using the API.
*Key tools:* `fit`/`predict`/`transform`, `Pipeline`, `BaseEstimator`, `GridSearchCV`, `get_tags`.

### Chapter 2 — Pre-Model Workflow and Data Preprocessing
"Garbage in, garbage out": data quality largely decides model quality. The chapter shows how raw data affects performance and how to handle common data issues — **missing values** (imputation), **scaling** numeric features, **encoding categorical variables** — and then combines these steps into **pipelines** (including how to visualize them). It closes with **feature engineering** and an exercise that builds a complete preprocessing pipeline.
*Key tools:* `SimpleImputer`, `KNNImputer`, `IterativeImputer`, `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`, `PolynomialFeatures`, `RFE`.

### Chapter 3 — Dimensionality Reduction Techniques
Why fewer, better features help (less noise, less computation, mitigating the curse of dimensionality) and the theory behind reducing dimensions while keeping information. Covers **PCA** (unsupervised projection onto directions of maximum variance), **LDA** (supervised projection that maximizes class separability), how PCA and LDA differ, and **t-SNE** for non-linear visualization — plus guidelines for choosing a technique and its impact on model performance.
*Key tools:* `PCA`, `LinearDiscriminantAnalysis`, `TSNE`.

### Chapter 4 — Building Models with Distance Metrics and Nearest Neighbors
Models that predict from the most similar training examples. Introduces **distance metrics** (Euclidean, Manhattan, Minkowski, …), the **k-nearest neighbors (KNN)** algorithm, **tuning** its hyperparameters (number of neighbors, weighting, metric) with grid search, and **evaluating** KNN classifiers with appropriate metrics.
*Key tools:* `KNeighborsClassifier`, `GridSearchCV`, `cross_val_score`, `learning_curve`, `confusion_matrix`.

### Chapter 5 — Linear Models and Regularization
Linear regression and the problem of overfitting. Covers **ordinary least squares**, **Ridge** (L2) and **Lasso** (L1) regression, **ElasticNet** (a mix of both), and the theory and practice of **regularization** — how penalizing coefficients trades a little bias for lower variance and how Lasso performs feature selection.
*Key tools:* `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `PolynomialFeatures`.

### Chapter 6 — Advanced Logistic Regression and Extensions
Logistic regression as a probabilistic classifier and its extensions: **multiclass** strategies (one-vs-rest and multinomial/softmax), **regularization** in logistic regression (`C`, L1/L2 penalties), **multilabel** classification, and the **evaluation metrics** used for classifiers (precision, recall, F1, ROC-AUC, …), with exercises on visualizing results.
*Key tools:* `LogisticRegression`, `classification_report`, `roc_curve`/`auc`, `DecisionBoundaryDisplay`.

### Chapter 7 — Support Vector Machines and Kernel Methods
**Support vector machines** find the maximum-margin decision boundary. The chapter explains **kernel functions** (linear, polynomial, RBF, …) and the kernel trick for non-linear problems, **tuning** `C` and `gamma`, SVMs in **high-dimensional** spaces, and **evaluating** and visualizing SVM decision boundaries.
*Key tools:* `SVC`, `SVR`, kernels, `GridSearchCV`.

### Chapter 8 — Tree-Based Algorithms and Ensemble Methods
**Decision trees** (recursive splits that reduce impurity) and **ensembles** that combine many trees: **random forests and bagging** (averaging decorrelated trees to cut variance) and **gradient boosting machines** (adding trees sequentially to correct errors). Includes hyperparameter tuning for trees and ensembles and a comparison of ensemble methods.
*Key tools:* `DecisionTreeClassifier`, `plot_tree`, `RandomForestClassifier`, `GradientBoostingClassifier`.

### Chapter 9 — Text Processing and Multiclass Classification
Turning text into numbers and classifying it. Covers **text preprocessing**, **vectorization** (bag-of-words and **TF-IDF**), **feature extraction** with n-grams, building **text classification** models, **multiclass strategies** (one-vs-rest, one-vs-one) and **evaluating** text models.
*Key tools:* NLTK, `CountVectorizer`, `TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, `OneVsRestClassifier`, `OneVsOneClassifier`.

### Chapter 10 — Clustering Techniques
Unsupervised learning that finds natural groupings without labels: **K-means**, **hierarchical clustering**, density-based **DBSCAN**, **cluster evaluation metrics** (silhouette, Davies–Bouldin, adjusted Rand index), how to choose the right algorithm, and advanced techniques such as spectral clustering and Gaussian mixture models.
*Key tools:* `KMeans`, `AgglomerativeClustering` (+ SciPy dendrograms), `DBSCAN`, `SpectralClustering`, `GaussianMixture`, `silhouette_score`.

### Chapter 11 — Novelty and Outlier Detection
Finding unusual observations — **outliers** in the training data and **novelties** in new data. Covers **Isolation Forest**, **One-Class SVM** and the **Local Outlier Factor (LOF)**, how to evaluate detectors, how to handle detected outliers, and how to choose the right technique.
*Key tools:* `IsolationForest`, `OneClassSVM`, `LocalOutlierFactor`.

### Chapter 12 — Cross-Validation and Model Evaluation Techniques
Reliable estimates of how a model will perform on unseen data. Covers **k-fold cross-validation** and more advanced schemes (stratified k-fold, leave-one-out, …), implementing CV in scikit-learn, **model selection** with grid and randomized hyperparameter search, and **generalization** diagnostics such as learning and validation curves.
*Key tools:* `cross_val_score`, `cross_validate`, `StratifiedKFold`, `LeaveOneOut`, `GridSearchCV`, `RandomizedSearchCV`, `learning_curve`, `validation_curve`.

### Chapter 13 — Deploying scikit-learn Models in Production
From notebook to production: **serializing and persisting** models (joblib, pickle), **scaling** models for production workloads, **monitoring and updating** deployed models (e.g. when data drift degrades accuracy), managing the **model life cycle**, and building **deployment pipelines** with automated validation checks.
*Key tools:* `joblib`, `pickle`, `Pipeline`, `SGDClassifier.partial_fit` (incremental model updates), validation thresholds before deployment.

---

## 🗂️ Repository structure

```
ML_scikit-learn_Cookbook_O-Reilly/
├── README.md                                          ← this file (overview of every chapter)
├── requirements.txt                                   ← Python dependencies
└── Chapter_01_Common_Conventions_and_API_Elements/
    ├── README.md                                      ← chapter summary
    └── Chapter_01_Common_Conventions_and_API_Elements.ipynb
```

New chapters are added as `Chapter_XX_<Title>/` folders with the same layout.

## 📓 How each notebook is organized

| Marker | Meaning |
|---|---|
| **📝 Summary** | What the book says in the recipe, condensed |
| **📖 Theory** | The concepts behind the recipe: math, algorithms, design reasoning |
| **📘 Book code** | The listing reproduced from the book (with its page number) + an analysis of the output |
| **🧪 Extra** | Additional experiments written for this repository (not part of the book) |

All notebooks are committed **with their outputs**, use fixed random seeds, and rely only on datasets bundled with scikit-learn or generated synthetically, so they run offline and reproduce the same results.

## 🚀 Running the notebooks

**Colab:** open a notebook on GitHub and use its **Open in Colab** badge.

**Locally:**

```bash
git clone https://github.com/fxrdhan/ML_scikit-learn_Cookbook_O-Reilly.git
cd ML_scikit-learn_Cookbook_O-Reilly
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

The book targets **scikit-learn 1.5**; the notebooks were executed with **scikit-learn 1.9.1 on Python 3.12**. Differences between the book's version and the current API are pointed out where they matter (for example, the public estimator-tags API introduced in scikit-learn 1.6).

## 📖 References

- Sukup, J. (2025). *scikit-learn Cookbook* (3rd ed.). Packt Publishing.
- Official code repository of the book: <https://github.com/PacktPublishing/scikit-learn-Cookbook-Third-Edition>
- scikit-learn documentation: <https://scikit-learn.org/stable/>
- Buitinck, L., et al. (2013). API design for machine learning software: experiences from the scikit-learn project. <https://arxiv.org/abs/1309.0238>
