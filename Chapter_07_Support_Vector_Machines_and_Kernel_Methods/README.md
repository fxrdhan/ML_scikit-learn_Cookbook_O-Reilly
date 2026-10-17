# Chapter 7 — Support Vector Machines and Kernel Methods

**Notebook:** [`Chapter_07_Support_Vector_Machines_and_Kernel_Methods.ipynb`](Chapter_07_Support_Vector_Machines_and_Kernel_Methods.ipynb)
<a href="https://colab.research.google.com/github/fxrdhan/ML_scikit-learn_Cookbook_O-Reilly/blob/main/Chapter_07_Support_Vector_Machines_and_Kernel_Methods/Chapter_07_Support_Vector_Machines_and_Kernel_Methods.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Summary

Chapter 7 introduces **support vector machines (SVMs)**, which separate classes with the hyperplane of widest **margin**, and **kernel functions**, which let that hyperplane live in a high-dimensional feature space without ever computing it. The recipes:

- fit `SVC` and `SVR` on Iris and draw a decision surface;
- compare the **linear, polynomial and RBF kernels**;
- **tune** `C` and the kernel parameters with grid search and cross-validation;
- apply SVMs to a **high-dimensional** dataset (1,000 features) and look at it through PCA;
- **evaluate** an SVM with a classification report and a ROC curve on the Breast Cancer data.

Three exercises then build a simple classifier, run a grid search and draw a decision boundary.

The notebook reproduces every listing of the chapter and the official solutions of the three exercises. It adds a theoretical deep-dive for every recipe (margin, hinge loss, dual problem, kernel trick, Platt scaling), and experiments that test the book's claims.

## Recipes

| # | Recipe | Book code | What the notebook demonstrates |
|---|---|---|---|
| 1 | Introduction to SVMs | pp. 159–163 | The margin and the support vectors drawn and measured, the dual solution checked, what the book's decision-boundary figure really shows |
| 2 | Kernel functions and their applications | pp. 165–170 | The kernel trick verified, the three kernels over 50 cross-validation folds, concentric circles where the kernel choice decides everything |
| 3 | Tuning SVM parameters | pp. 171–175 | Redundant candidates and ties in the book's grid, `C` and `gamma` tuned together |
| 4 | SVMs in high-dimensional spaces | pp. 176–179 | Why the training accuracy is perfect, five alternatives compared, why the PCA picture looks messy |
| 5 | Evaluating SVM models | pp. 181–183 | A stale variable in the book's printout, probabilities versus decision scores, what scaling does to an SVM |
| 6 | Practical exercises with SVMs | pp. 185–186 (official solutions) | The grid search with and without scaling, the margins behind exercise 3's boundary |

## Highlights from the notebook

- **The printed outputs reproduce** on scikit-learn 1.9.1: 93% accuracy for the linear SVC, MSE 0.20 for the SVR, and a ROC AUC of 0.98.
- **The geometry made concrete:**
  - a hard-margin SVM between setosa and versicolor has a 1.68 cm margin defined by just two support vectors, whose decision values are exactly $\pm 1$;
  - `coef_` equals $\sum_i \alpha_i y_i x_i$ over the support vectors.
- **Kernel ranking:**
  - the book's 45-flower test split ranks polynomial (0.96) > linear (0.93) > RBF (0.91), one flower apart each;
  - over 50 cross-validation folds the **linear kernel is best** (0.977).
  - On concentric circles, the linear kernel (0.55) and the *default* polynomial kernel (0.64, homogeneous cubic) fail, while a degree-2 polynomial with `coef0=1` and the RBF kernel both reach 1.00.
- **Tuning:**
  - the book's 27-candidate grid contains only 15 distinct models, and 8 candidates tie at the best score;
  - the "best" `C=0.1, poly, degree=4` wins by grid order and scores 0.933 on the test set, below the default polynomial SVC (0.956);
  - the `C` curve starts at 0.362, the share of the largest class, and peaks near $C \approx 3$.
- **High dimensions** (1,000 features, 700 training samples):

  | Model | CV accuracy | Test accuracy |
  |---|---|---|
  | linear SVC (book) | 0.70 | 0.75 |
  | $L_1$ `LinearSVC` (53 weights) | 0.77 | 0.83 |
  | RBF SVC | 0.86 | 0.88 |

  - The linear SVM separates the training set perfectly, as it would even with random labels, because $p > n$.
  - PC1 and PC2 explain only 4.6% of the variance, which is why the PCA plot looks messy.
- **Scaling** the Breast Cancer features:
  - raises the linear SVM from 0.947 to 0.982 accuracy (AUC 0.984 to 0.999), and makes the fit two to three orders of magnitude faster;
  - in exercise 2, the scaled grid search runs in under a second instead of about a minute, and scores 0.982 instead of 0.947 on the test set.

## Notes on the book vs. the current library

- All listings and the three official exercise solutions run on scikit-learn 1.9.1.
- **`SVC(probability=True)`** (recipe 5) is deprecated since scikit-learn 1.9 and will be removed in 1.11. The warning is silenced in the setup cell. The notebook shows the replacement, `CalibratedClassifierCV(SVC(), ensemble=False)`, and that the ROC curve only needs `decision_function`.
- **A stale variable:** the evaluation listing prints `accuracy` without computing it, so the book's code prints the previous recipe's 0.75; the real accuracy is 0.947.
- The notebook is deterministic, except two cells that report measured fitting times, and the random HTML ids of the styled reports.
- Recipe 1's decision-boundary listing plots the **regression** model (`SVR`), so its colours are predicted class codes; its extra zero columns are never used.
- Precisions to the chapter text:
  - a hyperplane is flat;
  - the linear kernel does not map the data into a higher-dimensional space;
  - the RBF kernel's feature space is infinite-dimensional;
  - the `C` curve peaks rather than plateaus;
  - in the high-dimensional recipe the RBF kernel beats the linear one.
- Every dataset ships with scikit-learn or is generated with a fixed seed, so the notebook runs offline. Exercise 2's official solution takes about a minute.
