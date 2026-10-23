# Chapter 9 — Text Processing and Multiclass Classification

**Notebook:** [`Chapter_09_Text_Processing_and_Multiclass_Classification.ipynb`](Chapter_09_Text_Processing_and_Multiclass_Classification.ipynb)
<a href="https://colab.research.google.com/github/fxrdhan/ML_scikit-learn_Cookbook_O-Reilly/blob/main/Chapter_09_Text_Processing_and_Multiclass_Classification/Chapter_09_Text_Processing_and_Multiclass_Classification.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Summary

Chapter 9 turns **text into numbers** and classifies it into more than two categories:

- **bag of words** — `CountVectorizer` counts the tokens of each document in a sparse matrix;
- **TF-IDF** — `TfidfVectorizer` down-weights the words found in most documents, and word embeddings are the dense alternative;
- **feature extraction** — n-grams, part-of-speech (POS) tags and named entities add context, syntax and names;
- **classification** — Naive Bayes, an SVM and logistic regression on the same TF-IDF matrix;
- **multiclass strategies** — one-vs-rest and one-vs-one, which turn K classes into binary problems;
- **evaluation** — precision, recall, F1 and the confusion matrix.

The recipes use three NLTK corpora (movie reviews, Reuters news wires and the Brown corpus of 15 genres). Three exercises close the chapter on tiny corpora.

The notebook reproduces every listing of the chapter and the official solutions of the three exercises. It adds a theoretical deep-dive for every recipe (the bag-of-words model, Zipf's law, the TF-IDF formula, n-grams, POS tagging and NER, multinomial Naive Bayes, the reduction of K classes to binary problems, the confusion matrix and its averages), and experiments that test the book's explanations.

## Recipes

| # | Recipe | Book code | What the notebook demonstrates |
|---|---|---|---|
| 1 | Introduction to text processing | pp. 214–218 | The bag of words built by hand, sparsity, Zipf's law, what the stop-word list removes, which preprocessing options matter |
| 2 | Text vectorization techniques | pp. 218–223 | TF-IDF computed by hand, what the "top 20" weights mean, BoW against TF-IDF on a classifier, LSA, the hashing trick |
| 3 | Feature extraction from text | pp. 223–231 | Vocabulary growth of n-grams, tagging the whole Brown corpus, the tagger's accuracy, POS tags as features, NER |
| 4 | Implementing text classification models | pp. 231–240 | Naive Bayes by hand, why all three models look weak, regularization sweeps, a confusion matrix of 15 genres |
| 5 | Multiclass classification strategies | pp. 240–245 | OvR against OvO counted and compared by repeated cross-validation, a two-level hierarchy |
| 6 | Evaluating text models | pp. 245–249 | The metrics from the confusion matrix, confidence intervals, ROC curve, learned words, learning curves, confident errors |
| 7 | Practical exercises in text processing | pp. 249–250 (official solutions) | What the toy corpora show, and why exercise 3 scores zero |

## Highlights from the notebook

- **The printed output of recipe 6 reproduces exactly:** precision 0.8034, recall 0.803 and F1 0.8030. The qualitative statements of the other recipes agree with the runs (Naive Bayes is the weakest, the SVM and logistic regression are close, OvR is slightly ahead of OvO, and `romance` has an F1 of 0 under OvO).
- **The bag of words:**
  - the book's 350 x 18,881 matrix is 1.36% filled, and Zipf's law holds (slope -0.98): 35% of the vocabulary occurs once;
  - scikit-learn's stop-word list removes every negation (*not*, *no*, *never*), yet stop-word removal changes the sentiment accuracy by only 0.5 points, while `binary=True` gains 2.3.
- **TF-IDF:**
  - the hand-made formula equals `TfidfVectorizer`;
  - the "odd" top weights of the book are Reuters shorthand (`vs`, `mln`, `cts`) and `lt`, an artifact of the markup `&lt;`;
  - for a linear SVM, TF-IDF beats counts by 2 points of accuracy and 6 points of macro F1, but Naive Bayes does not profit;
  - 400 LSA dimensions come to about half a point below the full matrix, and a hashing vectorizer with 4,096 columns loses 0.5 points.
- **Feature extraction:**
  - bigrams add 1.65 points on counts, and with binary presence 4.0 points in all, at the price of 14 times as many features;
  - tagging the **whole sentence before removing stop words** agrees with the corpus annotation on 94.0% of the content words, against 89.8% in the book's order;
  - 44 POS-tag proportions predict the genre (0.488) almost as well as 42,396 words (0.510);
  - NER is announced in the text but absent from the listing; the notebook adds it with NLTK's chunker.
- **The text classifiers:**

  | Model | Book default | Strength tuned by validation |
  |---|---|---|
  | Naive Bayes | 0.228 | 0.452 (`alpha` = 0.0001) |
  | SVM (RBF kernel) | 0.332 | 0.428 (`C` = 10) |
  | Logistic regression | 0.324 | 0.492 (`C` = 1000) |

  The majority class scores 0.160. The defaults over-regularize unit-length TF-IDF rows (no model even fits its training set), and the tuned models cannot be ranked on 250 test documents. Naive Bayes predicts only the two largest genres, and the prior is not the cause.
- **Multiclass strategies:** OvR trains 15 models and OvO 105. With the book's `C = 1`, OvR beats OvO by 5.7 points on every fold; with `C = 100` the gap is 2.4 points, within the noise. A multinomial model ties with OvR, and a two-level hierarchy ties with the flat model (0.514 against 0.517).
- **Evaluation:** weighted recall is the accuracy; the 95% interval of the 0.803 is 0.777 to 0.826, the AUC is 0.891, and the model still improves with more data.
- **Exercise 3:** the official solution scores 0 by construction, because the two test classes never occur in the training set.

## Notes on the book vs. the current library

- All listings and the three official exercise solutions run on scikit-learn 1.9.1 and NLTK 3.10.4.
- **NLTK data:** the notebook downloads each resource only when it is missing (`ensure_nltk()`), where the book calls `nltk.download()` unconditionally. NLTK 3.10 refuses to download through a proxy that performs the egress; in that case, fetch the zip files by hand into the NLTK data folder.
- **A broken probe in recipe 4:** the listing catches `nltk.downloader.DownloadError`, which does not exist in NLTK 3.10. The listing works only because recipe 3 downloaded the legacy `punkt` package first; in a clean environment it stops with an `AttributeError`.
- **Slips in the text:**
  - the quoted description of the corpus in recipe 5 belongs to NLTK's `webtext`, not to `brown`;
  - the comment "With 6 documents in brown" is a leftover;
  - the printed listing of recipe 5 splits without `stratify`, while the official notebook stratifies;
  - the comment saying that `confusion_matrix` "computes class-wise accuracy" is wrong;
  - recipe 3 announces NER and the listing has none.
- The POS-tag figure removes the stop words **before** tagging, which lowers the tagger's accuracy.
- Precisions to the chapter text:
  - the book's claim that TF-IDF classifies "much more efficiently" holds for a linear SVM but not for Naive Bayes;
  - OvO is not better than OvR here, although the book says it "often" is on balanced data;
  - the book's models look weak mainly because of default regularization, not because of the dataset size alone.
- The notebook is deterministic, except one cell that reports measured fitting times, and the random HTML ids of the styled reports.
- It needs NLTK (>= 3.9) and about six minutes on four CPU cores to run; the corpora (about 20 MB) are downloaded on the first run.
