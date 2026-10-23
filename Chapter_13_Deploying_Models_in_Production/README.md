# Chapter 13 — Deploying scikit-learn Models in Production

**Notebook:** [`Chapter_13_Deploying_Models_in_Production.ipynb`](Chapter_13_Deploying_Models_in_Production.ipynb)
<a href="https://colab.research.google.com/github/fxrdhan/ML_scikit-learn_Cookbook_O-Reilly/blob/main/Chapter_13_Deploying_Models_in_Production/Chapter_13_Deploying_Models_in_Production.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Summary

Chapter 13 takes a trained model out of the notebook. It covers:

- **packaging and persisting** a model with `joblib` and `pickle`, and the precautions that come with it (trust, library versions, ONNX and PMML for Python-free environments);
- **scaling** training and prediction: `n_jobs`, `joblib` backends, Dask, and why batch serving is cheaper than serving one row at a time;
- **monitoring** a deployed model on a stream of batches, detecting degradation, and updating it with `partial_fit`;
- **managing the life cycle** of a model: a versioned artifact, metadata, a validation snapshot, a way back;
- **deployment pipelines**: a `Pipeline` exported as one artifact, promoted only if a validation threshold is met, in the CI/CD/CT cycle.

Three exercises then repeat the workflow.

The notebook reproduces every listing of the chapter and the official solutions of the three exercises. It adds a theoretical deep-dive for every recipe (what a pickle contains, latency as a fixed cost plus a cost per row, Amdahl's law, covariate shift against concept drift, the learning-rate schedule of `partial_fit`, the statistics of a threshold gate, versioning and registries), and experiments that test the book's explanations. All files that the listings write are created in a scratch directory, not in the repository.

## Recipes

| # | Recipe | Book code | What the notebook demonstrates |
|---|---|---|---|
| 1 | Overview of model deployment | pp. 332–334 | What the artifact contains, a round-trip test that can detect a broken model, the cost of one call against one batch, a minimal prediction service |
| 2 | Serialization and persistence techniques | pp. 334–336 | Fidelity of the round trip, size and speed of `pickle` and `joblib` (protocols, compression, memory mapping), why unpickling executes code and how to restrict it, code that cannot travel with a pickle |
| 3 | Scaling models for production | pp. 336–338 | Latency against batch size and `n_jobs`, speed-up of training and cross-validation (Amdahl), number of trees against latency, why the book's CV accuracy changes between runs |
| 4 | Monitoring and updating deployed models | pp. 338–341 | What the book's "stream" really is, false alarms of a threshold, five update policies under a simulated drift, drift alarms against accuracy alarms, what `partial_fit` does |
| 5 | Managing the model life cycle | pp. 341–343 | The book's validation line fails, honest metadata and a validation snapshot, a verified loader, a simulated library upgrade, a toy registry with rollback, reproducibility |
| 6 | Setting up deployment pipelines | pp. 343–345 | The gate that can never open, how noisy a threshold gate is, a gate that fails the build, a gate with several checks, training-serving skew |
| 7 | Practical exercises in model deployment | pp. 345–346 (official solutions) | A round trip verified in a fresh process, a monitoring exercise whose setting does not degrade the model, a deployment check that rejects for the wrong reason |

## Highlights from the notebook

- **Persistence.**
  - The artifact of recipe 1 is 1,007 bytes of learned state plus a reference to code; the book's round-trip test (one random point) would miss a model with zeroed coefficients 55% of the time, while comparing 10,000 probabilities cannot miss it.
  - `pickle` (protocol 4 or 5) writes the same size as `joblib` (16.0 MB for a 100-tree forest) and was not slower for a forest, while `joblib` loaded a large array faster; the real advantages of `joblib` are `compress=3` (3.7 MB, 4.4 times smaller) and `mmap_mode="r"` (loading a 40 MB model takes a fraction of a millisecond instead of 10 to 20 ms).
  - Unpickling executes code: a harmless payload prints a line when loaded, a forest asks the loader for only seven names, and a restricted unpickler blocks the payload; an HMAC signature detects one flipped bit.
  - A lambda cannot be pickled, a function defined in a notebook pickles but cannot be loaded in a fresh process, and a function in an importable module loads only where the module is installed.
- **Scaling.**
  - A call has a fixed cost: a linear model needs about 100 microseconds for 1, 10 or 100 rows, so a row alone costs roughly 900 times more per row than inside a batch of 1,000.
  - With the book's `n_jobs=-1`, a 100-tree forest predicts small batches 3 to 6 times slower than with `n_jobs=1`, the book's own 1,000-row batch about 2.5 to 3 times slower, and 100,000 rows about 3 times faster.
  - On 4 cores training is 2.5 to 3 times faster (Amdahl parallel fraction 0.8 to 0.9); in cross-validation the folds are the level to parallelise.
  - Beyond about 10 trees the accuracy stays flat while size and latency grow in proportion. The book's printed `CV Accuracy` has a standard deviation of 0.003 over seeds, so it is not reproducible digit by digit.
- **Monitoring.**
  - The book's stream is a different problem, not a drifted one: its four batch accuracies are 0.38, 0.30, 0.36 and 0.38, below chance, and over 199 other seeds the model scores 0.497 on average. The "upward trend" in the text is not there, and the retraining step lowers the accuracy on the original training data from 0.892 to 0.544.
  - On batches of 50 rows, a model of true accuracy 0.835 raises the alarm "below 0.8" in 19% of the batches.
  - Under a simulated 90-degree concept drift, a frozen model falls to chance (mean accuracy 0.696), updating only on the alarming batch (a per-batch version of the book's rule) reaches 0.849, and updating on every batch with a constant learning rate reaches 0.890. A Kolmogorov–Smirnov monitor alarms for an irrelevant feature shift (no harm), alarms while accuracy rises (a relevant feature shift), and stays silent under concept drift (accuracy 0.499).
  - `warm_start` has no effect on `partial_fit`; the counter `t_` explains why the default learning rate adapts ever more slowly.
- **Life cycle.**
  - The book's validation line (`load_iris` against a 20-feature model) raises a `ValueError`; the official version validates on rows used to fit the model (accuracy 1.0000 against 0.985 on unseen rows), and the metadata hard-codes `"sklearn": "1.3.2"`.
  - A simulated library upgrade raises 52 `InconsistentVersionWarning`; a verified loader refuses a flipped bit, a version mismatch and a different model whose checksum was recomputed (only the validation snapshot catches that one).
  - A toy registry shows promotion with a margin and a one-write rollback; `n_jobs` changes the bytes of a pickle but not its predictions.
- **Deployment pipelines.**
  - The book's gate draws random inputs and random labels: accuracy 0.500 on average and a probability of 1.4e-10 of ever reaching 0.8, so *Auto-deploy allowed* cannot be printed; and its `print` ends with exit code 0, which a CI system reads as success.
  - A threshold gate is noisy: with 100 test rows a model of true accuracy 0.82 opens the gate 66% of the time; a Wilson lower-bound rule needs 87 of 100 correct.
  - A gate with schema, quality, non-regression and behavior checks refuses five faulty candidates for the right reasons; a scaler lost between training and serving drops the accuracy from 0.940 to 0.523, which a `Pipeline` prevents.
- **Exercises.** The reloaded pipeline of Exercise 1 is identical in a fresh process; the setting of Exercise 2 (`alpha=1.0`) is as accurate as the recipe's model on a coherent stream and degrades only from `alpha=10`; Exercise 3 rejects a model with accuracy 0.11 because the test set is another problem, and approves it for only 13% of 200 test seeds.

## Notes on the book vs. the current library

- All listings and the three official exercise solutions run on scikit-learn 1.9.1.
- Precisions to the chapter text:
  - `joblib` is not part of the standard library, and the round-trip test of recipe 1 predicts one random point;
  - the serialized model is a byte stream, not "Python arrays", and `pickle` no longer "struggles with large NumPy arrays" (protocol 5);
  - the batch of recipe 3 is random numbers, not "text data", and `n_jobs=-1` makes its 1,000-row batch slower;
  - the stream of recipe 4 has four batches (not three), comes from another `make_classification` seed (another problem), shows no upward trend, and `warm_start` is irrelevant for `partial_fit`;
  - the validation line of recipe 5 fails with a `ValueError` for a 20-feature model, and the official replacement validates on training rows;
  - the "test set" of recipe 6 has random labels, so the gate cannot open, and a `print` does not fail a CI job;
  - Exercises 2 and 3 reuse earlier variables or another problem's data, as described above.
- The listings that write files (`model.joblib`, `rf.pkl`, `pipeline.joblib`, ...) are run in a temporary directory created by the setup cell; the listings that draw random inputs from the unseeded global generator get `np.random.seed(2024)` as a marked first line, and the printed time and `CV Accuracy` of recipe 3 remain run-dependent.
- The notebook is deterministic, except the cells that report measured times (marked in their second line). The extras never touch NumPy's global random generator, so they cannot change the book's numbers.
- Every dataset is generated by NumPy and scikit-learn, so the notebook runs offline, in about two and a half minutes. It uses no package beyond the dependencies of the repository: Dask, ONNX (`skl2onnx`), `skops` and MLflow are described but not run.
