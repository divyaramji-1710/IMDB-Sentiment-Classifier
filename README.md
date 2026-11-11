# IMDB Sentiment Classifier

A Python project that profiles 50,000 IMDB movie reviews and classifies their sentiment with TF-IDF and Multinomial Naive Bayes.

## What the project covers

1. **Statistics:** label balance, the 50% majority-class baseline, review-length statistics, skewness, average length by sentiment, and duplicate rows.
2. **NLP cleaning:** HTML removal, lowercasing, NLTK tokenization, negation-preserving stop-word removal, alphabetic filtering, and WordNet lemmatization. Includes a stemming comparison and a 1,000-review timing experiment.
3. **Classification:** a 20,000-review trial followed by a full-corpus experiment, stratified train/test splits, evaluation reports, confusion matrices, and influential unigrams and bigrams.

## Setup

Python 3.11 is recommended. From the repository folder:

```sh
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```sh
source .venv/bin/activate
```

Then install dependencies and NLTK resources:

```sh
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4
```

## Run the complete project

```sh
python src/sentiment_classifier.py
```

This prints the full analysis and saves the same output to `results/console_output.txt`. Evaluation metrics and feature tables are saved under `results/`. The fitted vectorizer and classifier are saved together in `models/sentiment_pipeline.joblib`; this generated binary is excluded from Git and can be recreated with the command above.

## Notebook workflow

```sh
jupyter notebook
```

Run each notebook from top to bottom:

- `notebooks/part1.ipynb`: original corpus statistics.
- `notebooks/part2_cleaning.ipynb`: cleaning demonstration, timing, and negation checks.
- `notebooks/part3_classification.ipynb`: trial and full-corpus training, evaluation, and predictions.

Parts 2 and 3 import the shared implementation in `src/sentiment_classifier.py` so the notebook and command-line workflows use the same cleaning and model settings. They each load their own data and do not require hidden state from another notebook.

## Evaluation design

The full-corpus workflow processes all 50,000 source rows. Before modeling, empty cleaned texts are excluded, conflicting labels for identical cleaned text are excluded, and repeated cleaned texts are removed. This means the full training and testing population contains fewer than 50,000 rows. Both experiments use an 80/20 stratified split with random_state=42. No identical cleaned text may occur in both splits. TF-IDF is fitted on training data only.

The model uses `TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.8, max_features=50000, sublinear_tf=True)` and `MultinomialNB(alpha=0.5)`. Feature association is calculated as positive minus negative class log probability from `feature_log_prob_`; this avoids confusing generally frequent words with sentiment-specific terms. The reports include the strongest bigrams separately as well.

## Recorded results

| Experiment | Training reviews | Test reviews | Accuracy |
| --- | ---: | ---: | ---: |
| 20,000-review trial | 16,000 | 4,000 | 88.28% |
| Full corpus after duplicate handling | 39,659 | 9,915 | 88.23% |

The assignment baseline is 50%. Results use a fixed held-out split; they are not a guarantee for new domains. See the project report and saved console output for details.

## Results and documentation

- `results/console_output.txt`: saved output of the complete script.
- `results/*_metrics.json`: accuracy, split sizes, classification metrics, confusion matrices, and leading terms.
- `results/*_classification_report.txt`: readable precision, recall, and F1 scores.
- `results/*_terms.csv`: top 20 terms for each class.
- `docs/project_report.md`: findings, methodological choices, and limitations.
- `docs/requirements_checklist.md`: mapping from assignment requirements to implementation.

## Dataset and citation

File: `data/IMDB Dataset.csv`, with `review` and `sentiment` columns.

Kaggle distribution: [IMDB Dataset of 50K Movie Reviews, Lakshmi N Pathi](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

Original source: [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

Citation: Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., and Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis*. Proceedings of ACL, 142-150. Consult the original dataset's terms before reusing the data.

## Limitations

The classifier can miss sarcasm and negation that spans more than a bigram. Results on IMDB reviews may not transfer to other languages, domains, or contemporary review styles. Keeping negation tokens improves representation but does not guarantee every negative phrase is classified correctly. The dataset has no film identifiers, so per-film positivity scores cannot be computed from this file alone.
