# IMDB review sentiment classification

## Objective and data

Classify IMDB reviews as positive or negative and describe the review corpus. The supplied CSV contains 50,000 rows, equally divided between positive and negative sentiment. Always predicting one class therefore gives a 50% baseline.

## Corpus findings

The average review contains 231.16 whitespace-separated words, while the median is 173. The most populated word-count band is 100–149, the sample variance is 29,358.77, the sample standard deviation is 171.34, and skewness is 2.1683. A small number of long reviews pull the mean upward, so the median better describes a typical review.

Positive reviews average 232.85 words and negative reviews 229.46, a difference of approximately 3.38 words. This is small compared with the overall spread of review lengths; it does not suggest a useful practical separation based on length alone. No statistical significance test was performed.

## Cleaning and leakage prevention

HTML tags are replaced with spaces, entities are decoded, and text is lowercased. Negative contractions are expanded before NLTK word_tokenize. Stop words are removed except not, no, never, nor, neither, and cannot; only alphabetic tokens remain. WordNet noun lemmatization reduces inflection while retaining readable features. Treebank tokenization may split cannot into can and not; can is removed and not remains.

There are 418 exact duplicate rows. After cleaning, 426 repeated cleaned texts are removed, including texts made identical by normalization. No reviews become empty, and no cleaned texts have conflicting labels. The final modeling population contains 49,574 unique reviews. All 50,000 source rows are processed, but repeated text is excluded before splitting so test performance is not inflated by exact cleaned-text overlap.

The cleaner constructs reusable resources once rather than once per review. For the recorded complete-script run, median times over three warmed runs on the first 1,000 reviews were 3.255 seconds with reusable resources and 8.401 seconds with repeated setup, a difference of 5.146 seconds. Timing varies with hardware, background work, and caching; notebook timings may differ.

## Model and evaluation

The 20,000-review trial is sampled with stratification from usable reviews. Both experiments then use an 80/20 stratified train/test split with random_state=42. TF-IDF is fitted only on training text, using unigrams and bigrams, min_df=5, max_df=0.8, max_features=50000, and sublinear_tf=True. The classifier is MultinomialNB(alpha=0.5). These are fixed assignment settings, not the result of tuning against test labels.

| Experiment | Training rows | Test rows | Test accuracy | Baseline |
| --- | ---: | ---: | ---: | ---: |
| 20,000-review trial | 16000 | 4000 | 88.28% | 50.00% |
| Full corpus after duplicate handling | 39659 | 9915 | 88.23% | 50.00% |

The full-corpus accuracy exceeds the assignment baseline by 38.23 percentage points. The trial and full-corpus results are similar; they use different test sets, so this comparison does not establish that either training size is superior.

Full-corpus confusion matrix (rows are true labels; columns are predicted labels):

| | Predicted negative | Predicted positive |
| --- | ---: | ---: |
| True negative | 4332 | 607 |
| True positive | 560 | 4416 |

The complete precision, recall, and F1 reports are saved under results/. Leading features are ranked by the difference between positive and negative feature_log_prob_ values. Positive associations include cast excellent, highly recommended, and acting superb; negative associations include worst movie, worst film, and one worst. The full run's top 20 lists contain 13 positive-associated bigrams and 18 negative-associated bigrams. Names such as goldsworthy and uwe also appear, illustrating that the model can learn corpus-specific associations.

## Negation checks and limitations

The cleaner retains negative meaning in the tested contractions and negation tokens. The fitted model predicts “This movie is good” as positive and “This movie is not good” as negative. However, it predicts “I never enjoyed this film” as positive. These illustrative examples show why preserving negation is necessary but does not guarantee correct interpretation.

Two main limitations remain: the unigram/bigram representation can miss sarcasm and long-range negation, and a held-out IMDB score does not establish accuracy in other languages, domains, or newer review styles. Noun-only lemmatization also leaves some inflected verbs unchanged. The data contains no film identifiers, so the business scenario's per-film positivity scores would require additional metadata.

## Reproducibility

Run python src/sentiment_classifier.py from the repository root after following the README setup. The script regenerates the console log, experiment metrics, term tables, and fitted pipeline. Notebook Parts 2 and 3 exercise the same implementation and can run independently. The generated model binary is excluded from Git and is recreated locally. Only load joblib files from sources you trust.

Recorded script environment: Python 3.11.9; pandas 3.0.3; numpy 2.4.6; nltk 3.10.3; scikit-learn 1.9.0; joblib 1.5.3.

## Reference

Maas, A. L., et al. (2011). Learning Word Vectors for Sentiment Analysis. Proceedings of ACL, 142–150. Original data: https://ai.stanford.edu/~amaas/data/sentiment/ . Kaggle distribution: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews .
