# Assignment checklist

| Requirement | Implementation / evidence |
| --- | --- |
| Corpus size, labels, percentages, baseline | part1.ipynb; statistics(); console output |
| Word-count mean, median, mode band, variance, standard deviation | part1.ipynb; statistics() |
| Skewness and choice of summary | part1.ipynb; statistics(); project report |
| Mean word count by sentiment and interpretation | statistics(); project report |
| Duplicate row count | part1.ipynb; statistics() |
| HTML stripping, lowercase, word_tokenize, preserved negation, alphabetic tokens, lemmatization | clean_review(); part2_cleaning.ipynb |
| Reusable resources and timed 1,000-review comparison | module-level resources; cleaning_demo(); console output |
| Raw-to-lemmatized example including stemming and justification | cleaning_demo(); part2_cleaning.ipynb |
| Whole-column cleaning and empty count | clean_corpus(); console output |
| Stratified split before vectorization; 20,000 then full-corpus experiment | run(); evaluate(); part3_classification.ipynb |
| Required TF-IDF and Naive Bayes settings | evaluate() |
| Accuracy versus 0.500, full report, confusion matrix | results/console_output.txt and experiment metrics |
| 20 positive and 20 negative terms from feature_log_prob_; bigram examples | evaluate(); results/*_terms.csv; console output |
| Python file with explanatory docstrings/comments | src/sentiment_classifier.py |
| Saved console output | results/console_output.txt |
| Paragraph naming two limitations | README.md; docs/project_report.md |

The assignment numbering skips item 6. No extra requirement is inferred from that numbering gap.
