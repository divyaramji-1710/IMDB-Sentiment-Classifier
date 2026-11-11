"""Explore IMDB reviews, preserve negation, and evaluate TF-IDF Naive Bayes."""
from pathlib import Path
import argparse
import contextlib
import html
import json
import re
import sys
from time import perf_counter

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
NEGATIONS = {'not', 'no', 'never', 'nor', 'neither', 'cannot'}
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = None


def prepare_resources():
    """Initialize reusable resources after the documented NLTK download step."""
    global STOP_WORDS
    STOP_WORDS = set(stopwords.words('english')) - NEGATIONS
    LEMMATIZER.lemmatize('movies')
    word_tokenize('A short review.')


def strip_html(text):
    """Separate HTML-adjacent words and decode entities before tokenization."""
    return html.unescape(re.sub(r'<[^>]*>', ' ', str(text)))


def tokens_for_review(text):
    """Expand negative contractions so alphabetic filtering retains their meaning."""
    text = strip_html(text).lower().replace('\u2019', "'")
    text = re.sub(r"\bwon't\b", 'will not', text)
    text = re.sub(r"\bcan't\b", 'cannot', text)
    text = re.sub(r"n't\b", ' not', text)
    # Treebank splits 'cannot' into 'can not'; both preserve negative meaning.
    return word_tokenize(text)


def clean_review(text):
    """Keep negation while removing HTML, stop words, and nonalphabetic tokens."""
    if STOP_WORDS is None:
        raise RuntimeError('Call prepare_resources() before cleaning reviews.')
    return ' '.join(LEMMATIZER.lemmatize(t) for t in tokens_for_review(text)
                    if t.isalpha() and t not in STOP_WORDS)


def clean_review_repeated_setup(text):
    """Comparison implementation: recreate the same resources for each review."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) - NEGATIONS
    return ' '.join(lemmatizer.lemmatize(t) for t in tokens_for_review(text)
                    if t.isalpha() and t not in stop_words)


def load_reviews():
    """Require the assignment's two columns and valid binary labels."""
    df = pd.read_csv(ROOT / 'data' / 'IMDB Dataset.csv')
    if df[['review', 'sentiment']].isna().any().any():
        raise ValueError('Missing reviews or sentiment labels.')
    if set(df.sentiment.unique()) != {'negative', 'positive'}:
        raise ValueError('Unexpected sentiment labels.')
    return df


def statistics(df):
    """Report the corpus profile before removing duplicate review texts."""
    counts = df.sentiment.value_counts()
    words = df.review.str.split().str.len()
    bands = pd.cut(words, [0,20,40,60,80,100,150,200,300,500,np.inf], right=False)
    print('PART 1: CORPUS STATISTICS')
    print('Rows:', len(df))
    print('Label counts:\n', counts.to_string())
    print('Label percentages:\n', (counts / len(df) * 100).to_string())
    print('Majority-class baseline:', counts.max()/len(df))
    print('Word count: mean={:.2f}, median={:.2f}, mode band={}, variance={:.2f}, std={:.2f}, skew={:.4f}'.format(
        words.mean(), words.median(), bands.value_counts().idxmax(), words.var(), words.std(), words.skew()))
    print('The right tail pulls up the mean; the median better describes typical review length.')
    means = words.groupby(df.sentiment).mean()
    print('Mean word counts by sentiment:\n', means.to_string())
    print('Difference (positive minus negative):', means['positive']-means['negative'])
    print('The roughly three-word difference is small relative to the 171-word overall standard deviation; length alone is a weak practical separator. This is not a significance test.')
    print('Duplicate rows:', df.duplicated().sum())


def cleaning_demo(df):
    """Show each required stage and measure equivalent cleaners on 1,000 rows."""
    raw = df.review.iloc[1]
    tokens = tokens_for_review(raw)
    filtered = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]
    stemmer = PorterStemmer()
    for label, value in [('Raw', raw), ('HTML stripped', strip_html(raw)),
                         ('Tokenized (lowercased, contractions expanded)', tokens),
                         ('Stop words removed; alphabetic only', filtered),
                         ('Stemmed', [stemmer.stem(t) for t in filtered]),
                         ('Lemmatized', [LEMMATIZER.lemmatize(t) for t in filtered])]:
        print('\n'+label+':', value)
    print('Lemmatization retains dictionary forms, making features easier to interpret than truncated stems. The final pipeline uses WordNet noun lemmatization consistently, although it does not resolve every verb form without POS tags.')
    sample = df.review.iloc[:1000]
    times = {}
    expected = None
    for name, cleaner in [('reused_resources', clean_review), ('repeated_setup', clean_review_repeated_setup)]:
        runs = []
        for _ in range(3):
            start = perf_counter()
            values = sample.map(cleaner)
            runs.append(perf_counter()-start)
            if expected is None:
                expected = values
            assert values.equals(expected)
        times[name] = float(np.median(runs))
    print('1,000 reviews, median of three warmed runs (seconds):', times)
    print('Repeated minus reused seconds:', times['repeated_setup']-times['reused_resources'])
    print('Timing depends on hardware and cached resources.')
    return times


def clean_corpus(df):
    """Clean all rows, then prevent identical texts crossing the train/test boundary."""
    cleaned = df.copy()
    cleaned['cleaned_review'] = cleaned.review.map(clean_review)
    print('Reviews cleaned to an empty string:', int(cleaned.cleaned_review.eq('').sum()))
    usable = cleaned[cleaned.cleaned_review.ne('')].copy()
    conflicts = usable.groupby('cleaned_review').sentiment.nunique()
    ambiguous = set(conflicts[conflicts > 1].index)
    print('Conflicting cleaned texts excluded:', len(ambiguous))
    usable = usable[~usable.cleaned_review.isin(ambiguous)]
    before = len(usable)
    usable = usable.drop_duplicates('cleaned_review').reset_index(drop=True)
    print('Repeated cleaned texts removed:', before-len(usable))
    print('Usable unique reviews:', len(usable))
    return cleaned, usable


def evaluate(df, name, results_dir):
    """Fit vocabulary only on training reviews, then evaluate the untouched split."""
    train, test = train_test_split(df, test_size=0.2, stratify=df.sentiment, random_state=42)
    assert not set(train.cleaned_review) & set(test.cleaned_review)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.8,
                                max_features=50000, sublinear_tf=True)),
        ('nb', MultinomialNB(alpha=0.5))])
    pipeline.fit(train.cleaned_review, train.sentiment)
    predictions = pipeline.predict(test.cleaned_review)
    accuracy = accuracy_score(test.sentiment, predictions)
    report = classification_report(test.sentiment, predictions, digits=4)
    matrix = confusion_matrix(test.sentiment, predictions, labels=['negative','positive'])
    print('\nEXPERIMENT:', name, 'train:', len(train), 'test:', len(test))
    print('Accuracy:', accuracy, '| assignment baseline: 0.500 | improvement:', accuracy-0.5)
    print('Training-majority baseline on test:', float((test.sentiment == train.sentiment.mode().iloc[0]).mean()))
    print(report)
    print('Confusion matrix: rows=true, columns=predicted; negative, positive\n', matrix)
    vectorizer, model = pipeline.named_steps['tfidf'], pipeline.named_steps['nb']
    terms = vectorizer.get_feature_names_out()
    positive = list(model.classes_).index('positive')
    negative = list(model.classes_).index('negative')
    # Log probability difference measures class association, rather than frequency alone.
    scores = model.feature_log_prob_[positive]-model.feature_log_prob_[negative]
    tables = {}
    for label, order in [('positive', np.argsort(scores)[::-1]), ('negative', np.argsort(scores))]:
        top = order[:20]
        table = pd.DataFrame({'term':terms[top], 'positive_minus_negative_log_probability':scores[top]})
        print('\n20 most '+label+' terms:\n'+table.to_string(index=False))
        print('Bigrams in top 20:', sum(' ' in t for t in terms[top]))
        # Explicitly show bigram evidence even if fewer than three land in the overall top 20.
        bigrams = [i for i in order if ' ' in terms[i]][:3]
        print('Three strongest '+label+' bigrams:', terms[bigrams].tolist())
        table.to_csv(results_dir / f'{name}_{label}_terms.csv', index=False)
        tables[label] = table.to_dict('records')
    metrics = {'experiment':name,'train_rows':len(train),'test_rows':len(test),
               'accuracy':accuracy,'assignment_baseline':0.5,'confusion_matrix':matrix.tolist(),
               'classification_report':classification_report(test.sentiment,predictions,output_dict=True),
               'top_terms':tables}
    (results_dir/f'{name}_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (results_dir/f'{name}_classification_report.txt').write_text(report,encoding='utf-8')
    return pipeline, metrics


def run():
    """Run the required sample experiment before the full-corpus experiment."""
    prepare_resources()
    results = ROOT/'results'
    results.mkdir(exist_ok=True)
    df = load_reviews()
    statistics(df)
    cleaning_demo(df)
    _, usable = clean_corpus(df)
    sample, _ = train_test_split(usable, train_size=20000, stratify=usable.sentiment, random_state=42)
    evaluate(sample, 'sample_20000', results)
    pipeline, _ = evaluate(usable, 'full_corpus', results)
    (ROOT/'models').mkdir(exist_ok=True)
    joblib.dump(pipeline, ROOT/'models'/'sentiment_pipeline.joblib')
    print('\nNegation examples (illustrations, not an additional evaluation set):')
    for review in ['This movie is good.', 'This movie is not good.', 'I never enjoyed this film.', 'I cannot recommend this movie.']:
        cleaned = clean_review(review)
        print(review, '=>', cleaned, '=>', pipeline.predict([cleaned])[0])
    print('\nLimitations: Bag-of-words and bigrams can miss sarcasm and long-range negation. Performance on held-out IMDB reviews may not transfer to other domains, languages, or recent review styles.')


class Tee:
    """Mirror console output so the submission contains the actual run log."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)
    def flush(self):
        for stream in self.streams:
            stream.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results'/'console_output.txt')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)):
            run()
