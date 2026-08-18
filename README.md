# IMDB Sentiment Classifier

## Project Description

This project develops a machine learning based sentiment classifier using the IMDB movie review dataset.

The objective is to classify movie reviews into two sentiment classes:

- Positive
- Negative

The project begins with statistical exploration of the dataset and will progress through preprocessing, model development, and evaluation.

---

## Project Structure

    IMDB-Sentiment-Classifier/
    |
    +-- data/
    |   +-- IMDB Dataset.csv
    |
    +-- notebooks/
    |   +-- Part_1_Statistics.ipynb
    |
    +-- src/
    |
    +-- models/
    |
    +-- results/
    |
    +-- requirements.txt
    +-- README.md
    +-- .gitignore

---

## Environment

Python 3.11 or compatible Python version.

The project is developed using Jupyter Notebook.

---

## Setup

### 1. Create a virtual environment

    python -m venv .venv

### 2. Activate the virtual environment

Windows PowerShell:

    .venv\Scripts\Activate.ps1

### 3. Install dependencies

    pip install -r requirements.txt

### 4. Register the Jupyter kernel

    python -m ipykernel install --user --name imdb-sentiment --display-name "IMDB Sentiment"

### 5. Open the notebook

Open:

    notebooks/Part_1_Statistics.ipynb

Select the "IMDB Sentiment" Jupyter kernel.

---

## Dataset

The project uses the IMDB movie review dataset.

Dataset location:

    data/IMDB Dataset.csv

Expected columns:

- review
- sentiment

---

## Project Progress

### Part 1 — Load and Explore (Statistics)

#### Step 1 — Dataset statistics

- Load IMDB Dataset.csv
- Report row count
- Report sentiment label counts
- Report label percentages
- Calculate majority-class baseline

#### Step 2 — Word-count statistics

- Add word_count column
- Calculate mean word count
- Calculate median word count
- Calculate mode band
- Calculate variance
- Calculate standard deviation

#### Step 3 — Distribution analysis

- Compare mean and median word count
- Calculate skewness
- Determine the direction of skewness
- Explain what skewness implies
- Determine which statistic is more appropriate to quote

#### Step 4 — Mean Word Count by Sentiment

- Calculate mean word count separately for positive and negative reviews
- Compare the average review length between sentiment classes
- Calculate the difference between positive and negative mean word counts

#### Step 5 — Duplicate Row Check

- Check for duplicate rows using `duplicated().sum()`
- Report the number of duplicate rows found
- Verify whether duplicate records are present in the dataset

---

## Collaboration

Before starting work:

    git pull

After completing work:

    git add .
    git commit -m "Describe the work completed"
    git push

Each contributor should create their own virtual environment.

The .venv directory must not be committed to Git.

---

## Important

Use relative paths in notebooks rather than personal computer-specific paths.

For example:

    data/IMDB Dataset.csv

rather than:

    C:/Users/username/Desktop/IMDB-Sentiment-Classifier/data/IMDB Dataset.csv
