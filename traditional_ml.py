"""
This function contains the pipelines, gridsearch parameters, and helper
functions for some scikit-learn functions. Naive Bayes, Random Forest,
and Support Vector Machines are implemented so far.
"""

import pickle
import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def import_data(datasets):
    """
    Accept an array of filepaths to datasets and return
    a DataFrame of those datasets, appended if multiple.

    Parameters
    ----------
    datasets (list[list[str]]):
        List of lists where each inner list has up to two elements:
        - Filepath to a dataset of interest
        - Table name if dataset is in database format. If using
          CSV leave blank or empty.

    Returns
    -------
    df (Pandas.DataFrame): Dataframe oject of all datasets.
    db_name (str): Name of all datasets joined by '_'.
    """

    print("Importing data")
    df = pd.DataFrame()
    db_name = ""
    assert type(datasets).__name__ == "list", "Please enter a list"
    for dataset in datasets:
        data_type = dataset[0][dataset[0].rfind("."):]
        db_name_temporary = dataset[0][dataset[0].rfind("/") + 1: dataset[0].rfind(".")]
        if data_type == ".db":
            cnx = sqlite3.connect(dataset[0])
            df_temporary = pd.read_sql_query(f"SELECT * FROM {dataset[1]}", cnx)
        elif data_type == ".csv":
            df_temporary = pd.read_csv(dataset[0])
        print(db_name_temporary, df_temporary.shape)
        df = df.append(df_temporary, ignore_index=True, sort=False)
        db_name += db_name_temporary + "_"

    # Change to whatever column text is located in.
    # Prevents blank values from causing errors.
    df.text = df.text.astype(str)

    return df, db_name[:-1]


def run_nb(X, y, pipeline, params, filename):
    """
    Run Naive Bayes model on input data.

    Parameters:
    X (Pandas.DataFrame): Training data
    y (Pandas.DataFrame): Labels
    pipeline (sklearn.Pipeline): Pipeline to set up model.
    params: The gridsearch parameters you want to iterate over.
    filename: Where to save the gridsearch model.
    """

    nb_clf = NB.fit(X, y=y)
    gs_nb_clf = GridSearchCV(nb_clf, PARAMS_NB, n_jobs=-1)
    gs_nb_clf = gs_nb_clf.fit(X, y)

    print(f"NB best score: {gs_nb_clf.best_score_}")
    print(f"NB best params: {gs_nb_clf.best_params_}")
    
    pickle.dump(gs_nb_clf, open(filename, "wb"), protocol=pickle.HIGHEST_PRIORITY)


def run_svm(X, y, pipeline, params, filename):
    """
    Run SVM model on input data.

    Parameters:
    X (Pandas.DataFrame): Training data
    y (Pandas.DataFrame): Labels
    pipeline (sklearn.Pipeline): Pipeline to set up model.
    params: The gridsearch parameters you want to iterate over.
    filename: Where to save the gridsearch model.
    """

    svm_clf = SVM.fit(X, y=y)
    gs_svm_clf = GridSearchCV(svm_clf, PARAMS_SVM, n_jobs=-1)
    gs_svm_clf = gs_svm_clf.fit(X, y)

    print(f"SVM best score: {gs_svm_clf.best_score_}")
    print(f"SVM best params: {gs_svm_clf.best_params_}")

    pickle.dump(gs_svm_clf, open(filename, "wb"), protocol=pickle.HIGHEST_PRIORITY)


def run_rf(X, y, pipeline, params, filename):
    """
    Run Random Forest model on input data.

    Parameters:
    X (Pandas.DataFrame): Training data
    y (Pandas.DataFrame): Labels
    pipeline (sklearn.Pipeline): Pipeline to set up model.
    params: The gridsearch parameters you want to iterate over.
    filename: Where to save the gridsearch model.
    """

    rf_clf = RF.fit(X, y=y)
    gs_rf_clf = GridSearchCV(rf_clf, PARAMS_RF, n_jobs=-1)
    gs_rf_clf = gs_rf_clf.fit(X, y)

    print(f"RF best score: {gs_rf_clf.best_score_}")
    print(f"RF best params: {gs_rf_clf.best_params_}")

    pickle.dump(gs_rf_clf, open(filename, "wb"), protocol=pickle.HIGHEST_PRIORITY)


if __name__ == "__main__":
    # Data generated with module at http://github.com/jsafy1996/Social-Media-Scrapers
    df, db_name = import_data(["./tweets_users.csv"])
    X, y = df["text"], df["label"]

    PARAMS_NB = {
        "vect__ngram_range": [(1, 1), (1, 2)],
        "tfidf__use_idf": (True, False),
        "nb_clf__alpha": (1e-2, 1e-3),
    }

    NB = Pipeline(
        [
            ("vect", CountVectorizer(stop_words="english")),
            ("tfidf", TfidfTransformer()),
            ("nb_clf", MultinomialNB(fit_prior=False)),
        ]
    )

    PARAMS_SVM = {
        "vect__ngram_range": [(1, 1), (1, 2)],
        "tfidf__use_idf": (True, False),
        "svm_clf__alpha": (1e-2, 1e-3),
    }

    SVM = Pipeline(
        [
            ("vect", CountVectorizer(stop_words="english")),
            ("tfidf", TfidfTransformer()),
            ("svm_clf", SGDClassifier(max_iter=1000, tol=1e-3)),
        ]
    )

    PARAMS_RF = {
        "vect__ngram_range": [(1, 1), (1, 2)],
        "tfidf__use_idf": (True, False),
        "rf_clf__n_estimators": [64, 128],
    }

    RF = Pipeline(
        [
            ("vect", CountVectorizer(stop_words="english")),
            ("tfidf", TfidfTransformer()),
            ("rf_clf", RandomForestClassifier()),
        ]
    )

    run_nb(X, y, NB, PARAMS_NB, f"./models/NB_{db_name}.sav")
    run_svm(X, y, SVM, PARAMS_SVM, f"./models/SVM_{db_name}.sav")
    run_rf(X, y, RF, PARAMS_RF, f"./models/RF_{db_name}.sav")
