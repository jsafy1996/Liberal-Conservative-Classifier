"""
This module contains classes and helper functions 
for running various types of neural networks.
"""

import pickle
import sqlite3

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, train_test_split

from keras.initializers import Constant
from keras.layers import (Activation, Conv1D, CuDNNLSTM, Dense, Dropout,
                          Embedding, GlobalMaxPooling1D, Input, MaxPooling1D)
from keras.models import Model, Sequential, load_model
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)

INPUT_GLOVE = "PATH/TO/GLOVE/glove.6B.100d.txt"


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


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


class MLP:
    """
    This class handles multilayer perceptrons.

    Parameters
    ----------
    datasets (list[str])
    split (float)
    batch_size (int)
    epochs (int)
    num_splits (int)
    num_layers (list[int])
    num_nodes (list[int])
    num_dropouts (list[int])
    """

    def __init__(
        self,
        datasets,
        split=0.2,
        batch_size=128,
        epochs=10,
        num_splits=1,
        num_layers=[1, 2],
        num_nodes=[32, 64],
        num_dropouts=[0, 20]
    ):
        assert type(num_layers).__name__ == type(num_nodes).__name__ == type(num_dropouts).__name__ == "list", "Please enter lists for num_layers, num_nodes, and/or num_dropouts" 
        self.datasets = datasets
        self.split = split
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_splits = num_splits
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.num_dropouts = num_dropouts

    def clean_data(self, df, db_name, sparse=True):
        """
        Cleans and processes data for training.

        Parameters
        ----------
        df (Pandas.DataFrame): Data to be processed. 
        db_name (str): Name of dataset to be processed.
        sparse (bool):
            Whether the data has additional features or not.
            If not, the model will perform simple bag of words,
            else it will include those features in the training.

        Returns
        -------
        X_train (Pandas.DataFrame): The training data.
        y_train (Pandas.DataFrame): The training data labels.
        X_test (Pandas.DataFrame): The test data.
        y_test (Pandas.DataFrame): The test data labels.
        input_shape (int): Size of the input the model will expect.
        output_shape (int) Number of unqiue labels.
        sparse (bool):
            Whether the data has additional features or not.
            If not, the model will perform simple bag of words,
            else it will include those features in the training.
        """

        print("Cleaning data")

        # Generates tokenizer and counts matrix.
        X_text = df["text"]
        print("Fitting tokenizer")
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X_text)
        with open(
            f"./models/tokenizer_{db_name}.pickle", "wb"
        ) as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("DONE")

        print("Generating count matrix")
        X_counts = tokenizer.texts_to_matrix(X_text, mode="tfidf")
        with open(
            f"./models/counts_{db_name}.pickle", "wb"
        ) as handle:
            pickle.dump(X_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("DONE")

        # Loads prebuilt tokenizer and counts matrix. Use if available, and comment out above block of code.
        # print("Loading tokenizer")
        # tokenizer = Tokenizer()
        # with open(f"./models/tokenizer_{db_name}.pickle", "rb") as handle:
        #         tokenizer = pickle.load(handle)
        # print("DONE")

        # print("Loading counts matrix")
        # with open(f"./models/counts_{db_name}.pickle", "rb") as handle:
        #     X_counts = pickle.load(handle)
        # print("DONE")

        if not sparse:
            if "X_counts" not in locals():
                X_counts = pd.DataFrame()
            print("Appending values")

            # Enter the columns of desired features.
            other_vars = df[df.columns[-10:]]

            X_counts = pd.DataFrame.to_dense(X_counts)
            X_counts = pd.DataFrame(X_counts)
            X_counts = pd.concat((X_counts, other_vars), axis=1)
            print("DONE")

        y = df["label"]
        output_shape = len(y.unique())
        if output_shape > 2:
            y = to_categorical(y, output_shape)

        print("Generating train-test split")
        X_train, X_test, y_train, y_test = train_test_split(
            X_counts,
            y,
            test_size=self.split,
        )
        print("DONE")

        input_shape = X_train.shape[1]
        print(input_shape, output_shape)
        return (
            X_train,
            X_test,
            y_train,
            y_test,
            input_shape,
            output_shape,
            sparse,
        )

    def mlp(self, num_layer, num_node, num_dropout, input_shape, output_counts):
        """
        Create user-specified multilayer perceptron.

        Parameters
        ----------
        num_layer (int): Number of hidden layers in the model.
        num_node (int): Number of nodes in each hidden layer.
        num_dropout (int): Dropout percentage per hidden layer.
        input_shape (int): Size of input layer.
        output_counts (int): Size out output layer.

        Returns
        -------
        model (Keras.Sequential): A sequential model as per user specifications.
        """

        model = Sequential()
        model.add(Dense(num_node, input_shape=(input_shape,), activation="relu"))
        model.add(Dropout(num_dropout))
        for layer in range(num_layer):
            model.add(Dense(num_node, activation="relu"))
            model.add(Dropout(num_dropout))

        if output_counts > 2:
            model.add(Dense(output_counts, activation="softmax"))
            model.compile(
                loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
            )
        else:
            model.add(Dense(1, activation="sigmoid"))
            model.compile(
                loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
            )

        return model

    def train(self, model, X_train, y_train, X_test, y_test, k_folds):
        """
        Train a user-specified multilayer perceptron.

        Parameters
        ----------
        model (Keras.Sequential): A user-specified model.
        X_train (Pandas.DataFrame): The training data.
        y_train (Pandas.DataFrame): The training data labels.
        X_test (Pandas.DataFrame): The test data.
        y_test (Pandas.DataFrame): The test data labels.
        k_folds (bool): Whether to use stratified k-fold cross validation.

        Returns
        -------
        history (Keras.History): Record of training metrics.
        score (int or list): Values of metric(s) tracked during training.
        """

        if k_folds:
            history = model.fit(
                X_train, y_train, epochs=self.epochs, batch_size=self.batch_size
            )
        else:
            # tensorboard = TensorBoard(log_dir=f"./logs/{name}")
            early_stopping = EarlyStopping(monitor="val_loss")
            # reduce = ReduceLROnPlateau(monitor='val_loss', factor=.2, patience=3, min_lr=.001)
            history = model.fit(
                X_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.split,
                callbacks=[early_stopping],
            )

        score = model.evaluate(X_test, y_test, batch_size=self.batch_size)
        print(score)
        return history, score

    def mlp_wrapper(self):
        """
        Run the MLP.
        """

        df, db_name = import_data(self.datasets)
        X_train, X_test, y_train, y_test, input_shape, output_shape, sparse = self.clean_data(
            df, db_name, sparse=False
        )

        count = 1
        total = len(self.num_layers) * len(self.num_nodes) * len(self.num_dropouts)
        best_score = 0
        best_model = ""
        best_model_name = ""

        for num_layer in self.num_layers:
            for num_node in self.num_nodes:
                for num_dropout in self.num_dropouts:
                    name = f"{num_layer}_layer_{num_node}_node_{num_dropout}%_dropout_{db_name}"
                    print(name, f"{count} out of {total}")
                    count += 1
                    if self.num_splits > 1:
                        skf = StratifiedKFold(n_splits=self.num_splits, shuffle=True)
                        for i, (train_indices, val_indices) in enumerate(
                            skf.split(X_train, y_train)
                        ):
                            print(train_indices, val_indices, type(train_indices))
                            print(f"Training on fold {i + 1} out of {self.num_splits}")
                            X_train_split, X_test_split = (
                                X_train[train_indices],
                                X_train[val_indices],
                            )
                            y_train_split, y_test_split = (
                                y_train[train_indices],
                                y_train[val_indices],
                            )

                            model = None
                            model = self.mlp(
                                num_layer,
                                num_node,
                                num_dropout,
                                input_shape,
                                output_shape,
                            )

                            history, score = self.train(
                                model,
                                X_train_split,
                                y_train_split,
                                X_test_split,
                                y_test_split,
                                True,
                            )

                            if score[1] > best_score:
                                best_score = score[1]
                                best_model = model
                                best_model_name = name
                    else:
                        model = self.mlp(
                            num_layer, num_node, num_dropout, input_shape, output_shape
                        )

                        history, score = self.train(
                            model, X_train, y_train, X_test, y_test, False
                        )

                        if score[1] > best_score:
                            best_score = score[1]
                            best_model = model
                            best_model_name = name

        # Change to your preferred filepaths.
        if sparse:
            best_model.save(
                f"./models/{best_model_name}_{round(best_score, 3)}_sparse.h5"
            )
        else:
            best_model.save(
                f"./models/{best_model_name}_{round(best_score, 3)}_dense.h5"
            )
        print(f"{best_model_name}_{round(best_score, 3)}")

    def test_mlp(self, model_loc, tokenizer_loc=None):
        """
        Test pretrained models.

        Parameters
        ----------
        model_loc (str): FIlepath of trained model to be tested.
        tokenizer_loc (str): Filepath of saved tokenizer. Requrired for bag of words.
        """

        df, name = import_data(self.datasets)
        model = load_model(model_loc)
        text, other_vars, y = df["text"], df[df.columns[-10:]], df["label"]

        if tokenizer_loc:
            tokenizer = Tokenizer()
            with open(tokenizer_loc, "rb") as handle:
                tokenizer = pickle.load(handle)

            X_counts = tokenizer.texts_to_matrix(text, mode="tfidf")
            X_counts = pd.DataFrame.to_dense(X_counts)
            X_counts = pd.DataFrame(X_counts)
            X_counts = pd.concat((X_counts, other_vars), axis=1)

        if len(y.unique()) > 2:
            y_cats = to_categorical(y, len(y.unique()))

        print(model.predict(X_counts))
        print(
            model.evaluate(
                X_counts,
                y if len(y.unique()) <= 2 else y_cats,
                batch_size=self.batch_size,
            )
        )


# Usage example
if __name__ == "__main__":
    mlp = MLP(["./data/training.csv"])
    mlp.mlp_wrapper()
    mlp = MLP(["./data/validation.csv"])
    mlp.test_mlp("./models/2_layer_32_node_20%_dropout_training_0.914_dense.h5", "./models/tokenizer_training.pickle")
