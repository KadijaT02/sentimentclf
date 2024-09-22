#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:App:       Sentiment Classification
:Purpose:   Handling of the experiments.

            Please refer to the :class:`~Experiments` class for further
            details.
:Platform:  Linux/Windows | Python 3.6+
:Developer: K Touré
:Email:     tourekadija02@outlook.com
:Comments: n/a

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# used to prevent Warning: oneDNN custom operations are on. You may see
# slightly different numerical results due to floating-point round-off
# errors from different computation orders. To turn them off, set the
# environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

import keras
import numpy as np
import pandas as pd
import re
from collections import defaultdict
from glob import glob
from keras import Input, Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, Flatten
from keras.utils import pad_sequences, set_random_seed


class Experiments:
    """Handling of the experiments.

    Args:
        x_train (np.ndarray): The films reviews in the training set.
        y_train (np.ndarray): Sentiment classification of the film
            reviews in the training set.
        x_test (np.ndarray): The film reviews in the test set.
        y_test (np.ndarray): Sentiment classification of the film
            reviews in the test set.
        max_review_lengths (list): A list of maximum lengths to try out
            during the experiments.
        trunc_types (list): A list of truncating types to try out during
         the experiments.

    """
    _PATH_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    _PATH_OUTPUT = os.path.join(_PATH_ROOT, 'output')
    _PATH_EXPERIMENTS = os.path.join(_PATH_OUTPUT, 'experiments')

    def __init__(self, x_train, y_train, x_test, y_test, max_review_lengths, trunc_types):
        """Initialise the instance of the class."""
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        # set parameters for the Keras model
        # -- for model training
        self._epochs = 4
        self._batch_size = 128
        # -- for vector space embedding
        self._n_unique_words = 5000
        self._n_dim = 64
        self._pad_type = 'pre'
        self._grid = ((maxlen, trunc) for maxlen in max_review_lengths for trunc in trunc_types)
        # for dense neural network architecture
        self._n_dense = 64
        self._dropout = 0.5

    def run(self):
        """Main program entry point and callable.

        The processing is the following:

            - Loop through the grid of parameters (i.e. the grid
            consisting of maximum lengths and truncating types to try
            out during the experiments).
            - For each set of parameters, standardise the film reviews
            using those.
            - Build and compile the model.
            - Fit the model.

        Returns:
            (bool): True if successful, False otherwise.

        """
        success = False
        try:
            print('Start of the experiments:')
            # loop through the grid of parameters
            for maxlen, trunc in self._grid:
                # create output directory
                output_dir = os.path.join(self._PATH_EXPERIMENTS, f'maxlen{maxlen}_' + trunc + 'trunc')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # standardise the film reviews
                x_train_std, x_test_std = self._standardise(
                    maxlen=maxlen,
                    trunc=trunc
                )
                # build and compile the model
                model = self._build_compile_model(maxlen=maxlen)
                # create ModelCheckpoint
                modelcheckpoint = ModelCheckpoint(
                    filepath=os.path.join(output_dir, "{epoch:02d}-{val_accuracy:.2f}.keras"),
                    monitor='val_accuracy',
                    save_best_only=True
                )
                # fit model
                print(f'Start of the fitting of the model: max_review_length={maxlen}'
                      f' and trunc_type=' + trunc)
                model.fit(
                    x=x_train_std,
                    y=self._y_train,
                    batch_size=self._batch_size,
                    epochs=self._epochs,
                    verbose='auto',
                    callbacks=[modelcheckpoint],
                    validation_split=0.2
                )
            print('- experiments completed.')
            success = True
        except Exception as err:
            print(err)
        return success

    def get_results(self):
        """Return a table of the results.

        The table consists of 4 columns:

            - maxlen: the chosen maximum length for the standardisation
            of the reviews.
            - trunc: the chosen truncating type for the standardisation
            of the reviews.
            - epoch: the epoch number.
            - val_acc: the validation accuracy.

        Returns:
            (pd.DataFrame): The results in table format.

        """
        # retrieve the results
        data = defaultdict(list)
        exp_fpaths = glob(os.path.join(self._PATH_EXPERIMENTS, '*/*.keras'))
        for f in exp_fpaths:
            model_file = os.path.basename(f)
            dir_name = os.path.basename(os.path.dirname(f))
            epoch, val_acc = [s for s in re.split('-|.keras', model_file) if s]
            maxlen, trunc = [s for s in re.split('maxlen|_|trunc', dir_name) if s]
            data['maxlen'].extend([maxlen])
            data['trunc'].extend([trunc])
            data['epoch'].extend([epoch])
            data['val_acc'].extend([val_acc])
        # populate the table
        results = pd.DataFrame(data)
        return results

    def _standardise(self, maxlen, trunc):
        """Standardise the length of the reviews.

        Args:
            maxlen (int): The maximum length.
            trunc (str): The truncating type.

        Return:
            x_train_std (np.ndarray): The standardised film reviews in
                the training set.
            x_test_std (np.ndarray): The standardised film reviews in
                the test set.

        """
        # standardise the film reviews in the training set
        x_train_std = pad_sequences(
            self._x_train,
            maxlen=maxlen,
            padding=self._pad_type,
            truncating=trunc,
            value=0
        )
        # standardise the film reviews in the test set
        x_test_std = pad_sequences(
            self._x_test,
            maxlen=maxlen,
            padding=self._pad_type,
            truncating=trunc,
            value=0
        )
        return x_train_std, x_test_std

    def _build_compile_model(self, maxlen: int):
        """Create and return Keras model.

        Arg:
            maxlen (int): The maximum length of the reviews.

        Return:
            model (Sequential): The built and compiled model.

        """
        model = None
        try:
            # set random seed to make any Keras program deterministic
            set_random_seed(42)
            # architecture of the sentiment classifier
            model = Sequential()
            # -- specify the initial `Input`
            model.add(Input(shape=(maxlen,)))  # shape of the input data
            # -- add embedding layer
            model.add(
                Embedding(
                    input_dim=self._n_unique_words,  # size of the vocabulary
                    output_dim=self._n_dim  # dimension of the dense embedding
                )
            )
            # -- flatten the input
            model.add(Flatten())
            # -- add dense layer
            model.add(
                Dense(
                    units=self._n_dense,  # dimensionality of the output space
                    activation='relu'
                )
            )
            # -- apply dropout
            model.add(Dropout(rate=self._dropout))
            # -- add output dense layer
            model.add(
                Dense(
                    units=1,  # dimensionality of the output space
                    activation='sigmoid'
                )
            )
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        except Exception as err:
            print(err)
        return model
