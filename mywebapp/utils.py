#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:App:       Sentiment Classification - The web application
:Purpose:   Utility functions.
:Platform:  Linux/Windows | Python 3.6+
:Developer: K Touré
:Email:     tourekadija02@outlook.com
:Comments: n/a

"""

from keras.datasets import imdb


def create_idx_word(index_from: int = 3, start_char: int = 1, oov_char: int = 2) -> dict:
    """Create an index-word dictionary for the IMDd dataset.

    The dictionary will map a word index to its associated word. It is built upon the original word-index dictionary provided with the dataset.

    Arg:
        index_from (int): Index actual words with this index and higher.
        start_char (int): The start of a sequence will be marked by this character.
        oov_char (int): The out-of-vocabulary character.

    Returns:
        idx_word (dict): The index word dictionary.

    """
    # retrieve word index dictionary
    word_idx = imdb.get_word_index()
    # reverse the word index dictionary to obtain `dict_idx_word`
    # And add `index_from` to all word indices
    idx_word = dict((i + index_from, word) for (word, i) in word_idx.items())
    # include `start_char` and `oov_char` to the dictionary
    idx_word[start_char] = '[START]'
    idx_word[oov_char] = '[OOV]'
    # return the dictionary
    return idx_word
