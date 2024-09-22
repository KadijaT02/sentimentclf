#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:App:       Sentiment Classification - The web application
:Purpose:   Creation of the blueprint "pages".
:Platform:  Linux/Windows | Python 3.6+
:Developer: K Touré
:Email:     tourekadija02@outlook.com
:Comments: n/a

"""

import os
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request
from sqlalchemy import create_engine
from keras.saving import load_model
from utils import create_idx_word
import json
import plotly


# useful paths
_PATH_ROOT = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
_PATH_OUTPUT = os.path.join(_PATH_ROOT, 'output')
_PATH_STATIC = os.path.join(_PATH_ROOT, 'mywebapp/static')

# create blueprints
bp = Blueprint("pages", __name__)


@bp.route("/", methods=["GET"])
def home():
    return render_template("pages/home.html")


@bp.route("/project", methods=["GET"])
def project():
    # retrieve the Figure objects
    fig1 = plotly.io.read_json(os.path.join(_PATH_STATIC, 'trainset_boxplot.json'))
    fig2 = plotly.io.read_json(os.path.join(_PATH_STATIC, 'trainset_pie.json'))
    # JSON serialise the Figure object
    fig1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    fig2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template("pages/project.html",
                           fig1JSON=fig1JSON,
                           fig2JSON=fig2JSON)


@bp.route("/classifier", methods=["GET", "POST"])
def classifier():
    # retrieve the Figure object
    fig1 = plotly.io.read_json(os.path.join(_PATH_STATIC, 'testset_hist.json'))
    # JSON serialise the Figure object
    fig1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    if request.method == "GET":
        return render_template('pages/classifier.html', fig1JSON=fig1JSON)
    if request.method == "POST":
        # retrieve user input `N`
        N = request.form["N"]
        if N:
            # convert `N` from string to integer
            N = int(N)
            # retrieve data from the database
            engine = create_engine(f"sqlite:///{os.path.join(_PATH_ROOT, 'mywebapp/database.db')}")
            df = pd.read_sql_table(table_name='database', con=engine)
            # -- retrieve the full review
            full_review = df.at[N-1, 'full_x_test']
            full_review = list(map(int, full_review.split()))
            # -- decode the review
            idx_word = create_idx_word()
            decoded_review = ' '.join(idx_word[idx] for idx in full_review)
            # make prediction on the review
            # -- load the model
            model = load_model(os.path.join(_PATH_ROOT, 'output/hyptuning/best_model.keras'))
            # -- make prediction
            review = df.at[N-1, 'x_test_exp18std']
            review = np.frombuffer(review, dtype=int)
            review = np.reshape(review, (1, -1))
            prediction = round(model.predict(review, verbose=0)[0][0], 2)
            return render_template('pages/classifier.html',
                                   fig1JSON=fig1JSON,
                                   decoded_review=decoded_review,
                                   prediction=prediction)
