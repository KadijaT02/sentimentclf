#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:App:       Sentiment Classification - The web application
:Purpose:   Creation of the application factory.
:Platform:  Linux/Windows | Python 3.6+
:Developer: K Touré
:Email:     tourekadija02@outlook.com
:Comments: n/a

"""

import os
from flask import Flask

import pages


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# used to prevent Warning: oneDNN custom operations are on. You may see
# slightly different numerical results due to floating-point round-off
# errors from different computation orders. To turn them off, set the
# environment variable `TF_ENABLE_ONEDNN_OPTS=0`.


# create application factory
def create_app():
    """Initialise and return the app.

    Returns:
        app (Flask object): The application.
    """
    app = Flask(__name__)
    app.register_blueprint(pages.bp)
    return app
