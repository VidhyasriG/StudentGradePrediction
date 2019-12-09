# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:28:18 2019

@author: vidhy
"""

import dash
import dash_bootstrap_components as dbc
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                        'https://codepen.io/bjpranav/pen/vqmROq.css',
                        'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css']


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
body = {'backgroundColor':'#efefef'}
app.config.suppress_callback_exceptions = True