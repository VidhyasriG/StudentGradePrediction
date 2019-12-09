# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:43:14 2019

@author: vidhy
"""

import upload
from upload import layout
import helper
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html


helper.app.layout = upload.layout 

if __name__ == '__main__':
    helper.app.run_server(debug=True)