# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:06:03 2019

@author: vidhy
"""

import base64
import io
import helper
import dash
import dash_table as dt
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import xgbregression
import plotly.express as px



layout = html.Div(children=[
    html.H3(children='STUDENT GRADE PREDICTION',
            style={'margin':'21px','padding':'30px', 'backgroundColor':'darkseagreen',
                   'text-align':'center', 'font-family':'initial'}),
    html.Hr(),
    html.H5(children='Term      : Spring 2020', 
            style={'margin-left':'21px', 'margin-right':'21px', 
                   'backgroundColor':'#D3D3D3','text-align':'left', 
                   'font-family':'Sans-serif'}),
    html.H5(children='Course  : XX XXX', 
            style={'margin-left':'21px', 'margin-right':'21px',
                   'backgroundColor':'#D3D3D3','text-align':'left',
                   'font-family':'Sans-serif'}),
    html.Hr(),
    html.H5(children='Upload the data',
            style={'margin-left':'21px', 'margin-right':'21px', 
                   'textColor':'#333','text-align':'left',
                   'font-family':'Sans-serif'}),
    dcc.Upload(
        id='upload-test-data',
        children=(
            'Drag and Drop or ',
            html.A('Select Files')),
        style={
            'width': '20%',
            'margin': '10px',
            'height': '30px',
            'lineHeight': '30px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '3px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-test-upload',
             style={'margin-left':'21px', 'margin-right':'21px', 
                   'text-align':'left', 
                   'font-size':'12px',
                   'font-family':'Sans-serif'}),
    html.Div(),
    html.Button('Predict', id='button',
                style={'margin-left':'21px', 'margin-right':'21px', 
                   'backgroundColor':'#D3D3D3','text-align':'left', 
                   'font-family':'Sans-serif'}),
    html.Hr(),
    html.Div(id='output-grade-table',
             style={'margin-left':'321px', 'margin-right':'321px'}
             )
])

@helper.app.callback(Output('output-test-upload', 'children'),
                     [Input('upload-test-data', 'contents')],
                     [State('upload-test-data', 'filename'),
                      State('upload-test-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            df.to_csv('data.csv')
        elif 'xlsx' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.StringIO(decoded.decode('utf-8')))
            df.to_csv('data.csv')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5('File uploaded successfully!'),
        #  html.H6(datetime.datetime.from timestamp(date)),

        #  dash_table.DataTable(
        #    data=df.to_dict('records'),
        #    columns=[{'name': i, 'id': i} for i in df.columns]
        #  ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        #  html.Div('Raw Content'),
        #  html.Pre(contents[0:200] + '...', style={
        #    'whiteSpace': 'pre-wrap',
        #    'wordBreak': 'break-all'
        #  })
    ])


@helper.app.callback(
    dash.dependencies.Output('output-grade-table', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')])

def update_table(n_clicks):
    if n_clicks!=None:   
        train_df = pd.read_csv('SyntheticData.csv')
        test_df = pd.read_csv('data.csv')
        grades= xgbregression.train_predict(train_df, test_df)
        return dt.DataTable(
            data=grades.to_dict('records'),
            sort_action="native",
            columns=[{'name': i, 'id': i} for i in grades.columns],
            style_cell={'textAlign': 'left','padding': '5px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)','fontWeight': 'bold'},
            style_data_conditional=[{ 'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'},
            {'if':{'column_id':'Grades',
                  'filter_query':'{Grades}<70.0'},
            'backgroundColor': '#F08080'}]  
            ), dcc.Graph( figure=px.histogram(grades, x="Grades", nbins=10, 
                title="Predicted Distribution",labels={'Grades':'Predicted Total'},
                color_discrete_sequence=['indianred']))



    

