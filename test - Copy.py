import numpy as np
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, dash_table
from dash import html
from dash import Input, Output
from PreProcessing import *
import plotly.graph_objects as go
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import boxcox

df_satisfied = df_2[df_2['satisfaction'] == 'satisfied']
df_unsatisfied = df_2[df_2['satisfaction'] != 'satisfied']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash("Moon Knight", external_stylesheets=external_stylesheets)

my_app.layout = html.Div(
    [
        html.H1("Moon Knight Air", style={'textAlign': 'center'}),

        dcc.Tabs(id='main-tabs',
                 children=[
                     dcc.Tab(label='Data Overview', value='dove'),
                     dcc.Tab(label='Exploratory Data Analysis', value='eda',
                             children=[dcc.Tabs(id="sub-tabs1"
                                                , children=[
                                     dcc.Tab(label="Correlation", value="cor"),
                                     dcc.Tab(label="Principal Component Analysis", value="pca"),
                                     dcc.Tab(label="Outlier Detection and Removal", value="outlier"),
                                     dcc.Tab(label="Normality Test", value="dnorm")
                                 ]
                                                )
                                       ]
                             ),
                     dcc.Tab(label='Pre-board', value='pb'),
                     dcc.Tab(label='On-board', value='ob'),
                     dcc.Tab(label='Post-flight', value='pf')
                 ]
                 , colors={"border": "black", "primary": "gold", "background": "cornsilk"}
                 ),

        html.Div(id='layout'),

    ]
)

pre_board_flight_layout = html.Div(
    [
        html.H3("Pre-Boarding Factors", style={'textAlign': 'center'}),
        html.Br(),

        html.H4("Choose a factor: "),
        dcc.Dropdown(id="drop-pre",
                     options=[
                         {'label': 'Ease of Online booking', 'value': 'Ease of Online booking'},
                         {'label': 'Gate location', 'value': 'Gate location'},
                         {'label': 'Online boarding', 'value': 'Online boarding'},
                         {'label': 'Checkin service', 'value': 'Checkin service'},
                         {'label': 'Departure Delay', 'value': 'Departure Delay'},
                         {'label': 'Departure/Arrival Time Convenient', 'value': 'Departure/Arrival time convenient'}
                     ],
                     value="Online boarding"
                     ),
        html.Br(),

        html.Div([
            html.H5("Customer Satisfaction Level:"),
            dcc.Checklist(
                id='x-axis2',
                options=['Satisfied', 'Not Satisfied'],
                value=['Satisfied', 'Not Satisfied'],
                inline=True
            )
        ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}),

        html.Div([
            html.H5("Customer Demographic:"),
            dcc.RadioItems(
                id='y-axis2',
                options=['Gender', 'Customer Type', 'Type of Travel', 'Class'],
                value='Gender',
                inline=True
            )], style={"width": '50%', "margin": 0, 'display': 'inline-block'}),

        html.Br(),

        html.Div(
            [
                dcc.Graph(id="graph-pb1")
            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        ),

        html.Div(
            [
                dcc.Graph(id="graph-pb2"),
                html.Div(id='txt2')
            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        )

    ]
)


@my_app.callback(
    Output(component_id='layout', component_property='children'),
    [Input(component_id='main-tabs', component_property='value'),
     Input(component_id='sub-tabs1', component_property='value')]
)
def update_layout(tab, subtab1):
    if tab == "pb":
        return pre_board_flight_layout


@my_app.callback(
    [
        Output(component_id='graph-pb1', component_property='figure'),
        Output(component_id='graph-pb2', component_property='figure'),
        Output(component_id='txt2', component_property='children')
    ],
    [
        Input(component_id='drop-pre', component_property='value'),
        Input(component_id='x-axis2', component_property='value'),
        Input(component_id='y-axis2', component_property='value')
    ]
)
def pb_input(dr, xa2, ya2):
    if xa2 == ['Satisfied']:
        df1 = df_satisfied.copy()
    elif xa2 == ['Not Satisfied']:
        df1 = df_unsatisfied.copy()
    else:
        df1 = df_2.copy()

    df1_hist = df1.copy()

    if dr != 'Departure Delay':
        df1[dr] = np.where(df1[dr] == 0, "Very Bad"
                           , np.where(df1[dr] == 1, "Bad"
                                      , np.where(df1[dr] == 2, "Average"
                                                 , np.where(df1[dr] == 3, "Good"
                                                            , np.where(df1[dr] == 4, 'Very Good'
                                                                       , np.where(df1[dr] == 5, 'Excellent'
                                                                                  , 'Average'))))))

    fig = px.pie(df1, dr)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig['layout'].update({
        'title': f'Data Distribution - {dr}'
    })

    fig1 = px.histogram(df1_hist, x=dr, color=ya2, barmode='group')

    fig1['layout'].update({
        'title': f'{dr} v/s {ya2}',
        'xaxis': {
            'title': f'{dr} -> (0: VeryBad, 1: Bad, 2: Average, 3: Good, 4: VeryGood, 5: Excellent)',
            'zeroline': False
        }
    })

    fig1.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig1.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)

    if dr == 'Departure Delay':
        fig1 = px.histogram(df1_hist, x=dr, color='satisfaction', barmode='group')
        txt = "NOTE: DEPARTURE DELAY WILL ONLY BE SHOWN w.r.t SATISFACTION"
    else:
        txt = ""

    return fig, fig1, txt


my_app.run_server(debug=True, port=8000)
