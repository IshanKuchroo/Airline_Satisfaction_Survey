##############################################
# Import Libraries #
#############################################

# Standard
import pandas as pd
import numpy as np
import numpy.linalg as LA
from scipy import stats
from scipy.stats import boxcox

# Visualization
from matplotlib import pyplot as plt
import seaborn as sns
import pylab

# scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.express as px
import dash
from dash import dcc, dash_table
from dash import html
from dash import Input, Output
import plotly.graph_objects as go
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import boxcox


import warnings

warnings.catch_warnings()
warnings.simplefilter("ignore")

#############################################################
# ------------------ Reading Dataset ------------------ #
############################################################

df = pd.read_csv("train.csv")

print("Original Dataset: \n", df.head())
print("Dataset Statistics: \n", df.describe())

df_2 = df.copy()

df_2.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)

df_2 = df_2.dropna(how="any")

#####################################################################
# ------------------ Outlier Detection - Removal ------------------ #
####################################################################

df_outlier = df_2.copy()

for x in ['Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']:
    iqr = (df_2[x].quantile(0.75)) - (df_2[x].quantile(0.25))

    upper_limit = (df_2[x].quantile(0.75)) + (1.5 * iqr)
    lower_limit = (df_2[x].quantile(0.25)) - (1.5 * iqr)

    df_2[x] = np.where(df_2[x] > upper_limit, upper_limit, np.where(df_2[x] < lower_limit, lower_limit, df_2[x]))

#####################################################################
# ------------------ Dimensionality Reduction ------------------ #
####################################################################

X = df_2.loc[:, ~df_2.columns.isin(['satisfaction'])]
Y = df_2['satisfaction']


# Handling categorical variables

def mapping(xx):
    dict = {}
    count = -1
    for x in xx:
        dict[x] = count + 1
        count = count + 1
    return dict


# for i in ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'Departure Delay', 'Arrival Delay']:
for i in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
    unique_tag = X[i].value_counts().keys().values
    dict_mapping = mapping(unique_tag)
    X[i] = X[i].map(lambda x: dict_mapping[x] if x in dict_mapping.keys() else -1)

# The StandardScaler
ss = StandardScaler()

# Standardize the training data
X = ss.fit_transform(X)

H = np.matmul(X.T, X)

s, d, v = np.linalg.svd(H, full_matrices=True)

pca = PCA(n_components='mle', svd_solver='full')
# pca = PCA(0.95)
principalComponents = pca.fit_transform(X)

print("#" * 100)

x = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1)

#####################################################################
# ------------------ Correlation Matrix ------------------ #
####################################################################

df_corr = df_2.copy()

for i in ['satisfaction']:
    unique_tag = df_corr[i].value_counts().keys().values
    dict_mapping = mapping(unique_tag)
    df_corr[i] = df_corr[i].map(lambda x: dict_mapping[x] if x in dict_mapping.keys() else -1)

#####################################################################
# ------------------ Statistics ------------------ #
####################################################################

num_col = df_2._get_numeric_data().columns

describe_num_df = df_2.describe(include=['int64', 'float64'])
describe_num_df.reset_index(inplace=True)
describe_num_df = describe_num_df[describe_num_df['index'] != 'count']

df_2['Departure Delay'] = np.where(df_2['Departure Delay in Minutes'] > 5, "Yes", "No")
df_2['Arrival Delay'] = np.where(df_2['Arrival Delay in Minutes'] > 5, "Yes", "No")
df_2['Age_Cat'] = np.where(df_2['Age'] <= 2, "Child"
                           , np.where(df_2['Age'] <= 19, "Teenager"
                                      , np.where(df_2['Age'] <= 60, "Adult"
                                                 , np.where(df_2['Age'] > 60, 'Senior Citizen'
                                                            , "Adult"))))


#####################################################################
# ------------------ App ------------------ #
####################################################################


df_satisfied = df_2[df_2['satisfaction'] == 'satisfied']
df_unsatisfied = df_2[df_2['satisfaction'] != 'satisfied']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash("Moon Knight", external_stylesheets=external_stylesheets)

my_app.layout = html.Div(
    [
        html.H1("Moon Knight Air", style={'textAlign': 'center'}),

        dcc.Tabs(id='main-tabs',
                 children=[
                     dcc.Tab(label='Data Overview', value='dove',
                             children=[dcc.Tabs(id="sub-tabs"
                                                , children=[
                                     dcc.Tab(label="Snapshot", value="snap"),
                                     dcc.Tab(label="Target Variable Analysis", value="tgta"),
                                 ], colors={"border": "black", "primary": "red", "background": "cornsilk"}
                                                )
                                       ]
                             ),
                     dcc.Tab(label='Exploratory Data Analysis', value='eda',
                             children=[dcc.Tabs(id="sub-tabs1"
                                                , children=[
                                     dcc.Tab(label="Correlation", value="cor"),
                                     dcc.Tab(label="Principal Component Analysis", value="pca"),
                                     dcc.Tab(label="Outlier Detection and Removal", value="outlier"),
                                     dcc.Tab(label="Normality Test", value="dnorm")
                                 ], colors={"border": "black", "primary": "red", "background": "cornsilk"}
                                                )
                                       ]
                             ),
                     dcc.Tab(label='Pre-board', value='pb'),
                     dcc.Tab(label='On-board', value='ob'),
                     dcc.Tab(label='Post-flight', value='pf')
                 ], colors={"border": "black", "primary": "gold", "background": "cornsilk"}
                 ),

        html.Div(id='layout'),

    ]
)


target_variable_layout = html.Div(
    [
        # html.H3("Post-Flight Factors", style={'textAlign': 'center'}),
        # html.Br(),

        html.H5("Choose a factor: "),
        dcc.Dropdown(id="drop-tgt",
                     options=[
                         {'label': 'Gender', 'value': 'Gender'},
                         {'label': 'Customer Type', 'value': 'Customer Type'},
                         {'label': 'Type of Travel', 'value': 'Type of Travel'},
                         {'label': 'Class', 'value': 'Class'},
                         {'label': 'Ease of Online booking', 'value': 'Ease of Online booking'},
                         {'label': 'Gate location', 'value': 'Gate location'},
                         {'label': 'Online boarding', 'value': 'Online boarding'},
                         {'label': 'Checkin service', 'value': 'Checkin service'},
                         {'label': 'Departure/Arrival Time Convenient', 'value': 'Departure/Arrival time convenient'},
                         {'label': 'Inflight wi-fi service', 'value': 'Inflight wifi service'},
                         {'label': 'Food and drink', 'value': 'Food and drink'},
                         {'label': 'Seat comfort', 'value': 'Seat comfort'},
                         {'label': 'Inflight entertainment', 'value': 'Inflight entertainment'},
                         {'label': 'On-board service', 'value': 'On-board service'},
                         {'label': 'Leg room', 'value': 'Leg room service'},
                         {'label': 'Inflight service', 'value': 'Inflight service'},
                         {'label': 'Cleanliness', 'value': 'Cleanliness'},
                         {'label': 'Baggage handling', 'value': 'Baggage handling'}
                     ],
                     value="Class"
                     ),
        html.Br(),

        html.Div(
            [
                dcc.Graph(id="graph-tgt1"),
            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        ),

        html.Div(
            [
                dcc.Graph(id="graph-tgt2"),
                html.Div(id='txt')

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        )

    ]
)

outlier_layout = html.Div(
    [
        html.H3("Inter-Quartile Range (IQR) Method", style={'textAlign': 'center'}),

        html.H4("Select factor:"),

        dcc.RadioItems(
            id='outl',
            options=['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'],
            value='Flight Distance',
            inline=True
        ),

        html.Br(),

        html.Div(
            [
                html.H4("Detection", style={'textAlign': 'center'}),
                dcc.Graph(id="graph-eda1"),

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        ),

        html.Div(
            [
                html.H4("Removal", style={'textAlign': 'center'}),
                dcc.Graph(id="graph-eda11"),

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        )

    ]
)

cor_layout = html.Div(
    [
        html.H3("Correlation Matrix", style={'textAlign': 'center'}),

        html.Div(
            [
                dcc.Graph(id="graph-eda2",
                          style={"width": '200vh', "margin": 0, 'display': 'inline-block', "height": '90vh'}
                          ),

            ], style={"width": '100%', "margin": 0, 'display': 'inline-block'}
        ),

        dcc.RadioItems(
            id='edd',
            options=['Correlation'],
            value='Correlation',
            inline=True
        )
    ]
)

pca_layout = html.Div(
    [
        html.H3("PCA", style={'textAlign': 'center'}),
        html.Br(),
        html.H4("Number of Principal Components: "),

        dcc.Slider(id='edd1',
                   min=2,
                   max=23,
                   step=1,
                   value=17,
                   marks={2: "2", 5: "5", 8: "8", 11: "11", 14: "14", 17: "17", 20: "20", 23: "23"}
                   ),

        html.Div(
            [
                dcc.Graph(id="graph-eda3",
                          style={"width": '100vh', "margin": 0, 'display': 'inline-block', "height": '70vh'}),

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        ),

        html.Div(
            [
                dcc.Graph(id="graph-eda4",
                          style={"width": '100vh', "margin": 0, 'display': 'inline-block', "height": '70vh'}),

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        )
    ]
)

dsnap_layout = dash_table.DataTable(
    data=df_2[:50].to_dict('records'),
    columns=[{'id': c, 'name': c} for c in df_2.columns],

    editable=True,
    sort_action="native",
    page_action="native",
    page_current=0,
    style_table={
        'maxHeight': '70vh',
        'overflowY': 'scroll',
        'margin-top': '5vh',
        'margin-left': '3vh',
        'width': '90%'
    },

    css=[{
        'selector': '.dash-cell div.dash-cell-value',
        'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
    }],

    style_cell={
        'whiteSpace': 'normal',
        'textAlign': 'left',
        'width': '100%',
    },

    style_header={
        'backgroundColor': 'rgb(210, 210, 210)',
        'color': 'black',
        'fontWeight': 'bold'
    },

    style_as_list_view=True
)

dstat_layout = dash_table.DataTable(
    data=describe_num_df.to_dict('records'),
    columns=[{'id': c, 'name': c} for c in describe_num_df.columns],

    editable=True,
    sort_action="native",
    # column_selectable="single",
    # row_selectable="multi",
    # selected_columns=[],
    # selected_rows=[],
    page_action="native",
    page_current=0,
    style_table={
        'maxHeight': '70vh',
        'overflowY': 'scroll',
        'margin-top': '5vh',
        'margin-left': '3vh',
        'width': '90%'
    },

    css=[{
        'selector': '.dash-cell div.dash-cell-value',
        'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
    }],

    style_cell={
        'whiteSpace': 'normal',
        'textAlign': 'left',
        'width': '100%',
    },

    style_header={
        'backgroundColor': 'rgb(210, 210, 210)',
        'color': 'black',
        'fontWeight': 'bold'
    },

    style_as_list_view=True
)

d_over_layout = html.Div([
    html.Div([html.H4("Snapshot", style={'textAlign': 'center'}),
              dsnap_layout,
              html.Br(),
              html.Br(),
              html.Br(),
              html.Br(),
              html.Br(),

              ], style={'width': '50%', 'display': 'inline-block'}),
    html.Div([html.H4("Statistics", style={'textAlign': 'center'}),
              dstat_layout,
              html.Br(),
              html.H4("Customer Satisfaction - Target", style={'textAlign': 'center'}),
              dcc.Graph(id="graph-target"),

              dcc.RadioItems(
                  id='tgt',
                  options=['satisfaction'],
                  value='satisfaction',
                  inline=True
              )
              ], style={'width': '50%', 'display': 'inline-block'})
])

dnorm_layout = html.Div(
    [
        html.H3("Normality Check", style={'textAlign': 'center'}),

        html.H4("Select factor: "),
        dcc.Dropdown(id="drop-dnorm",
                     options=[
                         {'label': 'Age', 'value': 'Age'},
                         {'label': 'Flight Distance', 'value': 'Flight Distance'},
                         {'label': 'Departure Delay', 'value': 'Departure Delay in Minutes'},
                         {'label': 'Arrival Delay', 'value': 'Arrival Delay in Minutes'}
                     ],
                     value="Age"
                     ),
        html.Br(),

        html.H4("Choose Transformation: "),

        dcc.RadioItems(
            id='norm-trans',
            options=['Log', 'Square-Root', 'Reciprocal', 'Box-Cox'],
            value='Log',
            inline=True
        ),

        html.Div(
            [
                dcc.Graph(id="graph-eda5"),

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        ),

        html.Div(
            [
                dcc.Graph(id="graph-eda6"),

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        )
    ]
)

on_board_flight_layout = html.Div(
    [
        html.H3("On-Board Factors", style={'textAlign': 'center'}),

        html.H5("Choose a factor: "),
        dcc.Dropdown(id="drop",
                     options=[
                         {'label': 'Inflight wi-fi service', 'value': 'Inflight wifi service'},
                         {'label': 'Food and drink', 'value': 'Food and drink'},
                         {'label': 'Seat comfort', 'value': 'Seat comfort'},
                         {'label': 'Inflight entertainment', 'value': 'Inflight entertainment'},
                         {'label': 'On-board service', 'value': 'On-board service'},
                         {'label': 'Leg room', 'value': 'Leg room service'},
                         {'label': 'Inflight service', 'value': 'Inflight service'},
                         {'label': 'Cleanliness', 'value': 'Cleanliness'}
                     ],
                     value="Inflight service"
                     ),
        html.Br(),

        html.Div([
            html.H5("Overall Customer Satisfaction:"),
            dcc.Checklist(
                id='x-axis3',
                options=['Satisfied', 'Not Satisfied'],
                value=['Satisfied', 'Not Satisfied'],
                inline=True
            )
        ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}),

        html.Div([
            html.H5("Customer Demographic"),
            dcc.RadioItems(
                id='y-axis3',
                options=['Gender', 'Customer Type', 'Type of Travel', 'Class', 'Age_Cat'],
                value='Gender',
                inline=True
            )], style={"width": '50%', "margin": 0, 'display': 'inline-block'}),

        html.Div(
            [
                dcc.Graph(id="graph-ob1"),

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        ),

        html.Div(
            [
                dcc.Graph(id="graph-ob2"),

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        )

    ]
)

post_flight_layout = html.Div(
    [
        html.H3("Post-Flight Factors", style={'textAlign': 'center'}),
        html.Br(),

        html.H5("Choose a factor: "),
        dcc.Dropdown(id="drop-pf",
                     options=[
                         {'label': 'Baggage handling', 'value': 'Baggage handling'},
                         {'label': 'Arrival Delay', 'value': 'Arrival Delay'}
                     ],
                     value="Baggage handling"
                     ),
        html.Br(),

        html.Div([
            html.H5("Overall Customer Satisfaction:"),
            dcc.Checklist(
                id='x-axis1',
                options=['Satisfied', 'Not Satisfied'],
                value=['Satisfied', 'Not Satisfied'],
                inline=True
            )
        ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}),

        html.Div([
            html.H5("Customer Demographic"),
            dcc.RadioItems(
                id='y-axis1',
                options=['Gender', 'Customer Type', 'Type of Travel', 'Class', 'Age_Cat'],
                value='Gender',
                inline=True
            )], style={"width": '50%', "margin": 0, 'display': 'inline-block'}),

        html.Div(
            [
                dcc.Graph(id="graph-pf1")

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        ),

        html.Div(
            [
                dcc.Graph(id="graph-pf2"),
                html.Div(id='txt')

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        ),

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
            html.H5("Overall Customer Satisfaction:"),
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
                options=['Gender', 'Customer Type', 'Type of Travel', 'Class', 'Age_Cat'],
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
     Input(component_id='sub-tabs', component_property='value'),
     Input(component_id='sub-tabs1', component_property='value')]
)
def update_layout(tab, subtab, subtab1):
    if tab == "pf":
        return post_flight_layout
    elif tab == "pb":
        return pre_board_flight_layout
    elif tab == "ob":
        return on_board_flight_layout
    elif tab == "eda":
        if subtab1 == "cor":
            return cor_layout
        elif subtab1 == "pca":
            return pca_layout
        elif subtab1 == "outlier":
            return outlier_layout
        elif subtab1 == "dnorm":
            return dnorm_layout
    elif tab == "dove":
        if subtab == "tgta":
            return target_variable_layout
        elif subtab == "snap":
            return d_over_layout
        
# Target Variable Layout Callback Function

@my_app.callback(
    [
        Output(component_id='graph-tgt1', component_property='figure'),
        Output(component_id='graph-tgt2', component_property='figure')
    ],
    [
        Input(component_id='drop-tgt', component_property='value')
    ]
)
def tgta_input(dr):
    df1 = df_2.copy()

    if dr != 'Customer Type':
        if dr != 'Gender':
            if dr != 'Class':
                if dr != 'Type of Travel':
                    df1[dr] = np.where(df1[dr] == 1, "Bad"
                                       , np.where(df1[dr] == 2, "Average"
                                                  , np.where(df1[dr] == 3, "Good"
                                                             , np.where(df1[dr] == 4, 'Very Good'
                                                                        , np.where(df1[dr] == 5, 'Excellent', 'Average')))))

    fig = px.pie(df1, 'satisfaction')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig['layout'].update({
        'title': f'Customer Satisfaction Distribution'
    })

    fig1 = px.histogram(df1, x=dr, color='satisfaction', barmode='group')

    fig1['layout'].update({
        'title': f'{dr} v/s Satisfaction',
    })

    fig1.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig1.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig1.update_xaxes(categoryorder="total ascending")

    return fig, fig1

# Post-flight Layout Callback Function

@my_app.callback(
    [
        Output(component_id='graph-pf1', component_property='figure'),
        Output(component_id='graph-pf2', component_property='figure'),
        Output(component_id='txt', component_property='children')
    ],
    [
        Input(component_id='drop-pf', component_property='value'),
        Input(component_id='x-axis1', component_property='value'),
        Input(component_id='y-axis1', component_property='value')
    ]
)
def pf_input(dr, xa, ya):
    if xa == ['Satisfied']:
        df1 = df_satisfied.copy()
    elif xa == ['Not Satisfied']:
        df1 = df_unsatisfied.copy()
    else:
        df1 = df_2.copy()

    df1_hist = df1.copy()

    if xa == ['Not Satisfied']:
        k = "Not Satisfied"
    elif xa == ['Satisfied']:
        k = "Satisfied"
    else:
        k = "Overall"

    if dr == 'Baggage handling':
        df1[dr] = np.where(df1[dr] == 1, "Bad"
                           , np.where(df1[dr] == 2, "Average"
                                      , np.where(df1[dr] == 3, "Good"
                                                 , np.where(df1[dr] == 4, 'Very Good'
                                                            , np.where(df1[dr] == 5, 'Excellent'
                                                                       , 'Average')))))

    fig = px.pie(df1, dr)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig['layout'].update({
        'title': f'Data Distribution - {dr} Reviews - {k}'
    })

    if dr == "Arrival Delay":
        fig1 = px.histogram(df1_hist, x=dr, color='satisfaction', barmode='group')

        fig1['layout'].update({
            'title': f'{dr} v/s Satisfaction',
        })

        txt = "NOTE: ARRIVAL DELAY WILL ONLY BE SHOWN w.r.t SATISFACTION"
    else:
        fig1 = px.histogram(df1, x=dr, color=ya, barmode='group')

        fig1['layout'].update({
            'title': f'{dr} Reviews v/s {ya} - {k}',
            'xaxis': {
                'title': f'{dr}',
                'zeroline': False
            }
        })
        txt = ""

    fig1.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig1.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig1.update_xaxes(categoryorder="total ascending")

    return fig, fig1, txt

# Pre-Boarding Layout Callback Function

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

    if xa2 == ['Not Satisfied']:
        k = "Not Satisfied"
    elif xa2 == ['Satisfied']:
        k = "Satisfied"
    else:
        k = "Overall"

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
        'title': f'Data Distribution - {dr} Reviews - {k}'
    })

    fig1 = px.histogram(df1, x=dr, color=ya2, barmode='group')

    fig1['layout'].update({
        'title': f'{dr} Reviews v/s {ya2} - {k}',
        'xaxis': {
            'title': f'{dr}',
            'zeroline': False
        }
    })

    if dr == 'Departure Delay':
        fig1 = px.histogram(df1_hist, x=dr, color='satisfaction', barmode='group')
        txt = "NOTE: DEPARTURE DELAY WILL ONLY BE SHOWN w.r.t SATISFACTION"
    else:
        txt = ""

    fig1.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig1.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig1.update_xaxes(categoryorder="total ascending")

    return fig, fig1, txt


# On-Board Layout Callback Function


@my_app.callback(
    [
        Output(component_id='graph-ob1', component_property='figure'),
        Output(component_id='graph-ob2', component_property='figure')
    ],
    [
        Input(component_id='drop', component_property='value'),
        Input(component_id='x-axis3', component_property='value'),
        Input(component_id='y-axis3', component_property='value')
    ]
)
def ob_input(dr, xa, ya):
    if xa == ['Satisfied']:
        df1 = df_satisfied.copy()
    elif xa == ['Not Satisfied']:
        df1 = df_unsatisfied.copy()
    else:
        df1 = df_2.copy()

    # df1_hist = df1.copy()

    if xa == ['Satisfied', 'Not Satisfied']:
        k = "Overall"
    elif xa == ['Satisfied']:
        k = "Satisfied"
    else:
        k = "Not Satisfied"


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
        'title': f'Data Distribution - {dr} Reviews - {k}'
    })

    fig1 = px.histogram(df1, x=dr, color=ya, barmode='group')

    fig1['layout'].update({
        'title': f'{dr} Reviews v/s {ya} - {k}',
        'xaxis': {
            # 'title': f'{dr} -> (0: VeryBad, 1: Bad, 2: Average, 3: Good, 4: VeryGood, 5: Excellent)',
            'title': f'{dr}',
            'zeroline': False

        }
    })

    fig1.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig1.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig1.update_xaxes(categoryorder="total ascending")

    return fig, fig1

# Outlier Layout Callback Function

@my_app.callback([Output(component_id='graph-eda1', component_property='figure'),
                  Output(component_id='graph-eda11', component_property='figure')],
                 [Input(component_id='outl', component_property='value')]
                 )
def eda2_input(ya):
    fig = px.box(df_outlier[ya].values)

    fig1 = px.box(df_2[ya].values)

    return fig, fig1


# Correlation Matrix Layout Callback Function


@my_app.callback(Output(component_id='graph-eda2', component_property='figure'),
                 [Input(component_id='edd', component_property='value')]
                 )
def eda_input(ya):
    fig = px.imshow(df_corr.corr().values,
                    x=df_corr.corr().columns,
                    y=df_corr.corr().index
                    , text_auto=True
                    )

    return fig

# PCA Layout Callback Function


@my_app.callback([Output(component_id='graph-eda3', component_property='figure'),
                  Output(component_id='graph-eda4', component_property='figure')
                  ],
                 [
                     Input(component_id='edd1', component_property='value')
                 ]
                 )
def eda1_input(ya):
    if ya == 0:
        pca = PCA(n_components='mle', svd_solver='full')
    else:
        pca = PCA(n_components=ya, svd_solver='full')

    principalComponents = pca.fit_transform(X)

    a, b = principalComponents.shape

    column = []

    for i in range(b):
        column.append(f"Principal Component {i + 1}")

    df_PCA = pd.DataFrame(principalComponents, columns=column)

    df_PCA = pd.concat([df_PCA], axis=1)

    fig1 = go.Figure(go.Scatter(x=x,
                                y=np.cumsum(pca.explained_variance_ratio_),
                                mode='lines'))

    fig1['layout'].update({
        'title': 'Cumulative Explained Variance v/s Number of Components',
        'xaxis': {
            'title': 'Principal Component',
            'zeroline': False
        },
        'yaxis': {
            'title': 'Explained Variance Ratio'
        },
        'showlegend': False
    })

    fig = px.imshow(df_PCA.corr().values,
                    x=df_PCA.corr().columns,
                    y=df_PCA.corr().index
                    , text_auto=True
                    )
    fig['layout'].update({
        'title': 'Correlation Matrix - Principal Components'
    })

    return fig1, fig

# Normality Layout Callback Function


@my_app.callback(
    [Output(component_id='graph-eda5', component_property='figure'),
     Output(component_id='graph-eda6', component_property='figure')
     ],
    [Input(component_id='drop-dnorm', component_property='value'),
     Input(component_id='norm-trans', component_property='value')
     ]
)
def eda5_input(ya, ya1):
    qqplot_data = qqplot(df_2[ya], line='s').gca().lines

    if ya1 == 'Log':
        log_target = np.log1p(df_2[ya])
        qqplot_data1 = qqplot(log_target, line='s').gca().lines
    elif ya1 == 'Square-Root':
        sqrt_target = df_2[ya] ** (1 / 2)
        qqplot_data1 = qqplot(sqrt_target, line='s').gca().lines
    elif ya1 == 'Reciprocal':
        reciprocal_target = 1 / df_2[ya]
        qqplot_data1 = qqplot(reciprocal_target, line='s').gca().lines
    elif ya1 == 'Box-Cox':
        if ya == 'Departure Delay in Minutes' or ya == 'Arrival Delay in Minutes':
            bcx_target, lam = boxcox(df_2[df_2[ya] > 0][ya])
        else:
            bcx_target, lam = boxcox(df_2[ya])
        qqplot_data1 = qqplot(bcx_target, line='s').gca().lines

    fig = go.Figure()

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[0].get_xdata(),
        'y': qqplot_data[0].get_ydata(),
        'mode': 'markers',
        'marker': {
            'color': '#19d3f3'
        }
    })

    fig.add_trace({
        'type': 'scatter',
        'x': qqplot_data[1].get_xdata(),
        'y': qqplot_data[1].get_ydata(),
        'mode': 'lines',
        'line': {
            'color': '#636efa'
        }

    })

    fig['layout'].update({
        'title': f'Quantile-Quantile Plot of {ya}',
        'xaxis': {
            'title': 'Theoritical Quantities',
            'zeroline': False
        },
        'yaxis': {
            'title': 'Sample Quantities'
        },
        'showlegend': False,
        'width': 800,
        'height': 700,
    })

    fig1 = go.Figure()

    fig1.add_trace({
        'type': 'scatter',
        'x': qqplot_data1[0].get_xdata(),
        'y': qqplot_data1[0].get_ydata(),
        'mode': 'markers',
        'marker': {
            'color': '#19d3f3'
        }
    })

    fig1.add_trace({
        'type': 'scatter',
        'x': qqplot_data1[1].get_xdata(),
        'y': qqplot_data1[1].get_ydata(),
        'mode': 'lines',
        'line': {
            'color': '#636efa'
        }

    })

    fig1['layout'].update({
        'title': f'Quantile-Quantile Plot of {ya1} transformed {ya}',
        'xaxis': {
            'title': 'Theoritical Quantities',
            'zeroline': False
        },
        'yaxis': {
            'title': 'Sample Quantities'
        },
        'showlegend': False,
        'width': 800,
        'height': 700,
    })

    if ya == 'Age':
        fig1 = fig

    return fig, fig1


# Target Layout Callback Function

@my_app.callback(Output(component_id='graph-target', component_property='figure'),
                 [Input(component_id='tgt', component_property='value')]
                 )
def tgt_input(ya):
    fig = px.histogram(df_2, x=ya, color=ya)
    return fig


# if __name__ == '__main__':
#     my_app.run_server(debug=True, host='0.0.0.0', port=8080)
my_app.run_server(debug=True, port=8000)