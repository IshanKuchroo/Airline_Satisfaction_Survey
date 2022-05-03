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
                 ),

        html.Div(id='layout'),

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
              html.Br(),
              html.Br()
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

        html.H4("Select factor: "),
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

        html.Div(
            [
                html.P("Data Distribution"),

                dcc.Graph(id="graph-ob1"),

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        ),

        html.Div(
            [
                html.P("Overall Satisfaction Level"),
                dcc.Checklist(
                    id='x-axis3',
                    options=['Satisfied', 'Not Satisfied'],
                    value=['Satisfied'],
                    inline=True
                ),
                html.P("Demographics"),
                dcc.RadioItems(
                    id='y-axis3',
                    options=['Gender', 'Customer Type', 'Type of Travel', 'Class'],
                    value='Gender',
                    inline=True
                ),
                dcc.Graph(id="graph-ob2"),

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        )

    ]
)

post_flight_layout = html.Div(
    [
        html.H3("Post-Flight Factors", style={'textAlign': 'center'}),
        html.Br(),

        html.Div(
            [
                html.P("Satisfaction Level"),
                dcc.Checklist(
                    id='x-axis',
                    options=['Satisfied', 'Not Satisfied'],
                    value=['Satisfied'],
                    inline=True
                ),
                html.P("Demographics"),
                dcc.RadioItems(
                    id='y-axis',
                    options=['Gender', 'Customer Type', 'Type of Travel', 'Class'],
                    value='Gender',
                    inline=True
                ),
                dcc.Graph(id="graph-pf1"),

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        ),

        html.Div(
            [
                html.P("Satisfaction Level"),
                dcc.Checklist(
                    id='x-axis1',
                    options=['Satisfied', 'Not Satisfied'],
                    value=['Satisfied'],
                    inline=True
                ),
                html.P("Numerical Factors"),
                dcc.RadioItems(
                    id='y-axis1',
                    options=['Flight Distance', 'Departure Delay in Minutes'],
                    value='Departure Delay in Minutes',
                    inline=True
                ),
                dcc.Graph(id="graph-pf2"),

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        )

    ]
)

pre_board_flight_layout = html.Div(
    [
        html.H3("Pre-Boarding Factors", style={'textAlign': 'center'}),
        html.Br(),

        html.H4("Select the factor: "),
        dcc.Dropdown(id="drop",
                     options=[
                         {'label': 'Ease of Online booking', 'value': 'Ease of Online booking'},
                         {'label': 'Gate location', 'value': 'Gate location'},
                         {'label': 'Online boarding', 'value': 'Online boarding'},
                         {'label': 'Checkin service', 'value': 'Checkin service'},
                         {'label': 'Departure Delay', 'value': 'Departure Delay'},
                         {'label': 'Departure/Arrival time convenient', 'value': 'Departure/Arrival time convenient'}
                     ],
                     value="Online boarding"
                     ),
        html.Br(),

        html.Div(
            [
                html.P("Data Distribution"),

                dcc.Graph(id="graph-pb1"),

            ], style={"width": '50%', "margin": 0, 'display': 'inline-block'}
        ),

        html.Div(
            [
                html.P("Overall Satisfaction Level"),
                dcc.Checklist(
                    id='x-axis2',
                    options=['Satisfied', 'Not Satisfied'],
                    value=['Satisfied'],
                    inline=True
                ),
                html.P("Demographics"),
                dcc.RadioItems(
                    id='y-axis2',
                    options=['Gender', 'Customer Type', 'Type of Travel', 'Class'],
                    value='Gender',
                    inline=True
                ),
                dcc.Graph(id="graph-pb2"),

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
        return d_over_layout


@my_app.callback(
    [
        Output(component_id='graph-pf1', component_property='figure'),
        Output(component_id='graph-pf2', component_property='figure')
    ],
    [
        Input(component_id='x-axis', component_property='value'),
        Input(component_id='y-axis', component_property='value'),
        Input(component_id='x-axis1', component_property='value'),
        Input(component_id='y-axis1', component_property='value')
    ]
)
def pf_input(xa, ya, xa1, ya1):
    if xa == ['Satisfied']:
        df1 = df_satisfied
    elif xa == ['Not Satisfied']:
        df1 = df_unsatisfied
    else:
        df1 = df_2
    fig = px.histogram(df1[:2000], x='Baggage handling', color=ya, barmode='group')

    if xa1 == ['Satisfied']:
        df3 = df_satisfied
    elif xa1 == ['Not Satisfied']:
        df3 = df_unsatisfied
    else:
        df3 = df_2

    fig1 = px.scatter(df3[:2000], x='Arrival Delay in Minutes', y=ya1)

    return fig, fig1


@my_app.callback(
    [
        Output(component_id='graph-pb1', component_property='figure'),
        Output(component_id='graph-pb2', component_property='figure')
    ],
    [
        Input(component_id='drop', component_property='value'),
        Input(component_id='x-axis2', component_property='value'),
        Input(component_id='y-axis2', component_property='value')
    ]
)
def pb_input(dr, xa2, ya2):
    if xa2 == ['Satisfied']:
        df1 = df_satisfied
    elif xa2 == ['Not Satisfied']:
        df1 = df_unsatisfied
    else:
        df1 = df_2

    fig = px.pie(df1[:2000], dr)
    fig.update_traces(pull=[0, 0, 0.2, 0, 0])

    fig1 = px.histogram(df1[:2000], x=dr, color=ya2, barmode='group')

    if dr == 'Departure Delay':
        fig1 = px.histogram(df1[:2000], x=dr, color='satisfaction', barmode='group')

    return fig, fig1


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
        df1 = df_satisfied
    elif xa == ['Not Satisfied']:
        df1 = df_unsatisfied
    else:
        df1 = df_2

    fig = px.pie(df1[:2000], dr)
    fig.update_traces(pull=[0, 0, 0.2, 0, 0])

    fig1 = px.histogram(df1[:2000], x=dr, color=ya, barmode='group')

    return fig, fig1


@my_app.callback([Output(component_id='graph-eda1', component_property='figure'),
                  Output(component_id='graph-eda11', component_property='figure')],
                 [Input(component_id='outl', component_property='value')]
                 )
def eda2_input(ya):

    fig = px.box(df_outlier[ya].values)

    fig1 = px.box(df_2[ya].values)

    return fig, fig1


@my_app.callback(Output(component_id='graph-eda2', component_property='figure'),
                 [Input(component_id='edd', component_property='value')]
                 )
def eda_input(ya):
    fig = px.imshow(df_2.corr().values,
                    x=df_2.corr().columns,
                    y=df_2.corr().index
                    , text_auto=True
                    )

    return fig


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


@my_app.callback(
    [Output(component_id='graph-eda5', component_property='figure'),
     Output(component_id='graph-eda6', component_property='figure')
     ],
    [Input(component_id='drop-dnorm', component_property='value'),
     Input(component_id='norm-trans', component_property='value')
     ]
)
def eda5_input(ya, ya1):

    qqplot_data = qqplot(df_2[ya][:2000], line='s').gca().lines

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


@my_app.callback(Output(component_id='graph-target', component_property='figure'),
                 [Input(component_id='tgt', component_property='value')]
                 )
def tgt_input(ya):
    fig = px.bar(df_2[:5000], x=ya, color=ya)
    return fig


my_app.run_server(debug=True, port=8000)
