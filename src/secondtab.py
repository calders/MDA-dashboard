from dash import html, dcc, Output, Input
import dash_bootstrap_components as dbc
from joblib import load

def layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H3("Predict Survival", className="card-title text-center"),
                        dbc.Card(
                            dbc.CardBody([
                                html.Div([
                                    html.H5("Select Distance"),
                                    dcc.Slider(
                                        id='distance-slider',
                                        min=0,
                                        max=4000,
                                        step=100,
                                        value=2000,
                                        marks={
                                            0: '0m',
                                            1000: '1000m',
                                            2000: '2000m',
                                            3000: '3000m',
                                            4000: '4000m'
                                        },
                                        tooltip={
                                            "always_visible": True,
                                            "style": {"color": "LightSteelBlue", "fontSize": "20px"},
                                        },
                                    ),
                                ]),
                            ]),
                            style={"width": "100%", "margin-top": "20px"}
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                html.Div([
                                    html.H5("Month, weekday, hour of intervention"),
                                    html.Div([
                                        dcc.Dropdown(
                                            id='month-dropdown',
                                            options=[{'label': str(month), 'value': month} for month in range(1, 13)],
                                            value=None,
                                            placeholder='Select Month',
                                            style={'width': '42%', 'margin-right': '1%', 'color': 'black'}
                                        ),
                                        dcc.Dropdown(
                                            id='weekday-dropdown',
                                            options=[{'label': str(weekday), 'value': weekday} for weekday in range(7)],
                                            value=None,
                                            placeholder='Select Weekday',
                                            style={'width': '42%', 'margin-right': '1%', 'color': 'black'}
                                        ),
                                        dcc.Dropdown(
                                            id='hour-dropdown',
                                            options=[{'label': str(hour), 'value': hour} for hour in range(24)],
                                            value=None,
                                            placeholder='Select Hour',
                                            style={'width': '42%', 'color': 'black'}
                                        ),
                                    ], style={'display': 'flex'}),
                                ]),
                            ]),
                            style={"width": "100%", "margin-top": "20px"}
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                html.Div([
                                    html.H5("Different emergency services for urgent care"),
                                    dcc.Checklist(
                                        id='vector-type-checkboxes',
                                        options=[
                                            {'label': 'Ambulance', 'value': 'AMB'},
                                            {'label': 'Paramedisch Interventieteam (PIT)', 'value': 'PIT'},
                                            {'label': 'Mobiele urgentiegroep (MUG)', 'value': 'MUG'}
                                        ],
                                        value=['AMB', 'MUG', 'PIT'],
                                        labelStyle={"display": "flex", "align-items": "center"},
                                    ),
                                ]),
                            ]),
                            style={"width": "100%", "margin-top": "20px"}
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                html.H5("Predicted outcome", className="card-title"),
                                html.Div(id="prediction-output", className="card-text")
                            ]),
                            style={"width": "100%", "margin-top": "20px"}
                        ),
                    ]),
                    style={"width": "100%", "margin-top": "50px"}
                ),
                width=6
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H3("Aanrijtijd", className="card-title text-center"),
                        html.P("The rest has to be added after I've seen the variables.")
                    ]),
                    style={"width": "100%", "margin-top": "50px"}
                ),
                width=6
            ),
        ]),
    ])

# Load Stijns model
model1 = load('data/stijnknnl.pkl')


# The interaction for KNN
def register_callbacks(app):
    @app.callback(
        Output('prediction_result', 'children'),
        [
            Input('distance-slider', 'value'),
            Input('month-dropdown', 'value'),
            Input('weekday-dropdown', 'value'),
            Input('hour-dropdown', 'value'),
            Input('vector-type-checkboxes', 'value')
        ]
    )
    # The callback function with the five input arguments
    def update_prediction(distance, month, weekday, hour, emergency_services):
        # the KNN to make predictions
        prediction = model1.predict([[distance, month, weekday, hour, *emergency_services]])
        return f'Predicted outcome: {prediction}'

