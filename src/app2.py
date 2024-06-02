import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import dash_leaflet as dl
import pandas as pd
import joblib
from datetime import datetime as dt
import datetime
import xgboost
from src.GoogleMapsAPI import GoogleMapsAPI

# Initialize the global variables
df = pd.DataFrame()

####################
# helper functions #
####################

def preprocessing(coordinate_permanence: tuple, coordinate_intervention: tuple, datetime_intervention: datetime, vector_type: str) -> tuple:
    api_key = "AIzaSyCS4JmrsaaWSb-20YZQL_Furm94sQAjM_0"

    distance_driving = GoogleMapsAPI(api_key).get_distance(coordinate_permanence, coordinate_intervention, crow=False)
    duration_driving = GoogleMapsAPI(api_key).get_duration(coordinate_permanence, coordinate_intervention)
    distance_crow = GoogleMapsAPI(api_key).get_distance(coordinate_permanence, coordinate_intervention, crow=True)

    month = datetime_intervention.month
    weekday = datetime_intervention.weekday()
    hour = datetime_intervention.hour

    return distance_driving, duration_driving, distance_crow, month, weekday, hour, vector_type

def mimick_ohe(df: pd.DataFrame) -> pd.DataFrame:
    month = df['month'].values[0]
    for i in range(2, 13):
        df[f'month_{i}'] = 0
    if month != 1:
        df[f'month_{month}'] = 1

    weekday = df['weekday'].values[0]
    for i in range(1, 7):
        df[f'weekday_{i}'] = 0
    if weekday != 0:
        df[f'weekday_{weekday}'] = 1

    hour = df['hour'].values[0]
    for i in range(1, 24):
        df[f'hour_{i}'] = 0
    if hour != 0:
        df[f'hour_{hour}'] = 1

    vector_type = df['vector_type'].values[0]
    df['vector_type_MUG'] = 1 if vector_type == 'MUG' else 0
    df['vector_type_PIT'] = 1 if vector_type == 'PIT' else 0

    df.drop(columns=['month', 'weekday', 'hour', 'vector_type'], inplace=True)

    return df

def get_permanence_coordinates(n_clicks, ids):
    if not any(n_clicks):
        return None
    
    # Filter out None values from n_clicks
    filtered_clicks = [n_click for n_click in n_clicks if n_click is not None]
    
    # If all clicks are None, return the default message
    if not filtered_clicks:
        return None
    
    clicked_index = n_clicks.index(max(filtered_clicks))
    clicked_marker_id = ids[clicked_index]
    
    # Extract the marker's index from its ID
    selected_marker = df[df["name"] == clicked_marker_id["index"]]
    if selected_marker.empty:
        return None
    
    lat, lon = selected_marker["latitude"].values[0], selected_marker["longitude"].values[0]
    return (lat, lon)

##########################
# load Kasper's ML model #
##########################
modelKasper = joblib.load("data/model_aanrijtijd_20min_xgboost.pkl")

#########################
# load Stijn's ML model #
#########################
modelStijn = joblib.load('data/stijnknnl.pkl')

#####################
# Create a Dash app #
#####################

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Scenario analysis'),

    html.Div([
        dl.Map(center=[50.5, 4.5], zoom=8, children=[
            dl.TileLayer(id="base-layer"),
            dl.LayerGroup(id="ambulance-layer"),
            dl.LayerGroup(id="patient-layer"),
        ], style={'width': '50%', 'height': '50vh', 'margin': "auto", "display": "block"}, id="map")
    ]),
    html.Br(),
   
# Create a container for dropdowns
    html.Div([
        html.Div([
            html.Div("Vector type:"),
            dcc.Dropdown(
                id='vector-type',
                options=[
                    {'label': 'Ambulance', 'value': 'AMB'},
                    {'label': 'MUG', 'value': 'MUG'},
                    {'label': 'PIT', 'value': 'PIT', 'disabled': True}
                ],
                value='AMB'
            ),
        ], style={'display': 'inline-block', 'width': '45%', 'margin-right': '5%'}),
        
        html.Div([
            html.Div("Province:"),
            dcc.Dropdown(
                id='province',
                options=[
                    {'label': 'Antwerp', 'value': 'ANT'},
                    {'label': 'Limburg', 'value': 'LIM'},
                    {'label': 'Oost-Vlaanderen', 'value': 'OVL'},
                    {'label': 'Vlaams Brabant', 'value': 'VBR'},
                    {'label': 'West-Vlaanderen', 'value': 'WVL'},
                    {'label': 'Brussel/Bruxelles', 'value': 'BXL'},
                    {'label': 'Brabant wallon', 'value': 'WBR'},
                    {'label': 'Hainaut', 'value': 'HAI'},
                    {'label': 'Liège', 'value': 'LIE'},
                    {'label': 'Luxembourg', 'value': 'LUX'},
                    {'label': 'Namur', 'value': 'NAM'},
                ],
                value='VBR'
            ),
        ], style={'display': 'inline-block', 'width': '45%'})
    ]),
    html.Br(),

    html.Div([
        html.Div([
            html.Div('Select the date:'),
            dcc.DatePickerSingle(
                id='date-picker',
                date=datetime.datetime.now(),
                display_format='DD MMM YYYY'
            ),
        ], style={'display': 'inline-block', 'width': '45%', 'margin-right': '5%'}),
        html.Div([
            html.Div('Choose hour (0-23):', style={'margin-left': '20px'}),
            dcc.Input(
                id='hour-picker',
                value='0',
                type='text'
            ),
        ], style={'display': 'inline-block', 'width': '45%'})
    ]),

    html.H2('Input data for the models'),
    dcc.Textarea(
        id='output-container-input-data',
        value='Textarea content initialized\nwith multiple lines of text',
        style={'width': '100%', 'height': 100},
    ),

    html.Div([
        html.H2('Predictions'),
        html.Div(id='output-container-expected-aanrijtijd'),
        html.Br(),
        html.Div(id='output-container-expected-outcome'),
    ], style={'border': '2px solid #4CAF50', 'border-radius': '5px', 'padding': '15px', 'background-color': '#f2f2f2', 'font-size': '1.2em'}),
    html.Br(),
    ])

@app.callback(
    Output('ambulance-layer', 'children'),
    Input('vector-type', 'value'),
    Input('province', 'value')
)
def add_markers(vector_type, province):
    global df

    if vector_type == "AMB":
        df = pd.read_csv("data/ambulance_locations.csv")
    elif vector_type == "MUG":
        df = pd.read_csv("data/mug_locations.csv")
    else:
        raise ValueError("PIT not implemented yet")
    df = df[df["province"]==province]
    existing_markers = []
    for _, row in df.iterrows():
        new_marker = dl.Marker(position=[row["latitude"], row["longitude"]],
                               id={'type': 'ambulance-marker', 'index': row['name']},
                               icon={"iconUrl": "https://img.icons8.com/?size=100&id=dtyM9DnqKjDM&format=png&color=000000", "iconSize": [25,]},
                               children=dl.Tooltip(row["name"]))
        existing_markers.append(new_marker)
    return existing_markers

@app.callback(
    Output('patient-layer', 'children'),
    Input('map', 'clickData')
)
def add_click_marker(clickData):
    if clickData is None:
        return []
    else:
        lat_intervention = clickData["latlng"]["lat"]
        lon_intervention = clickData["latlng"]["lng"]
        return [dl.Marker(position=[lat_intervention, lon_intervention], 
                          icon={"iconUrl": "https://img.icons8.com/?size=100&id=aRMbtEpJbrOj&format=png&color=000000", "iconSize": [25,]})]

@app.callback(
    Output('output-container-input-data', 'value'),
    Input('map', 'clickData'),
    Input('date-picker', 'date'),
    Input('hour-picker', 'value'),
    Input('vector-type', 'value'),
    Input('province', 'value'),
    Input({'type': 'ambulance-marker', 'index': ALL}, 'n_clicks'),
    State({'type': 'ambulance-marker', 'index': ALL}, 'id'),
)
def show_input_data(clickData, date, hour, vector_type, province, n_clicks, ids):
    coordinate_permanence = get_permanence_coordinates(n_clicks, ids)
    if coordinate_permanence is None:
        coordinate_permanence = "not set"
    if clickData is None:
        coordinate_intervention = "not set"
    else:
        lat_intervention = clickData["latlng"]["lat"]
        lon_intervention = clickData["latlng"]["lng"]
        coordinate_intervention = (lat_intervention, lon_intervention)
    datetime_intervention = datetime.datetime.strptime(date[:10], '%Y-%m-%d')
    datetime_intervention = datetime_intervention.replace(hour=int(hour))
    return f"coordinate permanence: {coordinate_permanence}\ncoordinate intervention: {coordinate_intervention}\ndatetime: {datetime_intervention}\nprovince: {province}\nvector type: {vector_type}"

@app.callback(
    Output('output-container-expected-outcome', 'children'),
    Input('map', 'clickData'),
    Input('date-picker', 'date'),
    Input('hour-picker', 'value'),
    Input('vector-type', 'value'),
    Input({'type': 'ambulance-marker', 'index': ALL}, 'n_clicks'),
    State({'type': 'ambulance-marker', 'index': ALL}, 'id'),
)
def predict_outcome(clickData, date, hour, vector_type, n_clicks, ids):
    coordinate_permanence = get_permanence_coordinates(n_clicks, ids)
    if coordinate_permanence is None:
        return "Click on an icon to set the permanence location"
    if clickData is None:
        return "Click on the map to set the intervention location"
    else:
        lat_intervention = clickData["latlng"]["lat"]
        lon_intervention = clickData["latlng"]["lng"]
        coordinate_intervention = (lat_intervention, lon_intervention)
    datetime_intervention = datetime.datetime.strptime(date[:10], '%Y-%m-%d')
    datetime_intervention = datetime_intervention.replace(hour=int(hour))

    distance_driving, duration_driving, distance_crow, month, weekday, hour, vector_type = preprocessing(coordinate_permanence, coordinate_intervention, datetime_intervention, vector_type)
    df = pd.DataFrame([[distance_driving, duration_driving, distance_crow, month, weekday, hour, vector_type]], columns=['distance_driving', 'duration_driving', 'distance_crow', 'month', 'weekday', 'hour', 'vector_type'])
    X = df[['distance_driving', 'duration_driving', 'distance_crow', 'month', 'weekday', 'hour', 'vector_type']]
    X = mimick_ohe(X) # do the same transformation as pd.get_dummies in the training phase
    X = X.to_numpy()
    prediction = modelStijn.predict(X)

    if prediction == 0:
        message = html.Span(['The patient will probably be ', html.B('dead'), ' when the emergency services arrive.'])
    else:
        message = html.Span(['The patient will probably be ', html.B('alive'), ' when the emergency services arrive.'])
    return message

@app.callback(
    Output('output-container-expected-aanrijtijd', 'children'),
    Input('map', 'clickData'),
    Input('date-picker', 'date'),
    Input('hour-picker', 'value'),
    Input('vector-type', 'value'),
    Input('province', 'value'),
    Input({'type': 'ambulance-marker', 'index': ALL}, 'n_clicks'),
    State({'type': 'ambulance-marker', 'index': ALL}, 'id'),
)
def predict_aanrijtijd(clickData, date, hour, vector_type, province, n_clicks, ids):
    coordinate_permanence = get_permanence_coordinates(n_clicks, ids)
    if coordinate_permanence is None:
        return "Click on an icon to set the permanence location"
    else:
        longitude_permanence = coordinate_permanence[1]
        latitude_permanence = coordinate_permanence[0]
    if clickData is None:
        return "Click on the map to set the intervention location"
    else:
        latitude_intervention = clickData["latlng"]["lat"]
        longitude_intervention = clickData["latlng"]["lng"]
    coordinate_intervention = (latitude_intervention, longitude_intervention)
    datetime_intervention = datetime.datetime.strptime(date[:10], '%Y-%m-%d')
    datetime_intervention = datetime_intervention.replace(hour=int(hour))

    distance_driving, duration_driving, distance_crow, month, weekday, hour, vector_type = preprocessing(coordinate_permanence, coordinate_intervention, datetime_intervention, vector_type)
    df = pd.DataFrame([[longitude_permanence, latitude_permanence, longitude_intervention, latitude_intervention, distance_driving, duration_driving, distance_crow, month, weekday, hour, vector_type, province]], columns=['longitude_permanence', 'latitude_permanence', 'longitude_intervention', 'latitude_intervention','distance_driving', 'duration_driving', 'distance_crow', 'month_of_year', 'day_of_week', 'hour_of_day', 'vector_type', 'province_intervention'])
    prediction = modelKasper.predict(df)

    mins = int(prediction[0] / 60)
    secs = int(prediction[0] % 60)

    return f"Expected time between call and arrival: {mins} minutes and {secs} seconds"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
