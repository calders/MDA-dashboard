# Firsttab to make it workable, otherwise the file's too long. The callbacks are below
import json
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import joblib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import Dash, html, dcc, Output, Input
from dash_extensions.javascript import arrow_function, assign
from dash import dcc, html
import dash_bootstrap_components as dbc

from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_leaflet as dl

def layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Key statistics", style={'text-align': 'center'}))
        ]),
        # Here we start with the layout of the first tab
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        dcc.Graph(id='hourly-distribution-chart', figure=fig_hourly_distribution_outcome)
                    ]),
                    className="mb-4"
                ),
                width=6
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        dcc.Graph(id='pie-chart', figure=fig_pie_chart)
                    ]),
                    className="mb-4"
                ),
                width=6
            )
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        dcc.Dropdown(
                            id="variable-dropdown",
                            options=[{'label': var, 'value': var} for var in variables_list],
                            value=variables_list[0],
                            clearable=False,
                            style={'width': '100%', 'margin': '10px auto'}
                        ),
                        dl.Map(children=[
                            dl.TileLayer(),
                            dl.GeoJSON(data={}),
                            dl.Colorbar(colorscale=["blue", "red"]),
                            dl.Marker(position=[51, 5]),
                            html.Div(children="Map Info", id="info")
                        ], style={'height': '80vh'}, center=[51, 5], zoom=5)
                    ]),
                    className="mb-4"
                )
            )
        ])
    ])



#@ app.callback(
 #     Output("geojson", "hideout"),
 #     Output("choro_colorbar", "classes"),
 #     Output("choro_colorbar", "colorscale"),
 #     Output("choro_colorbar", "tickValues"),
 #     Output("choro_colorbar", "tickText"),
 #     Output("choro_colorbar", "max"),
 #     Input("variable-dropdown", "value")
  #          )
def update_choropleth(selected_variable):
    properties_list = [feature["properties"] for feature in geojson_data['features']]

    df = pd.DataFrame(properties_list)

    quantiles = pd.qcut(df[df[selected_variable] > 0][selected_variable], 8, duplicates='drop')

    classes_new = df.groupby(quantiles, observed=False)[selected_variable].min().tolist()

    colorscale_dict = {variable: color for variable, color in zip(variables_list, colors)}
    colorrange = generate_color_range(colorscale_dict[selected_variable][0], colorscale_dict[selected_variable][1],
                                          len(classes_new))

    indices = list(range(len(classes_new) + 1))
    ctg = ["{:.0f}+".format(cls) for cls in classes_new]

    return dict(colorscale=colorrange, classes=classes_new, style=style, colorProp=selected_variable), \
            indices, \
            colorrange, \
            [item + 0.5 for item in indices[:-1]], ctg, len(classes_new)

    @app.callback(
        Output("info", "children"),
        Input("geojson", "hoverData")
    )
    def info_hover(feature):
        return get_info(feature)



#
#
#
#

# Aron's work
with open('../data/muni_for_viz.json') as f:
    geojson_data = json.load(f)

with open('../data/permanence_points.json') as f:
    permanence_points = json.load(f)

# Read local data
df1 = pd.read_csv('../data/data_dashboard.csv')
# df2 = pd.read_csv('data/data_choropleth')
# //////////////////////// trying out
# Convert 'time0' column to datetime if it's not already
df1['time0'] = pd.to_datetime(df1['time0'])

# Replace numeric values with corresponding labels in the 'outcome' column
df1['outcome'] = df1['outcome'].replace({0: 'Deceased', 1: 'Survived'})

# Group data by hour and outcome and count interventions
hourly_distribution_by_outcome = df1.groupby([df1['time0'].dt.hour, 'outcome']).size().unstack(fill_value=0).reset_index()

# Create the line chart
fig_hourly_distribution_outcome = px.line(hourly_distribution_by_outcome, x='time0', y=hourly_distribution_by_outcome.columns[1:],
                                          title='Number of missions over time',
                                          labels={'time0': 'Hour of the Day', 'value': 'Number of Interventions', 'outcome': 'Outcome'})
# Set x-axis tick marks to represent each hour of the day
fig_hourly_distribution_outcome.update_xaxes(tickvals=list(range(24)))

# Create data for pie chart
labels = ['Category A', 'Category B', 'Category C']
values = [20, 30, 50]

# Create pie chart figure
fig_pie_chart = go.Figure(go.Pie(labels=labels, values=values, name='Pie Chart'))

# /////////////
def generate_color_range(color1, color2, num_colors):
    color1_rgb = mcolors.hex2color(color1)
    color2_rgb = mcolors.hex2color(color2)

    colors = [color1_rgb]
    for i in range(1, num_colors):
        ratio = i / (num_colors - 1)
        new_color_rgb = tuple((np.array(color1_rgb) * (1 - ratio) + np.array(color2_rgb) * ratio))
        colors.append(new_color_rgb)

    return [mcolors.rgb2hex(color) for color in colors]

def get_info(feature=None):
    header = [html.H4("Select in the dropdown menu what", style={"margin": "0 0 5px", "color": "#777"}),
              html.H4("variable you want to investigate", style={"margin": "0 0 5px", "color": "#777"})]
    if not feature:
        return header + [html.P("Hoover over a municipality")]
    return header + [html.B(feature["properties"]["Municipality name"]), html.Br(),
                     "{:.0f} AED's".format(feature["properties"]["Amount of AED's"]), html.Br(),
                     "{:.0f} public AED's".format(feature["properties"]["Public AED's"]), html.Br(),
                     "{:.0f} private AED's".format(feature["properties"]["Private AED's"]), html.Br(),
                     "{:.3f} average distance travelled by emergency vehicles (km)".format(
                         feature["properties"]["Average distance travelled by emergency vehicles (km)"]), html.Br(),
                     "{:.0f} average time for emergency vehicles to arrive (s)".format(
                         feature["properties"]["Average time for emergency vehicles to arrive (s)"]), html.Br(),
                     "{:.0f} patient(s) dying before intervention".format(
                         feature["properties"]["Patient dying before intervention"]), html.Br(),
                     "{:.0f} patient(s) surviving after intervention".format(
                         feature["properties"]["Patient surviving after intervention"]), html.Br(),
                     "{:.0f} patient(s) dying after intervention".format(
                         feature["properties"]["Patient dying after intervention"]), html.Br()
                     ]


variables_list = [
    "Amount of AED's",
    'Patient dying before intervention',
    'Average distance travelled by emergency vehicles (km)',
    "Public AED's",
    'Patient surviving after intervention',
    "Private AED's",
    'Patient dying after intervention',
    'Average time for emergency vehicles to arrive (s)']

colors = [
    ['#fff5f0', '#67000d'],  # Reds
    ['#f7fcf5', '#00441b'],  # greens
    ['#fff5eb', '#7f2704'],  # oranges
    ['#fcfbfd', '#3f007d'],  # purples
    ['#f7fbff', '#08306b'],  # blues
    ['#f7fcf0', '#084081'],  # GnBu
    ['#fee825', '#440154'],  # reversed viridis
    ['#ffffcc', '#800026']  # YlOrRd
]

properties_list = [feature["properties"] for feature in geojson_data['features']]
df = pd.DataFrame(properties_list)
quantiles = pd.qcut(df[df["Amount of AED's"] > 0]["Amount of AED's"], 8, duplicates='drop')

classes = df.groupby(quantiles, observed=False)["Amount of AED's"].min().tolist()

colorrange = generate_color_range(colors[0][0], colors[0][1], len(classes))

style = dict(weight=2, opacity=1, color='white', dashArray='', fillOpacity=0.7)

ctg = ["{:.0f}+".format(cls) for cls in classes]
indices = list(range(len(classes) + 1))
colorbar = dl.Colorbar(min=0, max=len(classes), classes=indices, colorscale=colors[0], width=300, height=30,
                       position="bottomleft", id='choro_colorbar', tooltip=False,
                       tickValues=[item + 0.5 for item in indices[:-1]], tickText=ctg)

nodata = dl.Colorbar(min=0, max=2, classes=[0, 1], colorscale=['#000000', '#000000'], width=30, height=30,
                     position="bottomleft", id='nodata_colorbar', tooltip=False,
                     tickValues=[0.5, 1.5], tickText=["No data", ""])

style_handle = assign("""function(feature, context){
    const {classes, colorscale, style, colorProp} = context.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value >= classes[i]) {
            style.fillColor = colorscale[i];  // set the fill color according to the class
        }
        if (value == 0) {
            style.fillColor = '#000000';
        }
    }
    return style;
}""")

geojson = dl.GeoJSON(data=geojson_data,
                     style=style_handle,
                     zoomToBounds=True,
                     zoomToBoundsOnClick=True,
                     hoverStyle=arrow_function(dict(weight=5, color='#666', dashArray='')),
                     interactive=True,
                     hideout=dict(colorscale=colorrange, classes=classes, style=style, colorProp="Amount of AED's"),
                     id="geojson")

info = html.Div(children=get_info(), id="info", className="info",
                style={"position": "absolute", "top": "10px", "right": "10px", "zIndex": "1000", "padding": "6px 8px",
                       "background": "rgba(255,255,255,0.8)", "border-radius": "5px",
                       "box-shadow": "0 0 15px rgba(0,0,0,0.2)"})

dd_options = [{'label': variable, 'value': variable} for variable in variables_list]