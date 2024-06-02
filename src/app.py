from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from firsttab import layout as firsttab_layout
from secondtab import layout as secondtab_layout
from secondtab import register_callbacks  # recognise the @app functions
from firsttab import  register_callbacks_for_first_tab

app = Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ],  suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Enhancing Emergency Response: Insights into AED Deployment and Scenario Analysis", style={'text-align': 'center'}))
        ]),
        dbc.Row([
            dbc.Tabs(id='tabs', active_tab='tab1', children=[
                dbc.Tab(label='Overview', tab_id='tab1'),
                dbc.Tab(label='Scenario analysis', tab_id='tab2'),
            ]),
            html.Div(id='tabs-content')
        ])
    ])
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'active_tab')])
def render_content(tab):
    if tab == 'tab1':
        return firsttab_layout()
    elif tab == 'tab2':
        return secondtab_layout()

# Register callbacks for the second tab
register_callbacks(app)
# Register callbacks for the first tab
register_callbacks_for_first_tab(app)

#
if __name__ == '__main__':
    app.run_server()
