from dash import Dash, html, dcc, callback
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd


# Load your dataset
def load_data():
    # Load data
    df = pd.read_csv('cancer_issue.csv')
    return df


# Initialize Dash
app = Dash(__name__)

# Prepare initial dataset and column options
df = load_data()
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns.remove('PatientID')
column_options = [{'label': col, 'value': col} for col in numeric_columns]

# Define dash layout
app.layout = html.Div([
    html.H1('Cancer data visualization'),

    html.Div([
        html.Label('Select x-Axis:'),
        dcc.Dropdown(
            id='x-axis-dropdown',
            options=column_options,
            value=column_options[0]['value']
        )
    ]),

    html.Div([
        html.Label('Select y-Axis:'),
        dcc.Dropdown(
            id='y-axis-dropdown',
            options=column_options,
            value=column_options[1]['value']
        )
    ]),

    dcc.Graph(id='scatter-plot'),
])


# Callback to update scatter plot
@callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')]
)
def update_scatter_plot(x, y):
    df = load_data()

    fig = px.scatter(
        df,
        x=x,
        y=y,
        title=f'{x} vs {y}',
        labels={'x': x, 'y': y}
    )
    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)