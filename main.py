import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Load data
def load_data():
    df = pd.read_csv('cancer_issue.csv')
    return df

# Initialize Dash
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    # Title
    html.H1("Cancer Dashboard"),

    # Dropdown for x
    html.Div([
        html.Label('Select x-axis:'),
        dcc.Dropdown(
            id='x-axis-dropdown',
            options=[],
            value=None
        )
    ]),

    #Dropdown for y
    html.Div([
        html.Label('Select y-axis:'),
        dcc.Dropdown(
            id='y-axis-dropdown',
            options=[],
            value=None
        )
    ]),
    # Scatter plot
    dcc.Graph(id='scatter-plot'),

    # Other viz options after testing
]) # closing layout options

app.callback(
    [Output('x-axis-dropdown', 'options'),
     Output('y-axis-dropdown', 'options')],
    [Input('x-axis-dropdown', 'value')]
)

def update_dropdown_options(selected_x):
    # Load df
    df = load_data()

    # Numeric column list for dropdown
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Options for dropdowns
    options = [{'label': col, 'value': col} for col in numeric_cols]

    return options, options

# Call to update scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')]
)

def update_scatter_plot(x, y):
    # Load df
    df = load_data()

    # Create scatter
    if x and y:
        fig = px.scatter(
            df,
            x=x,
            y=y,
            title=f'{x} vs {y}',
            labels={'x': x, 'y': y}
        )
        return fig
    # Return empty if columns aren't selected
    return {}
# Run app
if __name__ == '__main__':
 app.run_server(debug=True)