from dash import Dash, html, dcc, callback
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Load dataset
def load_data():

    df_load = pd.read_csv('cancer_issue.csv')
    return df_load

# Initialize the Dash app
app = Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1('Cancer Model Testing'),


    ]
)