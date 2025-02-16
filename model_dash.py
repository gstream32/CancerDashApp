import numpy as np
from dash import Dash, html, dcc, callback
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

from eda_dash import df_ui


# Load dataset function
def load_data():

    df_load = pd.read_csv('cancer_issue.csv')
    return df_load

# Load data for column selection
df_vis = load_data()

# Create list of cancer types
cancer_types = np.unique(df_vis['CancerType'])

# List of model options
models = ['Logistic Regression', 'SVC', 'Random Forrest', 'KNN']

# Initialize the Dash app
app = Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1('Cancer Model Testing',
            style={
                'textAlign': 'center',
                'color': '#2c3e50',
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '2.5rem',
                'marginTop': '20px',
                'marginBottom': '20px'
            }),

    # Cancer Type Selection Dropdown
    html.Div([
        html.Label('Cancer Selection',
                   style={
                       'fontWeight': 'bold',
                       'marginBottom': '8px',
                       'fontSize': '1rem',
                       'color': '#2c3e50'
                   }),
        dcc.Dropdown(
            id='cancer-dropdown',
            options=cancer_types,
            value=cancer_types[0] if cancer_types else None,
            placeholder='Select a cancer type',
            persistence=True,
            persistence_type='session',
            style={
                'border': '1px solid #ccc',
                'borderRadius': '5px',
                'padding': '5px',
                'marginBottom': '20px',
                'fontSize': '1rem'
            }
        )
    ],
    style={
        'margin': '0 auto',
        'width': '50%',
        'textAlign': 'left'
    }),

    # Model Type Selection Dropdown
    html.Div([
        html.Label('Model Selection',
                   style={
                       'fontWeight': 'bold',
                       'marginBottom': '8px',
                       'fontSize': '1rem',
                       'color': '#2c3e50'
                   }
                   ),
        dcc.Dropdown(
            id='model-dropdown',
            options=models,
            value=models[0] if models else None,
            placeholder='Select a model type',
            persistence=True,
            persistence_type='session',
            style={
                'border': '1px solid #ccc',
                'borderRadius': '5px',
                'padding': '5px',
                'marginBottom': '20px',
                'fontSize': '1rem'
            }
        )
    ]),

    # Submit Button
    html.Button('Run Analysis',
            id='button',
            n_clicks=0,
            style={
                'padding': '12px 24px',
                'fontSize': '16px',
                'backgroundColor': '#2c3e50',
                'color': 'white',
                'border': 'none',
                'borderRadius': '4px',
                'cursor': 'pointer',
                'margin': '20px auto',
                'display': 'block',
                'transition': 'background-color 0.3s',
                ':hover': {'backgroundColor': '#34495e'}
            }
    ),

    # Store data after analysis
    dcc.Store(id='data')

    ]
)