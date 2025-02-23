import numpy as np
from dash import Dash, html, dcc, callback, exceptions
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from model_creation import log_reg, svm_svc, clf



# Load dataset function
def load_data():

    df_load = pd.read_csv('cancer_data.csv')
    return df_load

# Load data for column selection
df_vis = load_data()

# Identify numeric and categorical columns
int_columns = df_vis.select_dtypes(include=['float64', 'int64']).columns.tolist()
int_columns.remove('id')
int_columns = sorted(int_columns)
str_columns = df_vis.select_dtypes(include=['object']).columns.tolist()

# Prepare column options
int_options = [{'label': col, 'value': col} for col in int_columns]
str_options = [{'label': col, 'value': col} for col in str_columns]

# List of model options
models = ['Logistic Regression', 'SVC', 'Random Forest']

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
            value=models,
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
## TODO add data scaler drop down
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

    # Line divider
    html.Hr(),

    # Wrapping pieces in dcc.loading to show spinner
    dcc.Loading(
        id='loading',
        type='circle',
        children=[

    # Store data for analysis
    dcc.Store(id='data'),
    dcc.Store(id='modeled-data'),

    # Model Metrics
    html.Div([
        html.Div([
            html.H2(
                id='f1-score',
                children='F1 Score: NA',
                style={
                    'textAlign': 'center'
                }
            ),
            html.P("F1 Value",
                   style={
                       'textAlign': 'center'
                   })
        ], style={
            'flex': '1',
            'border': '2px solid #3498db',
            'borderRadius': '8px',
            'padding': '20px',
            'margin': '10px',
            'boxShadow': '2px 2px 5px rgba(0,0,0,0.2)',
            'backgroundColor': '#ecf0f1'
        }),
        html.Div([
            html.H2(
                id='precision-score',
                children='Precision Score: NA',
                style={
                    'textAlign': 'center'
                }
            ),
            html.P("Precision Value",
                   style={
                       'textAlign': 'center'
                   })
        ], style={
            'flex': '1',
            'border': '2px solid #3498db',
            'borderRadius': '8px',
            'padding': '20px',
            'margin': '10px',
            'boxShadow': '2px 2px 5px rgba(0,0,0,0.2)',
            'backgroundColor': '#ecf0f1'
        }),
        html.Div([
            html.H2(
                id='recall-score',
                children='Recall Score: NA',
                style={
                    'textAlign': 'center'
                }
            ),
            html.P("Recall Value",
                   style={
                       'textAlign': 'center'
                   })
        ], style={
            'flex': '1',
            'border': '2px solid #3498db',
            'borderRadius': '8px',
            'padding': '20px',
            'margin': '10px',
            'boxShadow': '2px 2px 5px rgba(0,0,0,0.2)',
            'backgroundColor': '#ecf0f1'
        })
    ]),
## TODO add scaled and normal scatter plots
    # Scatter plot options
    html.Div([
        # X-Axis Dropdown
        html.Div([
            html.Label('Select X-Axis'),
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=int_options,
                value=None,
                placeholder='X-Axis',
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

        # Y-Axis Dropdown
        html.Div([
            html.Label('Select Y-Axis'),
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=int_options,
                value=None,
                placeholder='Y-Axis',
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
        ])
    ]),

    # Scatter plot
    dcc.Graph(id='scatter-plot'),
            ]
    )

    ],
    # background color
    style={'backgroundColor': '#f2f2f2'}
)

@callback(
    Output('data', 'data'),
        Input('button', 'n_clicks')
)

def filter_data(n_clicks):
    """Filters data based on dropdown values"""

    if n_clicks == 0:
        raise exceptions.PreventUpdate

    df = load_data()

    # Convert target column to binary
    df['diagnosis'] = df['diagnosis'].map(
        {'M': 1, 'B': 0}
    )

    return df.to_dict('records')

@callback(
    [
        Output('modeled-data', 'data'),
        Output('f1-score', 'children'),
        Output('recall-score', 'children'),
        Output('precision-score', 'children')
        ],
    [
        Input('data', 'data'),
        Input('model-dropdown', 'value')
    ]
)

def create_model(data, model_choice):
    """Takes filtered data and fits model"""

    if not data:
        raise exceptions.PreventUpdate

    if model_choice == "Logistic Regression":
       data, f1, recall, precision = log_reg(data, 'diagnosis')

    elif model_choice == "SVC":
        data, f1, recall, precision = svm_svc(data, 'diagnosis')

    elif model_choice == "Random Forest":
       data, f1, recall, precision = clf(data, 'diagnosis')

    else:
        raise ValueError("Incorrect Model Selection")

    return data, f1, recall, precision


@callback(
    Output('scatter-plot', 'figure'),
    [
        Input('modeled-data', 'data'),
        Input('x-axis-dropdown', 'value'),
        Input('y-axis-dropdown', 'value')
    ]
)

def scatter_plot(data, x, y):
    """Creates and returns scatterplot based on modeled data
    and x and y inputs from dropdown"""

    if not data:
        raise exceptions.PreventUpdate

    df = pd.DataFrame(data)

    #Create scatterplot
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color='accuracy',
        hover_data=['accuracy', 'prediction', x, y],
        color_continuous_scale='Viridis'
    )

    #Update layout
    fig.update_layout(
        title=f'Scatterplot of {y} vs {x}',
        xaxis_title=x,
        yaxis_title=y,
        font={
            'family': 'Arial, sans-serif',
            'size': 12,
            'color': '#333333'
        },
        title_font={
            'family': 'Arial, sans-serif',
            'size': 14,
            'color': '#222222'
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(245,245,245,1)',

    )

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)