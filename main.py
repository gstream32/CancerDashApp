from dash import Dash, html, dcc, callback
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd


# Load your dataset
def load_data():
    # Example: loading a sample dataset
    df_function = pd.read_csv('cancer_issue.csv')
    return df_function


# Initialize the Dash app
app = Dash(__name__)

# Prepare initial dataset
df = load_data()

# Identify numeric and categorical columns
int_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
int_columns.remove('PatientID')
str_columns = df.select_dtypes(include=['object']).columns.tolist()

# Prepare column options
int_options = [{'label': col, 'value': col} for col in int_columns]
str_options = [{'label': col, 'value': col} for col in str_columns]


# Prepare categorical filter options
def get_categorical_values(column):
    return [{'label': val, 'value': val} for val in df[column].unique()]


# Define the layout of the dashboard
app.layout = html.Div([
    html.H1('Cancer Data Visualization'),

    # Categorical Filter Dropdown
    html.Div([
        html.Label('Category Filter'),
        dcc.Dropdown(
            id='filter-dropdown',
            options=str_options,
            value=str_columns[0] if str_columns else None,
            placeholder='Select a category to filter'
        )
    ]),

    # Category Value Dropdown
    html.Div([
        html.Label('Category Filter Value'),
        dcc.Dropdown(
            id='filter-dropdown-value',
            options=[],
            placeholder='Select a category value'
        )
    ]),

    # Color By Dropdown
    html.Div([
        html.Label('Color By:'),
        dcc.Dropdown(
            id='color-dropdown',
            options=str_options,
            value=None,
            placeholder='Select a categorical column for color'
        )
    ]),

    # X-Axis Dropdown
    html.Div([
        html.Label('Select X-Axis'),
        dcc.Dropdown(
            id='x-axis-dropdown',
            options=int_options,
            value=int_options[0]['value']
        )
    ]),

    # Y-Axis Dropdown
    html.Div([
        html.Label('Select Y-Axis'),
        dcc.Dropdown(
            id='y-axis-dropdown',
            options=int_options,
            value=int_options[1]['value']
        )
    ]),

    # Scatter plot
    dcc.Graph(id='scatter-plot'),

    # Box Plot
    dcc.Graph(id='box-plot'),
])


# Callback to update category value dropdown
@callback(
    Output('filter-dropdown-value', 'options'),
    Input('filter-dropdown', 'value')
)
def update_category_values(selected_category):
    if not selected_category:
        return []
    return get_categorical_values(selected_category)


# Callback to update scatter plot
@callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('filter-dropdown', 'value'),
     Input('filter-dropdown-value', 'value'),
     Input('color-dropdown', 'value'),
     ]
)
def update_scatter_plot(x_column, y_column, filter_column, filter_value, cat):
    df = load_data()

    # Apply category filter if both category and value are selected
    if filter_column and filter_value:
        df = df[df[filter_column] == filter_value]

    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=cat,
        title=f'{x_column} vs {y_column}',
        labels={'x': x_column, 'y': y_column}
    )
    return fig


# Callback to update box plot
@callback(
    Output('box-plot', 'figure'),
    [Input('x-axis-dropdown', 'value'),
     Input('color-dropdown', 'value')]
)

def update_box_plot(int_column, color_column):
    df = load_data()

    if color_column:
        fig = px.box(
            df,
            x=color_column,
            y=int_column,
            title=f'Box Plot of {int_column} by {color_column}',
            labels={'x': color_column, 'y': int_column}
        )
    else:
        fig = px.box(
            df,
            y=int_column,
            title=f'Box Plot of {int_column}',
            labels={'y': int_column}
        )
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)