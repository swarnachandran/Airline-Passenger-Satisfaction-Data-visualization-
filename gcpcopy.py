import dash
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from numpy import linalg as la
from scipy.stats import kstest
from scipy.stats import normaltest
from scipy.stats import shapiro
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# from motionheading import app
import statsmodels.api as sm
import base64
import datetime
import io


# loading the datset
df = pd.read_csv('train .csv')
print(df.head())
print('*' * 100)

# dropped the missing values
df.dropna(inplace=True)
print(df.shape)
print('*' * 100)
print(df.isnull().sum())
print('*' * 100)

# Load external stylesheets and assign tab styles
external_stylesheets = ["https://unpkg.com/purecss@2.1.0/build/pure-min.css"]
my_app = dash.Dash('My App', external_stylesheets=external_stylesheets)
# server = my_app.server
tabs_styles = {
    'height': '45px'
}
tab_style = {
    'borderBottom': '2px solid #C8C8C8',
    'borderTop': '2px solid #C8C8C8',
    'borderRight': '2px solid #C8C8C8',
    'fontWeight': 'bold',
    'backgroundColor': '#73C2FB',
    'textAlign': 'center',
    'color': '#800020',
    'padding': '6px'

}

tab_selected_style = {
    'borderTop': '2px solid #007FFF',
    'borderBottom': '2px solid #007FFF',
    'borderRight': '2px solid #007FFF',
    'padding': '6px',
    'fontWeight': 'bold',
    'backgroundColor': '#C8C8C8',
    'textAlign': 'center',
    'color': '#007FFF'
}
# external_stylesheets = ["https://unpkg.com/purecss@2.1.0/build/pure-min.css"]
# my_app = dash.Dash('My App', external_stylesheets=external_stylesheets)
server = my_app.server
# Design the tab layout
my_app.layout = html.Div(
    style={
        'backgroundColor': '#73C2FB',
        'textAlign': 'center',
        'color': 'burgundy',
    },
    children=[
        html.H1(
            'AIRLINE PASSENGER SATISFACTION',
            style={
                'color': '#800020',
                'display': 'inline-block'}),
        dcc.Tabs(
            id='tabs1',
            children=[
                dcc.Tab(
                    label='Know your Data',
                    value='Know your Data',
                    style=tab_style,
                    selected_style=tab_selected_style),
                dcc.Tab(
                    label='Outlier Analysis',
                    value='Outlier Analysis',
                    style=tab_style,
                    selected_style=tab_selected_style),
                dcc.Tab(
                    label='PCA',
                    value='PCA',
                    style=tab_style,
                    selected_style=tab_selected_style),
                dcc.Tab(
                    label='Normality Tests',
                    value='Normality Tests',
                    style=tab_style,
                    selected_style=tab_selected_style),
                dcc.Tab(
                    label='Heatmap',
                    value='Heatmap',
                    style=tab_style,
                    selected_style=tab_selected_style),
                dcc.Tab(
                    label='Analysis',
                    value='Analysis',
                    style=tab_style,
                    selected_style=tab_selected_style),
                dcc.Tab(
                    label='Graphs',
                    value='Graphs',
                    style=tab_style,
                    selected_style=tab_selected_style),
                dcc.Tab(
                    label='Summary',
                    value='Summary',
                    style=tab_style,
                    selected_style=tab_selected_style)],
            style=tabs_styles,
            value='Know your Data'),
        html.Div(
            id='layout',
            style={
                'backgroundColor': '#0DF66C',
                'color': '#111111',
            })])


# Layput for first tab
tab1_layout = html.Div(style={'backgroundColor': '#73C2FB', 'color': 'white',
                              'width': '100%',
                              'height': '100%'},
                       children=[html.Br(), html.H3('About the Dataset:', style={'color': '#800020', 'margin': '0', 'textAlign': 'left'}),
                                 html.P('The airline passenger satisfaction dataset is a collection of responses from airline passengers regarding their level of satisfaction with various aspects of their air travel experience. The dataset contains 103,904 entries and 25 columns. The dataset includes both categorical and numerical columns. ', style={'display': 'inline-block', 'margin': '0', 'textAlign': 'left'}),
                                 html.Br(),
                                 html.Br(),
                                 html.P('The categorical columns include information such as the customers gender, type of travel, and class of service. The numerical columns include metrics such as the flight distance, the level of inflight service, and the delay times for departure and arrival.', style={'display': 'inline-block', 'margin': '0', 'textAlign': 'left'}),
                                 html.Br(),
                                 html.Br(),
                                 html.P('The dataset was compiled from a survey administered to airline passengers. The survey was designed to assess customer satisfaction with various aspects of air travel, including the booking process, check-in procedures, onboard service, and baggage handling.The ultimate goal of analyzing this dataset is to gain insights into the factors that contribute to customer satisfaction in air travel, and to identify areas where airlines can improve the passenger experience', style={'display': 'inline-block', 'margin': '0', 'textAlign': 'left'}),
                                 html.Hr(style={'border': '1px solid black'}),

                                 html.H3('About the data:', style={'color': '#800020', 'margin': '1px', 'textAlign': 'left'}),
                                 html.P('Click one option to understand the basic information about the data!', style={'margin': '1px', 'textAlign': 'left'}),
                                 dcc.Dropdown(id='infos', options=[
                                     {'label': 'Column Names', 'value': 'Column'},
                                     {'label': 'Count of rows', 'value': 'rows'},
                                     {'label': 'Count of columns', 'value': 'columns'
                                      }],
                           value=' ',
                           placeholder='Select an option',
                           style={'color': '#800020', 'margin-left': '5px', 'width': '50%', 'textAlign': 'left'}),
    html.Plaintext(id='datainfo', style={'backgroundColor': 'white', 'color': 'black', 'font-size': '15px', 'textAlign': 'left'}),
    html.Hr(style={'border': '1px solid black'}),

    html.H3('Data Preprocessing:', style={'color': '#800020', 'margin': '1px', 'textAlign': 'left'}),
    html.P('Choose an option:', style={'margin': '1px', 'textAlign': 'left'}),
    dcc.Dropdown(id='cleans', options=[
        {'label': 'Check for Null values', 'value': 'nulls'},
        {'label': 'Statistics', 'value': 'stats'},
        {'label': 'Data after preprocessing', 'value': 'head_d'}],
                           value='Column',
                           placeholder='Select an option',
                           style={'color': '#800020', 'margin-left': '5px', 'width': '50%', 'textAlign': 'left'}),
    html.Plaintext(id='preprocess', style={'backgroundColor': 'white', 'color': 'black', 'font-size': '15px', 'textAlign': 'left'}),
    html.Hr(style={'border': '1px solid black'}),

    html.H3('Download Data:', style={'color': '#800020', 'margin': '1px', 'textAlign': 'left'}),
    html.P('Click to download cleaned dataset', style={'margin': '1px', 'textAlign': 'left'}),
    html.Button("Download CSV", id="btn_csv", style={'background-color': '#111111', 'color': '#0DF66C'}),
    dcc.Download(id="download-dataframe-csv")

])

# Outlier detection and removal
df1 = df.copy()
cols_out = [
    'Age',
    'Flight Distance',
    'Inflight wifi service',
    'Departure/Arrival time convenient',
    'Ease of Online booking',
    'Gate location',
    'Food and drink',
    'Online boarding',
    'Seat comfort',
    'Inflight entertainment',
    'On-board service',
    'Leg room service',
    'Baggage handling',
    'Checkin service',
    'Inflight service',
    'Cleanliness',
    'Departure Delay in Minutes',
    'Arrival Delay in Minutes']

for i in cols_out:

    q1_h, q2_h, q3_h = df1[i].quantile([0.25, 0.5, 0.75])

    IQR_h = q3_h - q1_h
    lower1 = q1_h - 1.5 * IQR_h
    upper1 = q3_h + 1.5 * IQR_h
    df1 = df1[(df1[i] > lower1) & (df1[i] < upper1)]
    print(f'Q1 and Q3 of the {i} is {q1_h:.2f}  & {q3_h:.2f} \n IQR for the {i} is {IQR_h:.2f} \nAny {i} < {lower1:.2f}  and {i} > {upper1:.2f}  is an outlier')

# Design for second tab layout

tab2_layout = html.Div(style={'backgroundColor': '#73C2FB',
                       'color': 'white'}, children=[
    html.Br(),
    html.H3('Outlier Detection: An analysis of numeric variables using boxplot', style={'color': '#800020', 'margin': '0',
                                                                                        'textAlign': 'center'}),
    html.P('Choose a variables to view the boxplot:',
           style={'margin': '1px', 'textAlign': 'left', "margin-left": "20px"}),
    dcc.Dropdown(
        id='drop1',
        options=[
            {'label': 'Age', 'value': 'Age'},
            {'label': 'Flight Distance', 'value': 'Flight Distance'},
            {'label': 'Inflight wifi service',
             'value': 'Inflight wifi service'},
            {'label': 'Departure/Arrival time convenient',
             'value': 'Departure/Arrival time convenient'},
            {'label': 'Ease of Online booking',
             'value': 'Ease of Online booking'},
            {'label': 'Food and drink', 'value': 'Food and drink'},
            {'label': 'Online boarding', 'value': 'Online boarding'},
            {'label': 'Seat comfort', 'value': 'Seat comfort'},
            {'label': 'Inflight entertainment',
             'value': 'Inflight entertainment'},
            {'label': 'On-board service', 'value': 'On-board service'},
            {'label': 'Leg room service', 'value': 'Leg room service'},
            {'label': 'Baggage handling', 'value': 'Baggage handling'},
            {'label': 'Checkin service', 'value': 'Checkin service'},
            {'label': 'Inflight service', 'value': 'Inflight service'},
            {'label': 'Cleanliness', 'value': 'Cleanliness'},
            {'label': 'Departure Delay in Minutes',
             'value': 'Departure Delay in Minutes'},
            {'label': 'Arrival Delay in Minutes',
             'value': 'Arrival Delay in Minutes'},
        ],
        value=' ',
        placeholder='select an option',
        clearable=False,
        style={'color': '#800020', 'width': '200px', "margin-left": "20px"},
    ),
    html.Br(),
    dcc.Graph(id='graphbox1', style={'color': '#800020',
              'width': '800px', 'height': '500px', "margin-left": "20px"}),
    html.Hr(style={'border': '1px solid black'}),
    html.H3('Outlier Removal:IQR method', style={'color': '#800020',
            'margin': '1px', 'textAlign': 'left'}),
    html.Br(),
    html.H3('Outlier Removal: An analysis of numeric variables using boxplot', style={'color': '#800020', 'margin': '0',
                                                                                      'textAlign': 'left'}),
    html.P('Choose a variables to view the boxplot:',
           style={'margin': '1px', 'textAlign': 'left'}),
    dcc.Dropdown(
        id='drop2',
        options=[
            {'label': 'Age', 'value': 'Age'},
            {'label': 'Flight Distance', 'value': 'Flight Distance'},
            {'label': 'Inflight wifi service',
             'value': 'Inflight wifi service'},
            {'label': 'Departure/Arrival time convenient',
             'value': 'Departure/Arrival time convenient'},
            {'label': 'Ease of Online booking',
             'value': 'Ease of Online booking'},
            {'label': 'Food and drink', 'value': 'Food and drink'},
            {'label': 'Online boarding', 'value': 'Online boarding'},
            {'label': 'Seat comfort', 'value': 'Seat comfort'},
            {'label': 'Inflight entertainment',
             'value': 'Inflight entertainment'},
            {'label': 'On-board service', 'value': 'On-board service'},
            {'label': 'Leg room service', 'value': 'Leg room service'},
            {'label': 'Baggage handling', 'value': 'Baggage handling'},
            {'label': 'Checkin service', 'value': 'Checkin service'},
            {'label': 'Inflight service', 'value': 'Inflight service'},
            {'label': 'Cleanliness', 'value': 'Cleanliness'},
            {'label': 'Departure Delay in Minutes',
             'value': 'Departure Delay in Minutes'},
            {'label': 'Arrival Delay in Minutes',
             'value': 'Arrival Delay in Minutes'},
        ],
        value=' ',
        placeholder='select an option',
        clearable=False,
        style={'color': '#800020', 'width': '200px', "margin-left": "20px"},
    ),
    html.Br(),
    dcc.Graph(id='graphbox2', style={'width': '800px', 'height': '500px',
              "margin-left": "20px"}),
    html.P('Therefore the outliers from the following variables were removed: Flight distance, Checkin service ,Departure Delay in Minutes , Arrival Delay in Minutes', style={'backgroundColor': '#800020', 'color': 'white',
                                                                                                                                                                               'font-size': '20px'}),
    html.Hr(style={'border': '1px solid black'}),
])


# PCA
df = df.drop(['Unnamed: 0', 'id'], axis=1)
df.head()
print(df.isnull().sum())

# Create LabelEncoder object
le = LabelEncoder()

# Encode categorical columns
cat_cols = [
    'Gender',
    'Customer Type',
    'Type of Travel',
    'Class',
    'satisfaction']
for col in cat_cols:
    df[col] = le.fit_transform(df[col])


Features = df._get_numeric_data().columns.to_list()[:-1]
x = df[df._get_numeric_data().columns.to_list()[:-1]]

x = x.values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components='mle', svd_solver='full')
pca.fit(x)
x_pca = pca.transform(x)

# plot of cumsum
number_of_components = np.arange(
    1, len(np.cumsum(pca.explained_variance_ratio_)) + 1)
fig = px.line(
    x=number_of_components,
    y=np.cumsum(
        pca.explained_variance_ratio_))
fig.update_layout(title='Cumulative Explained Variance')

# svd and condition number
H = np.matmul(x.T, x)
_, d, _ = np.linalg.svd(H)


# svd and condition number-tranformed
H_pca = np.matmul(x_pca.T, x_pca)
_, d_pca, _ = np.linalg.svd(H_pca)

# PCA correlation matrix
fig1 = px.imshow(pd.DataFrame(x_pca).corr())
# Better visuals
plt.figure(figsize=(20, 20))
sns.heatmap(pd.DataFrame(x_pca).corr(), annot=True)
plt.title('correlation plot of PCA features')
plt.show()


# Design for third tab
tab3_layout = html.Div(style={'backgroundColor': '#73C2FB', 'color': 'white'},
                       children=[html.Br(),
                                 html.H3('Principal Component Analysis',
                                         style={'color': '#800020', 'margin': '0', 'textAlign': 'left'}),
                                 html.P('Choose options to view outputs of PCA:',
                                        style={'margin': '1px', 'textAlign': 'left'}),
                                 dcc.RadioItems(id='checkpca', options=[
                                     {'label': 'Original Space', 'value': 'Original'},
                                     {'label': 'Transformed Space', 'value': 'tranformed'}], value='Original', inputStyle={'color': '#800020', "margin-left": "20px"}),
                                 html.Plaintext(id='pcaout', style={'backgroundColor': '#800020', 'color': 'white', 'font-size': '15px'}),
                                 html.Hr(style={'border': '1px solid black'}),
                                 html.H3('Cumulative Explained Variance:',
                                         style={'color': '#800020', 'margin': '0', 'textAlign': 'left'}),
                                 html.Br(),
                                 dcc.Graph(figure=fig, style={'width': '800px', 'height': '500px'}),
                                 html.Hr(style={'border': '1px solid black'}),
                                 html.H3('PCA features correlation matrix:',
                                         style={'color': '#800020', 'margin': '0', 'textAlign': 'left'}),
                                 html.Br(),
                                 dcc.Graph(figure=fig1, style={'width': '800px', 'height': '500px'})
                                 ])


# Design for tab4
tab4_layout = html.Div(style={'backgroundColor': '#73C2FB', 'color': 'white'},
                       children=[html.H3('Normality Tests', style={'color': '#800020', 'margin': '0', 'textAlign': 'center'}),
                                 html.Br(),
                                 html.P('Choose variable:', style={'margin': '1px', 'textAlign': 'center'}),
                                 html.Br(),
                                 dcc.Dropdown(id='dropvar',
                                              options=[
                                                  {'label': 'Age', 'value': 'Age'},
                                                  {'label': 'Flight Distance', 'value': 'Flight Distance'},
                                                  {'label': 'Inflight wifi service', 'value': 'Inflight wifi service'},
                                                  {'label': 'Departure/Arrival time convenient',
                                                      'value': 'Departure/Arrival time convenient'},
                                                  {'label': 'Ease of Online booking', 'value': 'Ease of Online booking'},
                                                  {'label': 'Food and drink', 'value': 'Food and drink'},
                                                  {'label': 'Online boarding', 'value': 'Online boarding'},
                                                  {'label': 'Seat comfort', 'value': 'Seat comfort'},
                                                  {'label': 'Inflight entertainment', 'value': 'Inflight entertainment'},
                                                  {'label': 'On-board service', 'value': 'On-board service'},
                                                  {'label': 'Leg room service', 'value': 'Leg room service'},
                                                  {'label': 'Baggage handling', 'value': 'Baggage handling'},
                                                  {'label': 'Checkin service', 'value': 'Checkin service'},
                                                  {'label': 'Inflight service', 'value': 'Inflight service'},
                                                  {'label': 'Cleanliness', 'value': 'Cleanliness'},
                                                  {'label': 'Departure Delay in Minutes', 'value': 'Departure Delay in Minutes'},
                                                  {'label': 'Arrival Delay in Minutes', 'value': 'Arrival Delay in Minutes'},], value=' ', style={'color': '#800020', 'width': '200px', 'margin': '0 auto', 'textAlign': 'center'}, clearable=False),
                                 html.Br(),
                                 html.P('Choose the test', style={'margin': '40px', 'textAlign': 'center'}),
                                 html.Br(),
                                 dcc.Dropdown(id='droptest', options=[
                                     {'label': 'normaltest', 'value': 'normal-test'},
                                     {'label': 'kstest', 'value': 'kstest'},
                                     {'label': 'shapiro', 'value': 'shapiro'}
                                 ], value='normaltest', style={'color': '#800020', 'width': '200px', 'margin': '0 auto', 'textAlign': 'center'}),
                                 html.Br(),
                                 html.Plaintext(id='ntout', style={'backgroundColor': '#800020', 'color': 'white', 'font-size': '15px'}),
                                 html.Hr(style={'border': '1px solid black'}),

                                 ])

# Design tab5 layout
tab5_layout = html.Div(
    style={
        'backgroundColor': '#73C2FB', 'color': 'white'}, children=[
            dcc.Dropdown(
                id='drop_down', options=[
                    {
                        'label': 'Heatmap', 'value': 'Heatmap'}, {
                            'label': 'Scatter matrix', 'value': 'Scatter matrix'}, ], value='Heatmap', style={
                                'color': '#800020', 'width': '200px', 'display': 'inline-block'}), html.H3(
                                    'Heat map(pearson correlation coeeficient & Scatter matrix', style={
                                        'color': '#800020', 'margin': '0', 'textAlign': 'center'}), html.Br(), dcc.Graph(
                                            id='hs', style={
                                                'width': '1200px', 'height': '800px', 'margin-left': '10%', }), ])

# Design for tab6
tab6_layout = html.Div(style={'backgroundColor': '#73C2FB', 'color': 'white'},
                       children=[html.H3('Visualize data using various plots', style={'color': '#800020', 'margin': '0', 'textAlign': 'center'}),
                                 html.Br(),
                                 html.P('Choose variable:', style={'margin': '1px', 'textAlign': 'center', 'display': 'inline-block'}),
                                 html.Br(),
                                 dcc.Dropdown(id='options_dropdown',
                                              options=[
                                                  {'label': 'Age', 'value': 'Age'},
                                                  {'label': 'Flight Distance', 'value': 'Flight Distance'},
                                                  {'label': 'Inflight wifi service', 'value': 'Inflight wifi service'},
                                                  {'label': 'Departure/Arrival time convenient', 'value': 'Departure/Arrival time convenient'},
                                                  {'label': 'Ease of Online booking', 'value': 'Ease of Online booking'},
                                                  {'label': 'Food and drink', 'value': 'Food and drink'},
                                                  {'label': 'Online boarding', 'value': 'Online boarding'},
                                                  {'label': 'Seat comfort', 'value': 'Seat comfort'},
                                                  {'label': 'Inflight entertainment', 'value': 'Inflight entertainment'},
                                                  {'label': 'On-board service', 'value': 'On-board service'},
                                                  {'label': 'Leg room service', 'value': 'Leg room service'},
                                                  {'label': 'Baggage handling', 'value': 'Baggage handling'},
                                                  {'label': 'Checkin service', 'value': 'Checkin service'},
                                                  {'label': 'Inflight service', 'value': 'Inflight service'},
                                                  {'label': 'Cleanliness', 'value': 'Cleanliness'},
                                                  {'label': 'Departure Delay in Minutes', 'value': 'Departure Delay in Minutes'},
                                                  {'label': 'Arrival Delay in Minutes', 'value': 'Arrival Delay in Minutes'},], style={'color': '#800020', 'width': '200px', 'display': 'inline-block'}, clearable=False),
                                 html.P('Choose variable to color:', style={'display': 'inline-block', 'margin': '1px', 'textAlign': 'left'}),
                                 dcc.Dropdown(id='color',
                                              options=[
                                                  {'label': 'Gender', 'value': 'Gender'},
                                                  {'label': 'Customer Type', 'value': 'Customer Type'},
                                                  {'label': 'Type of Travel', 'value': 'Type of Travel'},
                                                  {'label': 'Class', 'value': 'Class'},], value='Gender', style={'color': '#800020', 'display': 'inline-block', 'width': '200px'}, clearable=False),
                                 html.Br(),

                                 html.P('Line Plot:', style={'margin': '1px', 'textAlign': 'center'}),
                                 dcc.Graph(id='line', style={'width': '800px', 'height': '400px', 'margin': '0 auto'}),
                                 html.Br(),
                                 html.P('COUNT PLOT:', style={'margin': '0 auto', 'textAlign': 'center'}),
                                 dcc.Slider(id='bins', min=20, max=100, value=50, tooltip={"placement": "bottom", "always_visible": True}),
                                 html.Br(),
                                 dcc.Graph(id='bar', style={'width': '800px', 'height': '400px', 'margin': '0 auto'}),
                                 html.P('HISTO (WITH VIOLIN&BOX):', style={'textAlign': 'center', 'margin': '0 auto'}),
                                 html.P("Select Distribution:", style={'margin': '1px', 'textAlign': 'left'}),
                                 dcc.RadioItems(
                           id='distribution',
                           options=[
                               {'label': 'box', 'value': 'box'},
                               {'label': 'violin', 'value': 'violin'},
                           ],

                           value='box', inputStyle={'color': '#800020', "margin-left": "20px"}),
    dcc.Graph(id="graphd", style={'width': '800px', 'height': '400px', 'margin': '0 auto'}),
])

# Design for Tab7
tab7_layout = html.Div(style={'backgroundColor': '#73C2FB', 'color': 'white'},
                       children=[html.H3('PIE PLOT', style={'color': '#800020', 'fontWeight': 'bold', 'margin': '1px', 'textAlign': 'center'}),
                                 html.Br(),
                                 html.P('Choose variable:', style={'margin': '0', 'textAlign': 'left', 'display': 'inline-block'}), dcc.Dropdown(id='pie_drop',
                                                                                                                                                 options=[
                                                                                                                                                     {'label': 'Gender', 'value': 'Gender'},
                                                                                                                                                     {'label': 'Customer Type', 'value': 'Customer Type'},
                                                                                                                                                     {'label': 'Type of Travel', 'value': 'Type of Travel'},
                                                                                                                                                     {'label': 'Class', 'value': 'Class'}, ], value='Gender', style={'color': '#800020', 'display': 'inline-block', 'width': '200px'},
                                                                                                                                                 clearable=False),
                                 html.Br(),
                                 dcc.Graph(id='pie', style={'width': '800px', 'height': '400px', 'margin-left': '23%'}),
                                 html.Br(),
                                 html.H3('SCATTER PLOT:', style={'color': '#800020', 'fontWeight': 'bold', 'margin': '1px', 'textAlign': 'center'}),
                                 html.Br(),
                                 html.P('Choose variable:', style={'margin': '1px', 'textAlign': 'left', 'display': 'inline-block'}),
                                 dcc.Dropdown(id='scat_drop',
                                              options=[
                                                  {'label': 'Age', 'value': 'Age'},
                                                  {'label': 'Flight Distance', 'value': 'Flight Distance'},
                                                  {'label': 'Inflight wifi service', 'value': 'Inflight wifi service'},
                                                  {'label': 'Departure/Arrival time convenient', 'value': 'Departure/Arrival time convenient'},
                                                  {'label': 'Ease of Online booking', 'value': 'Ease of Online booking'},
                                                  {'label': 'Food and drink', 'value': 'Food and drink'},
                                                  {'label': 'Online boarding', 'value': 'Online boarding'},
                                                  {'label': 'Seat comfort', 'value': 'Seat comfort'},
                                                  {'label': 'Inflight entertainment', 'value': 'Inflight entertainment'},
                                                  {'label': 'On-board service', 'value': 'On-board service'},
                                                  {'label': 'Leg room service', 'value': 'Leg room service'},
                                                  {'label': 'Baggage handling', 'value': 'Baggage handling'},
                                                  {'label': 'Checkin service', 'value': 'Checkin service'},
                                                  {'label': 'Inflight service', 'value': 'Inflight service'},
                                                  {'label': 'Cleanliness', 'value': 'Cleanliness'},
                                                  {'label': 'Departure Delay in Minutes', 'value': 'Departure Delay in Minutes'},
                                                  {'label': 'Arrival Delay in Minutes', 'value': 'Arrival Delay in Minutes'}],
                                              value='Age', style={'color': '#800020', 'margin': '10px', 'display': 'inline-block', 'width': '200px'}, clearable=False),
                                 html.P('Choose variable to color:', style={'display': 'inline-block', 'margin': '1px', 'textAlign': 'left'}),
                                 dcc.Dropdown(id='color',
                                              options=[
                                                  {'label': 'Age', 'value': 'Age'},
                                                  {'label': 'Flight Distance', 'value': 'Flight Distance'},
                                                  {'label': 'Inflight wifi service', 'value': 'Inflight wifi service'},
                                                  {'label': 'Departure/Arrival time convenient',
                                                   'value': 'Departure/Arrival time convenient'},
                                                  {'label': 'Ease of Online booking', 'value': 'Ease of Online booking'},
                                                  {'label': 'Food and drink', 'value': 'Food and drink'},
                                                  {'label': 'Online boarding', 'value': 'Online boarding'},
                                                  {'label': 'Seat comfort', 'value': 'Seat comfort'},
                                                  {'label': 'Inflight entertainment', 'value': 'Inflight entertainment'},
                                                  {'label': 'On-board service', 'value': 'On-board service'},
                                                  {'label': 'Leg room service', 'value': 'Leg room service'},
                                                  {'label': 'Baggage handling', 'value': 'Baggage handling'},
                                                  {'label': 'Checkin service', 'value': 'Checkin service'},
                                                  {'label': 'Inflight service', 'value': 'Inflight service'},
                                                  {'label': 'Cleanliness', 'value': 'Cleanliness'},
                                                  {'label': 'Departure Delay in Minutes',
                                                   'value': 'Departure Delay in Minutes'},
                                                  {'label': 'Arrival Delay in Minutes',
                                                   'value': 'Arrival Delay in Minutes'}, ], value='Age', style={'color': '#800020', 'margin': '10px', 'display': 'inline-block', 'width': '200px'},
                                              clearable=False),
                                 html.Br(),
                                 html.Br(),
                                 dcc.Graph(id="scatpd", style={'width': '800px', 'height': '400px', 'margin-left': '23%'}),
                                 html.Br(),
                                 dcc.Graph(
                           id='kde-plot',
                           figure={
                               'data': [
                                   {
                                       'y': df['satisfaction'],
                                       'x': df['Age'],
                                       'type': 'histogram',
                                       'name': 'satisfaction',
                                       'histnorm': 'probability density'
                                   },
                               ],
                               'layout': {
                                   'title': ' KDE Plot',
                                   'xaxis': {'title': 'Age'},
                                   'yaxis': {'title': 'satisfaction'}
                               },
                           },
                           style={'width': '800px', 'height': '400px', 'margin-left': '23%'},
                       ),

])

# cat plot
sns.catplot(x="satisfaction", kind="count", hue="Class", data=df)
plt.show()
# qqplot
sm.qqplot(df["satisfaction"], line="s")
plt.show()
# mbox
sns.boxplot(data=df, x="Age", hue="satisfaction", y="Customer Type")
plt.show()
# tab8 design
tab8_layout = html.Div(style={'backgroundColor': '#73C2FB', 'color': 'white'},
                       children=[html.H3('Summary', style={'color': '#800020', 'margin': '0', 'textAlign': 'left'}),
                                 html.Br(),
                                 html.P(
                           'The developed dash application helps user to visualize the analysis of the satisfaction of the passenger in the airline travel.\n  All the variable depends upon the target variable satisfaction.\n Observing the results, the user can easily identify the satisfactory level of customer under \n diffferent aspects',
                           style={
                               'margin': '1px',
                               'textAlign': 'left'}),
    html.Hr(style={'border': '1px solid black'}),
    html.H3(
                           'References',
                           style={
                               'color': '#800020',
                               'margin': '0',
                               'textAlign': 'left'}),
    html.Br(),
    html.Plaintext(
                           ' 1.https://dash.plotly.com/dash-core-components \n 2.https://dash.plotly.com/dash-html-components \n 3.https://dash.plotly.com/advanced-callbacks \n 4.https://plotly.com/python/box-plots/ \n 5.https://plotly.com/python/histograms/ \n 6. https://plotly.com/python/distplot/',
                           style={
                               'margin': '1px',
                               'textAlign': 'left'}),
    html.Hr(style={'border': '1px solid black'}),
    html.H3(
                           'Author Information',
                           style={
                               'color': '#800020',
                               'margin': '0',
                               'textAlign': 'left'}),

    html.Plaintext(
                           'Please feel free to drop an email if you have any questions or suggestions to improve the app!\nCreated by: Pon swarnalaya Ravichandran\nEmail:swarnalaya177@gwu.edu',
                           style={
                               'backgroundColor': '#800020',
                               'color': 'white',
                               'font-size': '15px'}),
    html.Plaintext(
                           'Data Source: Kaggle \nhttps://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?select=train.csv \n *The app has been created for 6401-Visulaization of Complex Data coursework at The George Washington University*',
                           style={
                               'backgroundColor': '#800020',
                               'color': 'white',
                               'font-size': '15px'}),
    dcc.Textarea(id='text-box',
                           placeholder='Place your suggestions',
                           value='',
                           style={'width': '40%'}
                 ),
    html.Br(),
    html.Button('Submit', id='submit-val', n_clicks=0),
    dcc.Upload(
                           id='upload-data',
                           children=html.Div([
                               'Upload csv ',
                               html.A('Select Files')
                           ]),
                           style={
                               'width': '30%',
                               'height': '60px',
                               'lineHeight': '60px',
                               'borderWidth': '1px',
                               'borderStyle': 'dashed',
                               'borderRadius': '5px',
                               'textAlign': 'center',
                               'margin': '50px auto'
                           },
                           # Allow multiple files to be uploaded
                           multiple=True,
                       ),
    html.Div(id='output-data-upload'),
])
# Main callback for the main layout


@my_app.callback(Output(component_id='layout', component_property='children'),
                 [Input(component_id='tabs1', component_property='value')
                  ])
def update_layout(tabselect):
    if tabselect == 'Know your Data':
        return tab1_layout
    elif tabselect == 'Outlier Analysis':
        return tab2_layout
    elif tabselect == 'PCA':
        return tab3_layout
    elif tabselect == 'Normality Tests':
        return tab4_layout
    elif tabselect == 'Heatmap':
        return tab5_layout
    elif tabselect == 'Analysis':
        return tab6_layout
    elif tabselect == 'Graphs':
        return tab7_layout
    elif tabselect == 'Summary':
        return tab8_layout
# Callback for tab1


@my_app.callback(Output(component_id='datainfo',
                        component_property='children'),
                 [Input(component_id='infos',
                        component_property='value')])
def update_graph(input):
    if input == 'Column':
        cols = df.columns
        return ['\n' + j for j in cols]
    elif input == 'rows':
        i = len(df)
        return f'Number of rows:{i}'
    elif input == 'columns':
        cols = df.columns
        return f'Number of columns:{len(cols)}'


@my_app.callback(Output(component_id='preprocess',
                        component_property='children'),
                 [Input(component_id='cleans',
                        component_property='value')])
def update_graph1(input):
    if input == 'nulls':
        df.isnull().sum()
        df.dropna(inplace=True)
        d = df.isnull().sum()
        return f"{d}\nDataset doesn't have missing values"
    if input == 'stats':
        return f'{df.describe()}'
    elif input == 'head_d':
        return f'{df.head()}'


@my_app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(df.to_csv, "train .csv")

# Callbacks fro tab2 components


@my_app.callback(Output(component_id='graphbox1', component_property='figure'),
                 [Input(component_id='drop1', component_property='value')])
def update_graph2(input):
    fig = px.box(df, y=input)
    fig.update_layout(title='Box plot')
    return fig


@my_app.callback(Output(component_id='graphbox2', component_property='figure'),
                 [Input(component_id='drop2', component_property='value')])
def update_graph3(input):
    fig = px.box(df1, y=input)
    fig.update_layout(title='Box plot')
    return fig

# PCA callbacks


@my_app.callback(Output(component_id='pcaout', component_property='children'),
                 [Input(component_id='checkpca', component_property='value')])
def update_graph4(input):
    if input == 'Original':
        return f'Features:{Features[:8]}\n{Features[8:16]}\n{Features[16:25]}\n\nOriginal Shape:{x.shape}\n\nSingular values:{d}\n\nCondition number:{la.cond(x)}'
    elif input == 'tranformed':
        return f'Transformed shape:{x_pca.shape}\n\nSingular values:{d_pca}\n\nCondition number:{la.cond(x_pca)}\n\nExplained Variance Ratio:{pca.explained_variance_ratio_}'

# Normality callbacks


@my_app.callback(
    Output(component_id='ntout', component_property='children'),
    [Input(component_id='dropvar', component_property='value'),
     Input(component_id='droptest', component_property='value')]
)
def tests(inp, inp2):
    f1 = df[inp]
    if inp2 == 'normal-test':
        return f'Normal test:{normaltest(f1)}'
    elif inp2 == 'kstest':
        ks = kstest(f1, 'norm')
        return f'KS test:{ks}'
    else:
        return f'Shapiro Wilk Test:{shapiro(f1)}'

# tab heatmap


@my_app.callback(
    Output(component_id='hs', component_property='figure'),
    [Input(component_id='drop_down', component_property='value')]
)
def update_graph5(input):
    if input == 'Heatmap':
        fig = px.imshow(df.corr())
        return fig
    if input == 'Scatter matrix':
        f = [
            'Inflight wifi service',
            'Seat comfort',
            'On-board service',
            'Checkin service',
            'Departure Delay in Minutes',
            'Arrival Delay in Minutes',
            'Baggage handling',
            'Departure/Arrival time convenient']
        fig = px.scatter_matrix(
            df,
            dimensions=f,
            labels={
                'Inflight wifi service': 'Infwifiserv',
                'Seat comfort': 'Stcfrt',
                'On-board service': 'On-bserv',
                'Checkin service': 'Chckserv',
                'Departure Delay in Minutes': 'Dep-Dly/min',
                'Arrival Delay in Minutes': 'Arr-Dly/min',
                'Baggage handling': 'Bagsserv',
                'Departure/Arrival time convenient': 'Dep/ArrTime'})
        return fig


# Analysis callbacks
@my_app.callback(Output(component_id='line', component_property='figure'),
                 Output(component_id='bar', component_property='figure'),
                 Output(component_id='graphd', component_property='figure'),
                 [Input(component_id='options_dropdown', component_property='value'),
                  Input(component_id='color', component_property='value'),
                  Input(component_id='bins', component_property='value'),
                  Input(component_id='distribution', component_property='value')])
def update_graph6(input, inp2, inp3, inp4):
    fig = px.line(df, x=input, y='satisfaction', color=inp2)  # line
    fig1 = px.histogram(df, x=input, nbins=inp3)  # count
    fig3 = px.histogram(
        df,
        x=input,
        y='satisfaction',
        color=inp2,
        marginal=inp4)  # histo
    return fig, fig1, fig3

# graphs callbacks


@my_app.callback(Output(component_id='pie', component_property='figure'),
                 Output(component_id='scatpd', component_property='figure'),
                 [Input(component_id='pie_drop', component_property='value'),
                  Input(component_id='color', component_property='value'),
                  Input(component_id='scat_drop', component_property='value')])
def update_graph7(inp, inp2, inp3):
    pie1 = px.pie(df, names=inp, values='satisfaction')
    scat1 = px.scatter(df, x=inp2, color=inp3, trendline='ols')
    return pie1, scat1
# summary


@my_app.callback(
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    prevent_initial_call=True,
)
def update_output(n_clicks, value):
    return 'The input value was "{}" and the button has been clicked {} times'.format(
        value, n_clicks)


def parse_contents(contents, filename, date):
    content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


@my_app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output1(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

if __name__ == '__main__':
    my_app.run_server(debug=True, host='0.0.0.0', port=8080)

