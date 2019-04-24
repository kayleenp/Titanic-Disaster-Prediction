import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import math, time, random, datetime


# Data Manipulation
from PySide.QtCore import *
from PySide.QtGui import *
import sys
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
import matplotlib
import missingno
import seaborn as sns
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
# Machine learning

from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import base64
import datetime
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly.plotly as py





train = pd.read_csv('C:/Users/User/Downloads/train.csv') #untuk baca file csv dengan memakai panda

PAGE_SIZE = 5
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'] #klo misalkan dash biasa digunakan stylesheet (biar bsa atur table mau gmna buat layout)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets) #declare aplikasi app= dash buat di run dibawahnya
app.layout = html.Div([# buat bikin layout
    dcc.Upload(
        id='upload-data',#buat bsa upload data
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),

       
        style={#untuk mempercantik style
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True,# untuk bisa upload lebih dari 1 file
    ),
    html.Div(id='output-data-upload', style = {'display' : 'inline-block', 'margin-left':150}, className = "col s6"),#output
    html.Div(id='output-data-upload-predict', style = {'display' : 'inline-block','margin-left' :50},className = "col s6"),#upload data hasil prediksi
    html.Div(dcc.Graph(id='output-graph'))#output graph
], className = "row"
    )

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})



def fitting( algo, X_train, y_train, cv): 
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train)*100,2)
    
    train_pred = model_selection.cross_val_predict(algo, X_train, y_train, cv=cv, n_jobs =-1)
    if algo == LinearRegression():
        acc_cv = accuracy_score(y_train, train_pred.round(), normalize=False)
        
    else:
        acc_cv= round(metrics.accuracy_score(y_train, train_pred)*100,2)
    

    return train_pred, acc, acc_cv
def parse_contents(contents, filename, date):#untuk menerjemahkan apa yang di upload ke sebuah content
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)#untuk mencari tahu path location
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            test=pd.read_csv(io.StringIO(decoded.decode('utf-8')))
           
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            test = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:#untuk eror handling 
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    filename = "Uploaded File" 

    return html.Div([#untuk mereturn hasil data table yang di upload
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=test.to_dict('rows'),
            columns=[{'name': i, 'id': i} for i in test.columns],
             style_cell_conditional = [ 
                { 
                    'if' : {'row_index' : 'odd'}, 
                    'backgroundColor' : 'rgb(248,248,248)'
                }
                ] +[ 
                    {
                        'if' : {'column_id' :c}, 
                        'textAlign' : 'left'
                    }for c in ['Age', 'PassengerId']
                ],
            n_fixed_rows=1,
            style_header = {'backgroundColor' : 'white', 'fontWeight' : 'bold'}, 
            style_cell = {'width' : '100px', 'textAlign' :'left'}, 
            style_table={
                    'maxHeight': '300px' ,
                    'maxWidth' : '500px',
                    'overflowY' :'scroll', 
                    'border' : 'thin lightgrey solid'}
        ),
         html.Hr()
    ])
def parse_dataframe(contents, filename):
    
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            test=pd.read_csv(io.StringIO(decoded.decode('utf-8')))
           
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            test = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    filename = "Uploaded File" 

    return test
   
        
    

def parse_contents_prediction(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            test=pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            dataset_bin = pd.DataFrame() 
            dataset_bin['Survived'] = train['Survived']
            dataset_bin['Pclass'] = train['Pclass']
            dataset_bin['SibSp'] = train['SibSp']   
            dataset_bin['Parch'] = train['Parch'   ]
            dataset_bin['Fare'] = train['Fare']
            dataset_bin['Embarked'] = train['Embarked']
            dataset_bin['Age'] = train['Age']
            dataset_bin.Age.fillna(dataset_bin.Age.mean(),inplace=True)
            dataset_bin['Age']=dataset_bin['Age'].round()
            dataset_bin = dataset_bin.dropna(subset=['Embarked']) #
            dataset_bin_enc = dataset_bin.apply(LabelEncoder().fit_transform) #
            chosen_ds = dataset_bin_enc
            X_train = chosen_ds.drop('Survived', axis=1)
            y_train = chosen_ds.Survived

            train_pred_dt, acc_dt, acc_cv_dt = fitting(DecisionTreeClassifier(), 
                                                                X_train, 
                                                                y_train,
                                                                10)
            dtc = DecisionTreeClassifier().fit(X_train, y_train)
#2. Choose Decision Trees
            chosen_test_col = X_train.columns

            predictions =dtc.predict(test[chosen_test_col]
                                     .apply(LabelEncoder().fit_transform))
          
                
            submission = pd.DataFrame() 
            submission['PassengerId'] = test['PassengerId']
            submission['Pclass'] = test['Pclass']
            submission['Name'] = test['Name']
            submission['Sex'] = test['Sex']
            submission['SibSp'] = test['SibSp']
            submission['Parch'] = test['Parch']
            submission['Ticket'] = test['Fare']
            submission['Embarked'] = test['Embarked']
            submission['Survived'] = predictions
            submission['Survived'].replace(0, 'No', inplace = True)
            submission['Survived'].replace(1, 'Yes', inplace = True)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            test = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    filename = "Prediction Result"

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=submission.to_dict('rows'),
            columns=[{'name': i, 'id': i} for i in submission.columns],
            style_cell_conditional = [ 
                { 
                    'if' : {'row_index' : 'odd'}, 
                    'backgroundColor' : 'rgb(248,248,248)'
                }
                ] +[ 
                    {
                        'if' : {'column_id' :c}, 
                        'textAlign' : 'left'
                    }for c in ['Age', 'PassengerId']
                ],
            n_fixed_rows=1,
            style_header = {'backgroundColor' : 'white', 'fontWeight' : 'bold'}, 
            style_cell = {'width' : '100px', 'textAlign' :'left'},
            style_table={
                    'maxHeight': '300px' ,
                    'maxWidth' : '500px',
                    'overflowY' :'scroll', 
                    'border' : 'thin lightgrey solid'}
        ),
         html.Hr(),
    ])

 
@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
    return children


@app.callback(Output('output-data-upload-predict', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])

def update_output_prediction(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents_prediction(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
    return children

@app.callback(Output('output-graph', 'figure'), 
              [Input('upload-data', 'contents')], 
              [State('upload-data', 'filename')])

def update_figure(dataframe, filename):  
     df = [parse_dataframe(c,n,d) 
     for c,n,d in 
           zip(dataframe,filename)]
     traces = []
     for i in df.continent.unique():
         df_by_continent = df[df['continent']== i]
         traces.append(go.Scatter( 
                 x = df_by_continent['Survived'] ,
                 y = df_by_continent['Age'], 
                 text = df_by_continent['Age'],
                 mode='markers',
            opacity=0.7,
            marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=i ))
     return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'log', 'range' : [0,1] },
            yaxis={'title': 'Age', 'range': [1, 90]},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }
     
     class MyWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self, None)

        vbox = QVBoxLayout(self)

        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setRange(0, 99)
        self.slider1.setValue(0)
        vbox.addWidget(self.slider1)

        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setRange(0, 99)
        self.slider2.setValue(99)
        vbox.addWidget(self.slider2)

        self.slider1.valueChanged.connect(self.slider2Changed)

    def slider2Changed(self, position):
        self.slider2.setValue(self.slider2.maximum() - position)

def main():
        app = QApplication(sys.argv)
        w = MyWindow()
        w.show()
        app.exec_()
        sys.exit(0)


if __name__ == '__main__':
    app.run_server(debug=True)
    

