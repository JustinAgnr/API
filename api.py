import flask
import numpy as np
from flask import request, jsonify, render_template
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV,train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDClassifier
from question2 import *

app = flask.Flask(__name__)
app.config["DEBUG"] = True

########################################################
#################### HOME PAGE #########################
########################################################


@app.route('/', methods=['GET'])
# JUST DISPLAY THE HOME PAGE # redirect for the other pages
def home():
    message = "Hello, World"
    return render_template('index.html', message=message)


#######################################################
###################### FUNCTION ########################
########################################################

@app.route('/functions', methods=['GET'])
# DISPLAY ALL THE FUNCTIONS IMPLEMENTED FUNCTIONS
def api_functions():
    return '''<h1> List of functions for our linear regression estimator made from scratch</h1>
    <ul>
    <li>fit(self, X, y, method, learning_rate=0.01, iterations=500, batch_size=32)</li>
    <ul>
        <li> <b>Purpose</b> : fit the model to the data and determine the weights of the model </li>

        <li> Training data : X{array-like, sparse matrix} of shape (n_samples, n_features)</li>
        <li> Target values : yarray-like of shape (n_samples,) or (n_samples, n_targets)</li>
        <li> method : string : ["ols","svg"] for Ordinary Least Square Estimator and Stochastic Gradient Descent</li>
        <li> learning_rate : float : parameter for the sdg method </li>
        <li> iterations : integer : parameter for the sdg</li>
        <li> batch_size : integer : parameter for the sdg method </li>
    </ul>
    <li>predict(self, X)</li>
    <ul>
        <li> <b>Purpose</b> : based on the fit method the predict method on use the weights and then predict new label with new features </li>
    </ul>
    <li>rmse(self, X, y)</li>
    <ul>
        <li> <b>Purpose</b> : return the root mean squared error </li>

    </ul>
    <li> get_params(self) </li>
    <ul>
        <li> <b>Purpose</b> : return the weights of the estimor that we determined thanks to the fit method </li>
    </ul>
    </ul>'''


########################################################
####################### PROCESS ########################
########################################################

@app.route('/process/<n_samples>/<n_features>', methods=['GET'])
def regression(n_samples,n_features):

    # STEP 1 : generate a dataset

    X, y = make_regression(int(n_samples),int(n_features),noise=1,random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)


    # STEP 2 : builds a model and returns a json

    reg = LinearRegression()
    reg.fit(X_train,y_train)

    # STEP 3 : return the statistics, the error rate and the final predictions

    return {"weights": reg.coef_.tolist(),
    "expected_labels":y_test.tolist(),
    "predictions": reg.predict(X_test).tolist(),
    "rmse" : mean_squared_error(y_test,reg.predict(X_test),squared=False)
    }


# ########################################################
# ##################### CLASSIFICATION ###################
# ########################################################

@app.route('/classification/process/<n_samples>/<n_features>', methods=['GET'])
def classification(n_samples,n_features):

    if int(n_features) < 2:
        return "n_features must be superior or equal to 2"
    elif int(n_samples) < 4:
        return "n_features must be superior or equal to 4"


    X, y = make_classification(int(n_samples),int(n_features),flip_y=0.2,n_redundant=0,n_informative=2)

    # STEP 2: builds a model and returns a json

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=10)
    clf.fit(X_train, y_train)



    return {
    "expected_labels":y_test.tolist(),
    "prediction":clf.predict(X_test).tolist(),
    "score" : clf.score(X_test,y_test)
    }

# ########################################################
# ###################### REGSCRATCH ######################
# ########################################################

@app.route('/regression/process/<n_samples>/<n_features>', methods=['GET'])
def class_regression(n_samples,n_features):

    # STEP 1 : generates a dataset

    X, y = make_regression(int(n_samples),int(n_features),noise=1,random_state=0)

    # STEP 2 : builds a model and returns a json

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    reg = LinearRegressionScratch()
    reg.fit(X_train,y_train,'ols')

    return {"weight": reg.get_weights().ravel().tolist(),
    "expected" : y_test.tolist(),
    "prediction":reg.predict(X_test).ravel().tolist(),
    "rmse" : reg.rmse(X_test,y_test)}



app.run(host='0.0.0.0',port=80)
