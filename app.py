#importing required pakages
from flask import Flask, render_template, request , send_file
import pickle
#import io
#import base64
#from   matplotlib.backend.backend_agg import FigurecanvasAgg as FigureCanvas

import seaborn as sns
import matplotlib.pyplot as plt




#load finaldata
model_df1 = pickle.load((open('model1.pkl', 'rb')))
#load customer data
model_df2 = pickle.load((open('model2.pkl', 'rb')))


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    var1 = request.form['CustomerID']              # customer id from html form

    data = model_df1.loc[model_df1['CustomerID'] == int(var1)]      # customer data of given id
    pre = data.iat[0, 7]                                        # predicted november 0/1



    return render_template('home.html', cid=var1, value=pre)    # return value to html

@app.route('/visualisation', methods=['POST','GET'])
def visualisation():
    idc = request.form['CustomerID']
    iddata = model_df2.loc[model_df2['CustomerID'] == int(idc)]  # customer's past data
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.set_style(style='darkgrid')
    sns.barplot(x='month', y='Totalprice', data=iddata)  # month vs total purchase price
    plt.xticks(rotation=60)  # x-axis label at 60degree
    plt.title("Montly purchase by customer")
    plt.show()


    return render_template(fig)
@app.errorhandler(IndexError)
def index_error(e):
    var1 = request.form['CustomerID']
    return render_template('home.html',cid=var1,dis=' Please enter valid Customer ID ')

@app.errorhandler(ValueError)
def value_error(e):
    return render_template('home.html',dis=' Please enter valid Customer ID ')

@app.errorhandler(TypeError)
def type_error(e):

    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
