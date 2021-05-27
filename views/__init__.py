from flask import render_template, request, redirect, url_for, session
from app import app
from model import *
from model.main import *
import json
import pandas as pd
import numpy as np

class DataStore():
    model=None
    model_month=None
    sale_model=None

data = DataStore()

@app.route('/', methods=["GET"])
def home():
    percent=percentageMethod()
    total_month=totalMonth()
    file1=pd.read_json('total_month.json',orient='index')
    month_index=np.array(file1['month_year'])
    month_data=np.array(file1['total'])
    with open('percent.json') as f:
        file2 = json.load(f)
    labels=file2['index']
    data=file2['data']
    if "username" in session:
        return render_template('index.html', last_year=lastYear(), last_month=lastMonth(),dataset=data, label=labels, percent=percent,
        month_index=month_index, month_data=month_data)
    else:
        return render_template('login.html')
# Register new user
@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    elif request.method == "POST":
        registerUser()
        return redirect(url_for("login"))

#Check if email already exists in the registratiion page
@app.route('/checkusername', methods=["POST"])
def check():
    return checkusername()

# Everything Login (Routes to renderpage, check if username exist and also verifypassword through Jquery AJAX request)
@app.route('/login', methods=["GET"])
def login():
    if request.method == "GET":
        if "username" not in session:
            return render_template("login.html")
        else:
            return redirect(url_for("home"))


@app.route('/checkloginusername', methods=["POST"])
def checkUserlogin():
    return checkloginusername()

@app.route('/checkloginpassword', methods=["POST"])
def checkUserpassword():
    return checkloginpassword()

#The admin logout
@app.route('/logout', methods=["GET"])  # URL for logout
def logout():  # logout function
    session.pop('username', None)  # remove user session
    return redirect(url_for("home"))  # redirect to home page with message

#Forgot Password
@app.route('/forgot-password', methods=["GET"])
def forgotpassword():
    return render_template('forgot-password.html')

#404 Page
@app.route('/404', methods=["GET"])
def errorpage():
    return render_template("404.html")

#Blank Page
@app.route('/blank', methods=["GET"])
def blank():
    return render_template('blank.html')

@app.route('/totalyear', methods=["GET"])
def total_year():
    total_year=totalYear()
    file1=pd.read_json('total_year.json',orient='index')
    year_index=np.array(file1['year'])
    year_data=np.array(file1['total']) 
    return render_template("total_year.html",year_index=year_index, year_data=year_data)

@app.route('/totalmonth', methods=["GET"])
def total_month():
    total_month=totalMonth()
    file1=pd.read_json('total_month.json',orient='index')
    month_index=np.array(file1['month_year'])
    month_data=np.array(file1['total'])
    num=6
    # Fit model
    model=fit_model()
    data.model_month=model
    predict_rs, fitted_data=predict(model,6)
    pred_index=np.array(predict_rs['month_year'])
    pred_data=np.array(predict_rs['total'])
    #Test model
    test_rs= test(pred_data[0], fitted_data)
    return render_template("total_month.html",month_index=month_index, month_data=month_data, stationary=check_stationary(), model=model, pred_index=pred_index, pred_data=pred_data, test_rs=test_rs, num=num)
def check_stationary():
    total_month=totalMonth()
    data1=total_month[['month_year','total']]
    data1.set_index('month_year', inplace=True)
    result=stationary(data1)
    return result
def fit_model():
    total_month=totalMonth()
    data1=total_month[['month_year','total']]
    data1.set_index('month_year', inplace=True)
    data=data1['total']
    stationary=check_stationary()
    p=stationary[1]
    if (p<0.05):
        result1 = fit_model_stationary(data)
    else:
        result1 = fit_model_non_stationary(data)
    return result1
def predict(model,num_predict):
    if num_predict==0:
        num_predict=6
    fitted_month, confint_month = model.predict(n_periods=num_predict, return_conf_int=True)
    df2=df[['total', 'date']]
    total_day = df2.groupby(['date'], as_index=False).sum().sort_values('date', ascending=True)
    data=total_day[['date','total']]
    data.set_index('date', inplace=True)
    date = pd.date_range(data.index[-1], periods=num_predict, freq='MS')
    fitted_seri_month = pd.Series(fitted_month, index=date)
    dff=pd.DataFrame(fitted_seri_month)
    dff=dff.reset_index()
    dff.columns=['date','total']
    dff['month_year'] = pd.to_datetime(dff['date']).dt.to_period('M')
    pred=dff[['month_year','total']]
    return pred, fitted_month
def test(y, yhat):
    e = y-yhat
    mse=np.mean(e**2)
    rmse=np.sqrt(mse)
    mae=np.mean(np.abs(e))
    mape=np.mean(abs(e/y))

    # print('Sai số bình phương trung bình MSE: {}'.format(mse))
    # print('Root Mean Square Error: {}'.format(rmse))
    # print('Mean Absolute Error: {}'.format(mae))
    # print('Mean Absolute Percentage Error: {}'.format(mape))
    return mse, rmse, mae, mape
@app.route('/totalmonth', methods=["POST"])
def total_month_num():
    total_month=totalMonth()
    file1=pd.read_json('total_month.json',orient='index')
    month_index=np.array(file1['month_year'])
    month_data=np.array(file1['total'])
    #Get data
    if request.method == "POST":
        num = int(request.form.get("num_month"))
    predict_rs, fitted_data=predict(data.model_month,num)
    pred_index=np.array(predict_rs['month_year'])
    pred_data=np.array(predict_rs['total'])
    #Test model
    test_rs= test(pred_data[0], fitted_data)
    return render_template("total_month.html",month_index=month_index, month_data=month_data, stationary=check_stationary(), model=data.model_month, pred_index=pred_index, pred_data=pred_data, test_rs=test_rs, num=num)
def check_stationary():
    total_month=totalMonth()
    data1=total_month[['month_year','total']]
    data1.set_index('month_year', inplace=True)
    result=stationary(data1)
    return result
def predict(model,num_predict):
    if num_predict==0:
        num_predict=6
    fitted_month, confint_month = model.predict(n_periods=num_predict, return_conf_int=True)
    df2=df[['total', 'date']]
    total_day = df2.groupby(['date'], as_index=False).sum().sort_values('date', ascending=True)
    data2=total_day[['date','total']]
    data2.set_index('date', inplace=True)
    date = pd.date_range(data2.index[-1], periods=num_predict, freq='MS')
    fitted_seri_month = pd.Series(fitted_month, index=date)
    dff=pd.DataFrame(fitted_seri_month)
    dff=dff.reset_index()
    dff.columns=['date','total']
    dff['month_year'] = pd.to_datetime(dff['date']).dt.to_period('M')
    pred=dff[['month_year','total']]
    return pred, fitted_month
def test(y, yhat):
    e = y-yhat
    mse=np.mean(e**2)
    rmse=np.sqrt(mse)
    mae=np.mean(np.abs(e))
    mape=np.mean(abs(e/y))

    # print('Sai số bình phương trung bình MSE: {}'.format(mse))
    # print('Root Mean Square Error: {}'.format(rmse))
    # print('Mean Absolute Error: {}'.format(mae))
    # print('Mean Absolute Percentage Error: {}'.format(mape))
    return mse, rmse, mae, mape
@app.route('/totaldate', methods=["GET"])
def total_date():
    total_date=totalDate()
    date_index=np.array(total_date['date'])
    date_data=np.array(total_date['total'])
    num=30
    # Fit model
    model_date=fit_model_date()
    data.model=model_date
    predict_rs_date, fitted_data_date=predict_date(model_date,30)
    pred_index_date=np.array(predict_rs_date['date'])
    pred_data_date=np.array(predict_rs_date['total'])
    #Test model
    test_rs= test_date(pred_data_date[0], fitted_data_date)
    return render_template("total_date.html",date_index=date_index, date_data=date_data, stationary=check_stationary_date(), model_date=model_date, pred_index=pred_index_date, pred_data=pred_data_date, test_rs=test_rs, num=num)
def check_stationary_date():
    total_date=totalDate()
    data1=total_date[['date','total']]
    data1.set_index('date', inplace=True)
    result=stationary_trend(data1)
    return result
def fit_model_date():
    total_date=totalDate()
    data1=total_date[['date','total']]
    data1.set_index('date', inplace=True)
    data=data1['total']
    result1 = fit_model_fast(data)
    return result1
def predict_date(model_date, num_predict):
    if num_predict==0:
        num_predict=30
    fitted_date, confint_date = model_date.predict(n_periods=num_predict, return_conf_int=True)
    df2=df[['total', 'date']]
    total_day = df2.groupby(['date'], as_index=False).sum().sort_values('date', ascending=True)
    data=total_day[['date','total']]
    data.set_index('date', inplace=True)
    date = pd.date_range(data.index[-1], periods=num_predict)
    fitted_seri_date = pd.Series(fitted_date, index=date)
    dff=pd.DataFrame(fitted_seri_date)
    dff=dff.reset_index()
    dff.columns=['date','total']
    dff['date'] = pd.to_datetime(dff['date']).dt.to_period('D')
    pred=dff[['date','total']]
    return pred, fitted_date
def test_date(y, yhat):
    e = y-yhat
    mse=np.mean(e**2)
    rmse=np.sqrt(mse)
    mae=np.mean(np.abs(e))
    mape=np.mean(abs(e/y))

    # print('Sai số bình phương trung bình MSE: {}'.format(mse))
    # print('Root Mean Square Error: {}'.format(rmse))
    # print('Mean Absolute Error: {}'.format(mae))
    # print('Mean Absolute Percentage Error: {}'.format(mape))
    return mse, rmse, mae, mape

@app.route('/totaldate', methods=["POST"])
def total_date_num():
    total_date=totalDate()
    date_index=np.array(total_date['date'])
    date_data=np.array(total_date['total'])
    #Get data
    if request.method == "POST":
        num = int(request.form.get("num_date"))
    predict_rs_date, fitted_data_date=predict_date(data.model,num)
    pred_index_date=np.array(predict_rs_date['date'])
    pred_data_date=np.array(predict_rs_date['total'])
    test_rs= test_date(pred_data_date[0], fitted_data_date)
    return render_template("total_date.html",date_index=date_index, date_data=date_data, stationary=check_stationary_date(), model_date=data.model, pred_index=pred_index_date, pred_data=pred_data_date, test_rs=test_rs, num=num)
def check_stationary_date():
    total_date=totalDate()
    data1=total_date[['date','total']]
    data1.set_index('date', inplace=True)
    result=stationary_trend(data1)
    return result
def predict_date(model_date, num_predict):
    if num_predict==0:
        num_predict=6
    fitted_date, confint_date = model_date.predict(n_periods=num_predict, return_conf_int=True)
    df2=df[['total', 'date']]
    total_day = df2.groupby(['date'], as_index=False).sum().sort_values('date', ascending=True)
    data2=total_day[['date','total']]
    data2.set_index('date', inplace=True)
    date = pd.date_range(data2.index[-1], periods=num_predict)
    fitted_seri_date = pd.Series(fitted_date, index=date)
    dff=pd.DataFrame(fitted_seri_date)
    dff=dff.reset_index()
    dff.columns=['date','total']
    dff['date'] = pd.to_datetime(dff['date']).dt.to_period('D')
    pred=dff[['date','total']]
    return pred, fitted_date
def test_date(y, yhat):
    e = y-yhat
    mse=np.mean(e**2)
    rmse=np.sqrt(mse)
    mae=np.mean(np.abs(e))
    mape=np.mean(abs(e/y))
    return mse, rmse, mae, mape
@app.route('/revenueyear', methods=["GET"])
def revenue_year():
    sale_year=saleYear()
    year_index=np.array(sale_year['year'])
    year_data=np.array(sale_year['quantity']) 
    return render_template("revenue_year.html",year_index=year_index, year_data=year_data)

@app.route('/revenuemonth', methods=["GET"])
def revenue_month():
    total_month=saleMonth()
    file1=pd.read_json('sale_month.json',orient='index')
    month_index=np.array(file1['month_year'])
    month_data=np.array(file1['quantity'])
    num_sale=6
    # Fit model
    model=fit_model()
    data.model_month=model
    predict_rs, fitted_data=predict(model,6)
    pred_index=np.array(predict_rs['month_year'])
    pred_data=np.array(predict_rs['quantity'])
    #Test model
    test_rs= test(pred_data[0], fitted_data)
    return render_template("revenue_month.html",month_index=month_index, month_data=month_data, stationary=check_stationary(), model=model, pred_index=pred_index, pred_data=pred_data, test_rs=test_rs, num_sale=num_sale)
def check_stationary():
    total_month=saleMonth()
    data1=total_month[['month_year','quantity']]
    data1.set_index('month_year', inplace=True)
    result=stationary(data1)
    return result
def fit_model():
    total_month=saleMonth()
    data1=total_month[['month_year','quantity']]
    data1.set_index('month_year', inplace=True)
    data=data1['quantity']
    stationary=check_stationary()
    p=stationary[1]
    if (p<0.05):
        result1 = fit_model_stationary(data)
    else:
        result1 = fit_model_non_stationary(data)
    return result1
def predict(model,num_predict):
    if num_predict==0:
        num_predict=6
    fitted_month, confint_month = model.predict(n_periods=num_predict, return_conf_int=True)
    df2=df[['quantity', 'date']]
    total_day = df2.groupby(['date'], as_index=False).sum().sort_values('date', ascending=True)
    data=total_day[['date','quantity']]
    data.set_index('date', inplace=True)
    date = pd.date_range(data.index[-1], periods=num_predict, freq='MS')
    fitted_seri_month = pd.Series(fitted_month, index=date)
    dff=pd.DataFrame(fitted_seri_month)
    dff=dff.reset_index()
    dff.columns=['date','quantity']
    dff['month_year'] = pd.to_datetime(dff['date']).dt.to_period('M')
    pred=dff[['month_year','quantity']]
    return pred, fitted_month
def test(y, yhat):
    e = y-yhat
    mse=np.mean(e**2)
    rmse=np.sqrt(mse)
    mae=np.mean(np.abs(e))
    mape=np.mean(abs(e/y))

    # print('Sai số bình phương trung bình MSE: {}'.format(mse))
    # print('Root Mean Square Error: {}'.format(rmse))
    # print('Mean Absolute Error: {}'.format(mae))
    # print('Mean Absolute Percentage Error: {}'.format(mape))
    return mse, rmse, mae, mape
@app.route('/revenuemonth', methods=["POST"])
def revenue_month_num():
    total_month=saleMonth()
    file1=pd.read_json('sale_month.json',orient='index')
    month_index=np.array(file1['month_year'])
    month_data=np.array(file1['quantity'])
    #Get data
    if request.method == "POST":
        num_sale= int(request.form.get("sale_month"))
    predict_rs, fitted_data=predict(data.model_month,num_sale)
    pred_index=np.array(predict_rs['month_year'])
    pred_data=np.array(predict_rs['quantity'])
    #Test model
    test_rs= test(pred_data[0], fitted_data)
    return render_template("revenue_month.html",month_index=month_index, month_data=month_data, stationary=check_stationary(), model=data.model_month, pred_index=pred_index, pred_data=pred_data, test_rs=test_rs, num_sale=num_sale)
def check_stationary():
    total_month=saleMonth()
    data1=total_month[['month_year','quantity']]
    data1.set_index('month_year', inplace=True)
    result=stationary(data1)
    return result
def predict(model,num_predict):
    if num_predict==0:
        num_predict=6
    fitted_month, confint_month = model.predict(n_periods=num_predict, return_conf_int=True)
    df2=df[['quantity', 'date']]
    total_day = df2.groupby(['date'], as_index=False).sum().sort_values('date', ascending=True)
    data2=total_day[['date','quantity']]
    data2.set_index('date', inplace=True)
    date = pd.date_range(data2.index[-1], periods=num_predict, freq='MS')
    fitted_seri_month = pd.Series(fitted_month, index=date)
    dff=pd.DataFrame(fitted_seri_month)
    dff=dff.reset_index()
    dff.columns=['date','quantity']
    dff['month_year'] = pd.to_datetime(dff['date']).dt.to_period('M')
    pred=dff[['month_year','quantity']]
    return pred, fitted_month
def test(y, yhat):
    e = y-yhat
    mse=np.mean(e**2)
    rmse=np.sqrt(mse)
    mae=np.mean(np.abs(e))
    mape=np.mean(abs(e/y))

    # print('Sai số bình phương trung bình MSE: {}'.format(mse))
    # print('Root Mean Square Error: {}'.format(rmse))
    # print('Mean Absolute Error: {}'.format(mae))
    # print('Mean Absolute Percentage Error: {}'.format(mape))
    return mse, rmse, mae, mape
@app.route('/revenuedate', methods=["GET"])
def revenue_date():
    total_date=saleDate()
    date_index=np.array(total_date['date'])
    date_data=np.array(total_date['quantity'])
    num=30
    # Fit model
    model_date=fit_model_date()
    data.sale_model=model_date
    predict_rs_date, fitted_data_date=predict_date(model_date,30)
    pred_index_date=np.array(predict_rs_date['date'])
    pred_data_date=np.array(predict_rs_date['quantity'])
    #Test model
    test_rs= test_date(pred_data_date[0], fitted_data_date)
    return render_template("revenue_date.html",date_index=date_index, date_data=date_data, stationary=check_stationary_date(), model_date=model_date, pred_index=pred_index_date, pred_data=pred_data_date, test_rs=test_rs, num=num)
def check_stationary_date():
    total_date=saleDate()
    data1=total_date[['date','quantity']]
    data1.set_index('date', inplace=True)
    result=stationary_trend(data1)
    return result
def fit_model_date():
    total_date=saleDate()
    data1=total_date[['date','quantity']]
    data1.set_index('date', inplace=True)
    data=data1['quantity']
    result1 = fit_model_fast(data)
    return result1
def predict_date(model_date, num_predict):
    if num_predict==0:
        num_predict=30
    fitted_date, confint_date = model_date.predict(n_periods=num_predict, return_conf_int=True)
    df2=df[['quantity', 'date']]
    total_day = df2.groupby(['date'], as_index=False).sum().sort_values('date', ascending=True)
    data=total_day[['date','quantity']]
    data.set_index('date', inplace=True)
    date = pd.date_range(data.index[-1], periods=num_predict)
    fitted_seri_date = pd.Series(fitted_date, index=date)
    dff=pd.DataFrame(fitted_seri_date)
    dff=dff.reset_index()
    dff.columns=['date','quantity']
    dff['date'] = pd.to_datetime(dff['date']).dt.to_period('D')
    pred=dff[['date','quantity']]
    return pred, fitted_date
def test_date(y, yhat):
    e = y-yhat
    mse=np.mean(e**2)
    rmse=np.sqrt(mse)
    mae=np.mean(np.abs(e))
    mape=np.mean(abs(e/y))

    # print('Sai số bình phương trung bình MSE: {}'.format(mse))
    # print('Root Mean Square Error: {}'.format(rmse))
    # print('Mean Absolute Error: {}'.format(mae))
    # print('Mean Absolute Percentage Error: {}'.format(mape))
    return mse, rmse, mae, mape

@app.route('/revenuedate', methods=["POST"])
def revenue_date_num():
    total_date=saleDate()
    date_index=np.array(total_date['date'])
    date_data=np.array(total_date['quantity'])
    #Get data
    if request.method == "POST":
        num = int(request.form.get("sale_date"))
    predict_rs_date, fitted_data_date=predict_date(data.sale_model,num)
    pred_index_date=np.array(predict_rs_date['date'])
    pred_data_date=np.array(predict_rs_date['quantity'])
    test_rs= test_date(pred_data_date[0], fitted_data_date)
    return render_template("revenue_date.html",date_index=date_index, date_data=date_data, stationary=check_stationary_date(), model_date=data.sale_model, pred_index=pred_index_date, pred_data=pred_data_date, test_rs=test_rs, num=num)
def check_stationary_date():
    total_date=saleDate()
    data1=total_date[['date','quantity']]
    data1.set_index('date', inplace=True)
    result=stationary_trend(data1)
    return result
def predict_date(model_date, num_predict):
    if num_predict==0:
        num_predict=6
    fitted_date, confint_date = model_date.predict(n_periods=num_predict, return_conf_int=True)
    df2=df[['quantity', 'date']]
    total_day = df2.groupby(['date'], as_index=False).sum().sort_values('date', ascending=True)
    data2=total_day[['date','quantity']]
    data2.set_index('date', inplace=True)
    date = pd.date_range(data2.index[-1], periods=num_predict)
    fitted_seri_date = pd.Series(fitted_date, index=date)
    dff=pd.DataFrame(fitted_seri_date)
    dff=dff.reset_index()
    dff.columns=['date','quantity']
    dff['date'] = pd.to_datetime(dff['date']).dt.to_period('D')
    pred=dff[['date','quantity']]
    return pred, fitted_date
def test_date(y, yhat):
    e = y-yhat
    mse=np.mean(e**2)
    rmse=np.sqrt(mse)
    mae=np.mean(np.abs(e))
    mape=np.mean(abs(e/y))
    return mse, rmse, mae, mape

#Tables Page
@app.route('/tables', methods=["GET"])
def tables():
    return render_template("tables.html")

#Utilities-animation
@app.route('/utilities-animation', methods=["GET"])
def utilitiesanimation():
    return render_template("utilities-animation.html")

#Utilities-border
@app.route('/utilities-border', methods=["GET"])
def utilitiesborder():
    return render_template("utilities-border.html")

#Utilities-color
@app.route('/utilities-color', methods=["GET"])
def utilitiescolor():
    return render_template("utilities-color.html")

#utilities-other
@app.route('/utilities-other', methods=["GET"])
def utilitiesother():
    return render_template("utilities-other.html")
