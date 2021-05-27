import pandas as pd
import numpy as np
import re
import calendar
from datetime import datetime
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
import statsmodels.api as sm
import json
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima

df = pd.read_csv('data\order.csv')
df['date'] = pd.to_datetime(df.date)
df['month_year'] = pd.to_datetime(df['date']).dt.to_period('M')


def totalYear():
    df2 = df[['total', 'year']]
    total_year = df2.groupby(['year'], as_index=False).sum(
    ).sort_values('year', ascending=True)
    data=total_year[['year','total']]
    data['year']=data['year'].astype(str)
    data.to_json('total_year.json', orient='index')
    return total_year


def lastYear():
    """lastYear() function: return the total revenue of the nearest year

    Returns:
        [type]: [description]
    """
    lastYear = totalYear().tail(1)
    last_year = lastYear.iloc[0]['total'].round(2)
    return last_year


def totalMonth():
    """This is a function to calculate the total revenue for each month.
    + Using 'pandas' library

    Returns:
        [dataframe]: A list of total revenue for each month
    """
    df4 = df[['total', 'month_year']]
    total_month = df4.groupby(['month_year'], as_index=False).sum(
    ).sort_values('month_year', ascending=True)
    data1=total_month[['month_year','total']]
    data1['month_year']=data1['month_year'].astype(str)
    data1.to_json('total_month.json', orient='index')
    return total_month


def lastMonth():
    """lastMonth() function: return the total revenue of the nearest month

    Returns:
        [type]: [description]
    """
    lastMonth = totalMonth().tail(1)
    last_month = lastMonth.iloc[0]['total'].round(2)
    return last_month


def totalDate():
    """[summary]
        This is a function to calculate the total revenue of each day
    Returns:
        [type]: [description]
    """   
    df5 = df[['total', 'date']]
    total_date = df5.groupby(['date'], as_index=False).sum(
    ).sort_values('date', ascending=True)
    data1=total_date[['date','total']]
    data1['date'] = pd.to_datetime(data1['date']).dt.date
    return data1

def lastDate():
    """[summary]
        lastDate() function: return the total revenue of the nearest day
    Returns:
        [type]: [description]
    """    
    lastDate = totalDate().tail(1)
    last_date = lastDate.iloc[0]['total'].round(2)
    return last_date

def saleYear():
    df6 = df[['quantity', 'year']]
    sale_year = df6.groupby(['year'], as_index=False).sum(
    ).sort_values('year', ascending=True)
    return sale_year

def saleMonth():
    df7 = df[['quantity', 'month_year']]
    sale_month = df7.groupby(['month_year'], as_index=False).sum(
    ).sort_values('month_year', ascending=True)
    data1=sale_month[['month_year','quantity']]
    data1['month_year']=data1['month_year'].astype(str)
    data1.to_json('sale_month.json', orient='index')
    return sale_month

def saleDate():
    df8 = df[['quantity', 'date']]
    sale_date = df8.groupby(['date'], as_index=False).sum(
    ).sort_values('date', ascending=True)
    data1=sale_date[['date','quantity']]
    data1['date'] = pd.to_datetime(data1['date']).dt.date
    return data1

def lastYearSale():
    lastYearSale = saleYear().tail(1)
    last_year_sale = lastYearSale.iloc[0]['quantity']
    return last_year_sale

def lastMonthSale():
    lastMonthSale = saleMonth().tail(1)
    last_month_sale = lastMonthSale.iloc[0]['quantity']
    return last_month_sale

def lastDateSale():
    lastDateSale = saleDate().tail(1)
    last_date_sale = lastDateSale.iloc[0]['quantity']
    return last_date_sale

def percentageMethod():
    df9=df[['method']]
    percent=(df9['method'].value_counts()/df9['method'].count())*100
    percent.to_json('percent.json', orient='split')
    return percent

def stationary_trend(data):
    result = sm.tsa.adfuller(data.dropna(), regression='c')
    # print('ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))
    return result


def stationary(data):
    result = sm.tsa.adfuller(data.dropna(), regression='ct')
    # print('ADF Statistic: %f' % result[0])
    # print('p-value: %f' % result[1])
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))
    return result


def fit_model_stationary(data):
    model = auto_arima(data, start_p=0, start_q=0,
                       max_p=5, max_q=5, m=12,
                       start_P=0, seasonal=True,
                       d=0, D=1, trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)

    print(model.aic())
    return model

def fit_model_non_stationary(data):
    model = auto_arima(data, start_p=0, start_q=0,
                       max_p=5, max_q=5, m=12,
                       start_P=0, seasonal=True,
                       d=1, D=1, trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)

    print(model.aic())
    return model

def fit_model_fast(data):
    model = auto_arima(data, start_p=5, start_q=0,
                           max_p=5, max_q=0, m=12,start_Q=0, max_Q=0,
                           start_P=2,max_P=2, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

    print(model.aic())
    return model

def fit_model_fast_stationary(data):
    model = auto_arima(data, start_p=5, start_q=0,
                           max_p=5, m=12,start_Q=0, max_Q=0,
                           start_P=2, seasonal=True,
                           d=0, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

    print(model.aic())
    return model

# def show_result(model):
#     return model.summary()

# def check_model(model):
#     model_sarima.plot_diagnostics(figsize=(15, 12))
#     url='/static/images/plot.png'
#     plt.savefig(url)
#     return url