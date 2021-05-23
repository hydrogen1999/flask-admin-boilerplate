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

df = pd.read_csv('data\order.csv')
df['date'] = pd.to_datetime(df.date)
df['month_year'] = pd.to_datetime(df['date']).dt.to_period('M')


def totalYear():
    df2 = df[['total', 'year']]
    total_year = df2.groupby(['year'], as_index=False).sum(
    ).sort_values('year', ascending=True)
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
    return total_date

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
    return sale_month

def saleDate():
    df8 = df[['quantity', 'date']]
    sale_date = df8.groupby(['date'], as_index=False).sum(
    ).sort_values('date', ascending=True)
    return sale_date

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

