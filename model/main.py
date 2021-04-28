import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from flask import render_template

df=pd.read_csv('data\order.csv')
df['date'] =pd.to_datetime(df.date)
df['month_year'] = pd.to_datetime(df['date']).dt.to_period('M')

def totalYear():
    df2 = df[['total', 'year']]
    total_year = df2.groupby(['year'], as_index=False).sum().sort_values('year', ascending=True)
    return total_year
def lastYear():
    lastYear=totalYear().tail(1)
    last_year=last_Year.iloc[0]['total']
    print(last_year)
    return render_template("index.html",last_year)