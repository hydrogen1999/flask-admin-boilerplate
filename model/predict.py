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
import pmdarima.auto_arima