import os, sys
import sys
sys.path.append('..')
sys.path.append(os.getcwd())
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from preprocessing.data_processing import *
from utils.utils import generate_prediction
from utils.utils import output_prediction
from datetime import datetime


DATA_DIR = './data'
train_data = os.path.join(DATA_DIR, 'train.csv')
test_data = os.path.join(DATA_DIR, 'test.csv')

train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)

train_final = os.path.join(DATA_DIR, 'train_with_mrt_mall_school.csv')
test_final = os.path.join(DATA_DIR, 'test_with_mrt_mall_school.csv')
train_final_df, test_final_df = pd.read_csv(train_final), pd.read_csv(test_final)
target = 'monthly_rent'
# train_df

features = ['floor_area_sqm',
                         'age',
                         'town_psqm',
                         'regional_psqm',
                         'nearest_mrt_dist',
                         'nearest_mall_dist',
                         'near_mall_count',
                         'near_mrt_count',
                         'near_school_count', 
                         'nearest_school_dist',
                         'date']
cat_features = ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town']
out_df = generate_prediction(features, cat_features, train_final_df, test_final_df, target, pipelines=[StandardScaler(), MLPRegressor(random_state=1, max_iter=1000)])
output_prediction(out_df)