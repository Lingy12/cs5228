from sklearn.neural_network import MLPRegressor
import pandas as pd
import plotly
from utils import train_kfold
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

import pandas as pd
from data_processing import clean_data, process_data

DATA_DIR = './data'
train = os.path.join(DATA_DIR, 'train.csv')
test = os.path.join(DATA_DIR, 'test.csv')

train_final = os.path.join(DATA_DIR, 'train_with_mrt_mall_school.csv')

train_final_df = pd.read_csv(train_final)
train_df, test_df = pd.read_csv(train), pd.read_csv(test)

train_df_cleaned = clean_data(train_df)
train_df_cleaned = process_data(train_df_cleaned)
train_df_cleaned_final = clean_data(train_final_df)
train_df_cleaned_final = process_data(train_df_cleaned_final)

train_kfold(['floor_area_sqm',
                         'age',
                         'town_psqm',
                         'regional_psqm',  
                         'nearest_mrt_dist',
                         'near_mrt_count',
                         'nearest_mall_dist',
                         'near_mall_count',
                         'near_school_count', 
                         'nearest_school_dist',
                         'date'
                         ], 
                           ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town'],
                        train_df_cleaned_final, 'monthly_rent', 10, 
                        pipelines=[StandardScaler(), 
                                   MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes=(200,), 
                                                learning_rate='adaptive', early_stopping=True)])