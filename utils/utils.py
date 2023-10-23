import sys
sys.path.append('..')
from datetime import datetime
import os
import pandas as pd
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import numpy as np
from preprocessing.data_processing import combined_encoding, generate_features, clean_data, process_data
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn import svm

def train_kfold(features, cat_features, df, target, k, feature_norm='', pipelines=[StandardScaler(), LinearRegression()]):
  # if no normalize, then do not need to split features and cat_features
  df = df.copy()
  # print(df.head())
  kf = KFold(n_splits=k)
  model = make_pipeline(*pipelines)
  print(model)
#   print(model)
#   scaler = StandardScaler()
  cat_one_hot_cols = []
  for cat_feat in cat_features:
    for value in df[cat_feat].unique():
      cat_one_hot_cols.append(cat_feat + '_' + value)
  
  
  if len(feature_norm) != 0:
    features = list(map(lambda x: x + '_' + feature_norm, features))
    
  features = features + cat_one_hot_cols
  # print(df.columns)
  metrics = {
    "r2_train": [],
    "train_mae": [],
    "val_mape": [],
    "val_pcc": [],
    "val_mae": []
  }
  
  for i, (train_index, val_index) in enumerate(kf.split(df)):
    train_df, val_df = df.iloc[train_index].reset_index(drop=True), df.iloc[val_index].reset_index(drop=True)
    train_df, val_df = generate_features(train_df, val_df)
    
    if len(cat_features) != 0:
        joined_df = combined_encoding(train_df, val_df, cat_features=cat_features, is_val=True)
        train_df, val_df = joined_df[joined_df['split'] == 'train'].reset_index(drop=True), joined_df[joined_df['split'] == 'test'].reset_index(drop=True)
    # print(val_df.columns)
    X_train, X_test, y_train, y_test = train_df[features].to_numpy(), val_df[features].to_numpy(), train_df[target].to_numpy(), val_df[target].to_numpy()
   
    model = make_pipeline(*pipelines)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)
    r2_train = r2_score(y_train, train_pred)

    if len(y_pred.shape) == 2:
      y_pred, y_test = np.squeeze(y_pred), np.squeeze(y_test)
    val_mae, val_pcc = mean_absolute_error(y_test, y_pred), np.corrcoef(y_pred, y_test)[0][1]
    val_mape = mean_absolute_percentage_error(y_test, y_pred)
    
    for metric in metrics:
      metrics[metric].append(eval(metric))
    print(f'fold: {i}: train_mse = {train_mae}, train_r2 = {r2_train}, mae = {val_mae}, pcc = {val_pcc}, mape = {val_mape}')
    
  print('Overall evaluation')
  for metric in metrics:
        arr = np.array(metrics[metric])
        print(f'{metric}: average = {arr.mean()}, std_dev = {arr.std()}')

def generate_prediction(features, cat_features, train_df, test_df, target, feature_norm='', pipelines=[StandardScaler(), LinearRegression()]):
  # if no normalize, then do not need to split features and cat_features
  train_df_cleaned = clean_data(train_df)
  test_df_cleaned = clean_data(test_df)
  processed_train = process_data(train_df_cleaned)
  processed_test = process_data(test_df_cleaned, mode='test')
  model = make_pipeline(*pipelines)
  print(model)
#   print(model)
  train_df, test_df = generate_features(processed_train, processed_test)
  print(len(train_df), len(test_df))
  
  if len(cat_features) != 0:
    joined_df = combined_encoding(train_df, test_df, cat_features=cat_features)
    train_df, test_df = joined_df[joined_df['split'] == 'train'].reset_index(drop=True), joined_df[joined_df['split'] == 'test'].reset_index(drop=True)
  # print(df.head())
  
  cat_one_hot_cols = []
  for cat_feat in cat_features:
    for value in joined_df[cat_feat].unique():
      cat_one_hot_cols.append(cat_feat + '_' + value)
  
  
  if len(feature_norm) != 0:
    features = list(map(lambda x: x + '_' + feature_norm, features))
    
  features = features + cat_one_hot_cols
  
  X = train_df[features].to_numpy()
  y = train_df[target].to_numpy()
#   X = scalar.fit_transform(X)
  model.fit(X, y)
  
  X_test = test_df[features].to_numpy()
#   X_test = scalar.transform(X_test)
  y = model.predict(X_test)
  out_df = pd.DataFrame(data=[(id, y[id]) for id in range(y.shape[0])], columns=['id', 'predicted']).set_index('id')
  return out_df

def output_prediction(prediction_df):
  if not os.path.exists('predictions'):
    os.mkdir('predictions')
  name = 'my_pred' + datetime.now().strftime('%Y%m%d%H%M')
  out_path = os.path.join('predictions', name + '.csv')
  prediction_df.to_csv(out_path)