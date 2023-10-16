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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn import svm

# def preprocess_data(train_df, test_df, num_processor, features, cat_one_hot_cols):
#     X_train_num, X_test_num = train_df[features].to_numpy(), test_df[features].to_numpy()
#     X_train_cat, X_test_cat = train_df[cat_one_hot_cols].to_numpy(), test_df[cat_one_hot_cols].to_numpy()
#     X_train_num = num_processor.fit_transform(X_train_num)
#     X_test_num = num_processor.transform(X_test_num)
#     X_train_cat = cat_processor.fit_transform(X_train_cat)
#     X_test_cat = cat_processor.transform(X_test_num)
    
#     X_train, X_test = np.concatenate([X_train_num, X_train_cat], axis=1), np.concatenate([X_test_num, X_test_cat], axis=1)
#     return X_train, X_test

def transform_features(train_df, test_df, features, ord_features, norm_features, num_processor, ord_processor, norm_processor):
  train_lst, test_lst = [], []
  X_train_num = num_processor.fit_transform(train_df[features].to_numpy())
  X_test_num = num_processor.transform(test_df[features].to_numpy())
  train_lst.append(X_train_num)
  test_lst.append(X_test_num)
  
  if len(ord_features) > 0:
    X_train_ord = ord_processor.fit_transform(train_df[ord_features].to_numpy())
    X_test_ord = ord_processor.transform(test_df[ord_features].to_numpy())
    train_lst.append(X_train_ord)
    test_lst.append(X_test_ord)
    
  if len(norm_features) > 0:
    X_train_norm = norm_processor.fit_transform(train_df[norm_features].to_numpy())
    X_test_norm = norm_processor.transform(test_df[norm_features].to_numpy())
    train_lst.append(X_train_norm)
    test_lst.append(X_test_norm)
  np.concatenate(test_lst, axis=1)
  X_train, X_test = np.concatenate(train_lst, axis=1), np.concatenate(test_lst, axis=1)
  return X_train, X_test
  
  
def train_kfold(features, ord_features, norm_features, df, target, k, feature_norm='', pipelines={'num_processor': [StandardScaler()], 
                                                                                   'ord_cat_processor': [OrdinalEncoder()], 
                                                                                   'norm_cat_processor': [OneHotEncoder(sparse_output=False)],
                                                                                   'model': [LinearRegression()]}):
  # if no normalize, then do not need to split features and cat_features
  df = df.copy()
  # print(df.head())
  kf = KFold(n_splits=k)
  model, num_processor, ord_cat_processor, norm_cat_processor = make_pipeline(*pipelines['model']), \
                                          make_pipeline(*pipelines['num_processor']), \
                                          make_pipeline(*pipelines['ord_cat_processor']), make_pipeline(*pipelines['norm_cat_processor'])

  print(model)
  cat_features = ord_features + norm_features
  cat_one_hot_cols = []
  for cat_feat in cat_features:
    for value in df[cat_feat].unique():
      cat_one_hot_cols.append(cat_feat + '_' + value)
  
  
  if len(feature_norm) != 0:
    features = list(map(lambda x: x + '_' + feature_norm, features))
    
  # all_features = features + cat_one_hot_cols
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
    
    X_train, X_test = transform_features(train_df, val_df, features, ord_features, norm_features, num_processor, ord_cat_processor, norm_cat_processor)
    y_train, y_test = train_df[target].to_numpy(), val_df[target].to_numpy()
   
    model = make_pipeline(*pipelines['model'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)
    r2_train = r2_score(y_train, train_pred)
    val_mae, val_pcc = mean_absolute_error(y_test, y_pred), np.corrcoef(y_pred, y_test)[0][1]
    val_mape = mean_absolute_percentage_error(y_test, y_pred)
    
    for metric in metrics:
      metrics[metric].append(eval(metric))
    print(f'fold: {i}: train_mse = {train_mae}, train_r2 = {r2_train}, mae = {val_mae}, pcc = {val_pcc}, mape = {val_mape}')
    
  print('Overall evaluation')
  for metric in metrics:
        arr = np.array(metrics[metric])
        print(f'{metric}: average = {arr.mean()}, std_dev = {arr.std()}')

def generate_prediction(features, ord_features, norm_features, train_df, test_df, target, feature_norm='', pipelines={'num_processor': [StandardScaler()], 
                                                                                   'ord_cat_processor': [OrdinalEncoder()], 
                                                                                   'norm_cat_processor': [OneHotEncoder(sparse_output=False)],
                                                                                   'model': [LinearRegression()]}):
  # if no normalize, then do not need to split features and cat_features
  train_df_cleaned = clean_data(train_df)
  test_df_cleaned = clean_data(test_df)
  processed_train = process_data(train_df_cleaned)
  processed_test = process_data(test_df_cleaned, mode='test')
  model, num_processor, ord_cat_processor, norm_cat_processor = make_pipeline(*pipelines['model']), \
                                          make_pipeline(*pipelines['num_processor']), \
                                          make_pipeline(*pipelines['ord_cat_processor']), make_pipeline(*pipelines['norm_cat_processor'])
  print(model)
#   print(model)
  train_df, test_df = generate_features(processed_train, processed_test)
  print(len(train_df), len(test_df))
  
  cat_features = ord_features + norm_features
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
    
  # features = features + cat_one_hot_cols
  X, X_test = transform_features(train_df, test_df, features, ord_features, norm_features, num_processor, ord_cat_processor, norm_cat_processor)
  # X = train_df[features].to_numpy()
  y = train_df[target].to_numpy()
#   X = scalar.fit_transform(X)
  model.fit(X, y)
  
  # X_test = test_df[features].to_numpy()
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