from operator import xor
import sys

import torch
sys.path.append('..')

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import numpy as np
from preprocessing.data_processing import combined_encoding, generate_features, clean_data, process_data
from training.torch_models import BaseMLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim import Adam
from torchsummary import summary

class MyDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.x = X
        self.y = y
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return [self.x[index], self.y[index]]

def create_dataloader(X, y, batch_size):
    ds = MyDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size)
    return loader

def evaluate(model, X_test, y_test, batch_size, device):
    # print(batch_size)
    model.eval()
    test_loader = create_dataloader(X_test, y_test, batch_size)
    pred = []

    for batch, data in enumerate(test_loader):
        X, y = data[0].to(device).float(), data[1].to(device).float()
        pred.append(model(X))
    predictions = torch.cat(pred)
    return predictions

def train(model, X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size, device, criterion, verbose=1):
    model.train()
    model.to(device)
    # print(model.parameters())
    optimizer = Adam(lr=learning_rate, params=model.parameters())
    train_loader = create_dataloader(X_train, y_train, batch_size)

    for epoch in tqdm(range(epochs)):
        for _, data in enumerate(train_loader):
            X, y = data[0].to(device).float(), data[1].to(device).float()
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs.squeeze(), y)
            loss.backward()
            optimizer.step()
        train_pred = evaluate(model, X_train, y_train, batch_size, device)
        predictions = evaluate(model, X_val, y_val, batch_size, device)
        if verbose == 0:
            print('[Epoch {}] train_mae = {}, val_mae = {}'.format(epoch, mean_absolute_error(train_pred.detach().cpu().numpy(), y_train), 
                                                               mean_absolute_error(predictions.detach().cpu().numpy(), y_val)))
    return model


def train_kfold(features, cat_features, df, target, k, model_class, model_params, epoches, feature_norm, device, batch_size, lr, verbose=1):
  # if no normalize, then do not need to split features and cat_features
  df = df.copy()
  # print(df.head())
  kf = KFold(n_splits=k)
  # model = make_pipeline(*pipelines)
  # model = BaseMLPRegressor()
  
#   print(model)
  scaler = StandardScaler()
  cat_one_hot_cols = []
  for cat_feat in cat_features:
    for value in df[cat_feat].unique():
      cat_one_hot_cols.append(cat_feat + '_' + value)
  
  
  if len(feature_norm) != 0:
    features = list(map(lambda x: x + '_' + feature_norm, features))
    
  features = features + cat_one_hot_cols
  model = model_class(input_size=len(features), **model_params)
  summary(model.to(device), (1, len(features)) )
  # model = BaseMLPRegressor(input_size=len(features), output_size=1, hidden_layers=1, hidden_unit=[100], dropout=0.0)
  # print(df.columns)
  metrics = {
    "train_mae": [],
    "val_mape": [],
    "val_pcc": [],
    "val_mae": []
  }
  
  for i, (train_index, val_index) in enumerate(kf.split(df)):
    model = model_class(input_size=len(features), **model_params)
    train_df, val_df = df.iloc[train_index].reset_index(drop=True), df.iloc[val_index].reset_index(drop=True)
    train_df, val_df = generate_features(train_df, val_df)
    
    if len(cat_features) != 0:
        joined_df = combined_encoding(train_df, val_df, cat_features=cat_features, is_val=True)
        train_df, val_df = joined_df[joined_df['split'] == 'train'].reset_index(drop=True), joined_df[joined_df['split'] == 'test'].reset_index(drop=True)
    # print(val_df.columns)
    X_train, X_test, y_train, y_test = train_df[features].to_numpy(), val_df[features].to_numpy(), train_df[target].to_numpy(), val_df[target].to_numpy()
    X_train = scaler.fit_transform(X_train) 
    X_test = scaler.transform(X_test) # standardize
    print(X_train.shape, y_train.shape)
    # model = make_pipeline(*pipelines)
    model = train(model, X_train, y_train, X_test, y_test, epoches, lr, batch_size, device, torch.nn.MSELoss(), verbose=verbose)
    y_pred = evaluate(model, X_test, y_test, batch_size, device)
    train_pred = evaluate(model, X_train, y_train, batch_size, device)
    train_mae = mean_absolute_error(y_train, train_pred.detach().cpu())
    val_mae, val_pcc = mean_absolute_error(y_test, y_pred.detach().cpu()), np.corrcoef(y_pred.squeeze().detach().cpu(), y_test)[0][1]
    val_mape = mean_absolute_percentage_error(y_test, y_pred.detach().cpu())
    
    for metric in metrics:
      metrics[metric].append(eval(metric))
    print(f'fold: {i}: train_mse = {train_mae}, mae = {val_mae}, pcc = {val_pcc}, mape = {val_mape}')
  output = "Overall evaluation: " 
  print('Overall evaluation')
  for metric in metrics:
    arr = np.array(metrics[metric])
    output += f'{metric}: average = {arr.mean()}, std_dev = {arr.std()}\n'
    print(f'{metric}: average = {arr.mean()}, std_dev = {arr.std()}')
  return output

def generate_prediction(features, cat_features, train_df, test_df, target, model_class, model_params, epoches, feature_norm, device, batch_size, lr, verbose=1):
  # if no normalize, then do not need to split features and cat_features
  train_df_cleaned = clean_data(train_df)
  test_df_cleaned = clean_data(test_df)
  processed_train = process_data(train_df_cleaned)
  processed_test = process_data(test_df_cleaned, mode='test')
  # model = make_pipeline(*pipelines)
  # print(model)
#   print(model)
  scaler = StandardScaler()
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
  model = model_class(input_size=len(features), **model_params)
  summary(model.to(device), (1, len(features)))
 
  X = train_df[features].to_numpy()
  y = train_df[target].to_numpy()
  X = scaler.fit_transform(X)
  model = train(model, X, y, X, y, epoches, lr, batch_size, device, torch.nn.MSELoss(),verbose=verbose)

  
  X_test = test_df[features].to_numpy()
  X_test = scaler.transform(X_test)
  y_test = np.zeros(len(X_test))
  y = evaluate(model, X_test, y_test, batch_size, device).detach().cpu().numpy()
  out_df = pd.DataFrame(data=[(id, y[id][0]) for id in range(y.shape[0])], columns=['id', 'predicted']).set_index('id')
  return out_df
