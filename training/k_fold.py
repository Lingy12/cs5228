import sys, os
sys.path.append('..')
sys.path.append(os.getcwd())
from sklearn.neural_network import MLPRegressor
import pandas as pd
import plotly
from utils.utils import train_kfold
from sklearn.decomposition import PCA
from sklearn import svm
import os
from config import sk_config
import fire
import pandas as pd
from preprocessing.data_processing import clean_data, process_data

def run(conf_name):
  print('running with ' + conf_name)
  conf = getattr(sk_config, conf_name)
  print(conf)
  DATA_DIR = './data'

  train_final = os.path.join(DATA_DIR, 'train_final.csv')

  train_final_df = pd.read_csv(train_final)

  train_df_cleaned_final = clean_data(train_final_df)
  train_df_cleaned_final = process_data(train_df_cleaned_final)
  if 'feature_norm' not in conf:
    feature_norm = ''
  else:
    feature_norm = conf['feature_norm']

  train_kfold(conf['features'], conf['cat_features'],
                          train_df_cleaned_final, conf['target'], 10, 
                          pipelines=conf['pipelines'], feature_norm=feature_norm)

if __name__ == '__main__':
  fire.Fire(run)
  