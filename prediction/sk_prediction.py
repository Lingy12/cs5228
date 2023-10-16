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
from config import sk_config
import fire

def run(conf_name):
    print('running with configuration ' + conf_name)
    conf = getattr(sk_config, conf_name)
    print(conf)
    DATA_DIR = './data'

    train_final = os.path.join(DATA_DIR, 'train_final.csv')
    test_final = os.path.join(DATA_DIR, 'test_final.csv')
    train_final_df, test_final_df = pd.read_csv(train_final), pd.read_csv(test_final)
    target = conf['target']
    # train_df

    features = conf['features']
    norm_features = conf['norm_features']
    ord_features = conf['ord_features']
    out_df = generate_prediction(features, ord_features, norm_features, train_final_df, test_final_df, 
                                target, pipelines=conf['pipelines'])
    output_prediction(out_df)
    
if __name__ == "__main__":
    fire.Fire(run)