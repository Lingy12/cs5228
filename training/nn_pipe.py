import sys, os
sys.path.append('..')
sys.path.append(os.getcwd())
from preprocessing.data_processing import clean_data, process_data
import pandas as pd
from torch import cat, nn
from training import torch_models
import torch
from utils.nn_utils import train_kfold, generate_prediction
from utils.utils import output_prediction
from config import sk_config, model_conf
import fire

def run_nn_pipeline(k_fold_val, epoches, conf_name, model_conf_name, model_class_name, lr, batch_size, verbose=1):
    print('running conf ' + conf_name)
    conf = getattr(sk_config, conf_name)
    print(conf)
    train_final = './data/train_final.csv'
    test_final = './data/test_final.csv'
    train_final_df = pd.read_csv(train_final)
    test_final_df = pd.read_csv(test_final)
    train_df_cleaned_final = clean_data(train_final_df)
    train_df_cleaned_final = process_data(train_df_cleaned_final)
    test_df_cleaned_final = clean_data(test_final_df)
    test_df_cleaned_final = process_data(test_df_cleaned_final, mode='test')
    print(train_df_cleaned_final.columns)
    print('cuda available = ' + str(torch.cuda.is_available()))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model_class = getattr(torch_models, model_class_name)
    model_config = getattr(model_conf, model_conf_name)
    print(model_config)


    features = conf['features']
    norm_features = conf['norm_features']
    ord_features = conf['ord_features']
    pipelines = conf['pipelines']

    target = 'monthly_rent'
    if k_fold_val:
        print('running k-fold')
        res = train_kfold(features, ord_features, norm_features, pipelines, train_df_cleaned_final, target, 10, 
                model_class, model_config, epoches=epoches, feature_norm='', 
                device=device, lr=lr, batch_size=batch_size)
        print(res)
    out_df = generate_prediction(features, ord_features, norm_features, pipelines, train_final_df, test_final_df, target,
                model_class, model_config, epoches=epoches, feature_norm='', 
                device=device, lr=lr, batch_size=batch_size, verbose=verbose)
    # print(out_df)
    # out_df.to_csv(output_name)
    output_prediction(out_df)
    # print(res)

if __name__ == "__main__":
    fire.Fire(run_nn_pipeline)
