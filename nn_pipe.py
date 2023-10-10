from data_processing import clean_data, process_data
import pandas as pd
from torch import cat, nn
from torch_models import BaseMLPRegressor
import torch
from nn_utils import train_kfold, generate_prediction
import fire

def run_nn_pipeline(output_name, k_fold_val, epoches, verbose=1):
    train_final = './data/train_with_mrt_mall_school.csv'
    test_final = './data/test_with_mrt_mall_school.csv'
    train_final_df = pd.read_csv(train_final)
    test_final_df = pd.read_csv(test_final)
    train_df_cleaned_final = clean_data(train_final_df)
    train_df_cleaned_final = process_data(train_df_cleaned_final)
    test_df_cleaned_final = clean_data(test_final_df)
    test_df_cleaned_final = process_data(test_df_cleaned_final, mode='test')
    print(train_df_cleaned_final.columns)
    print('cuda available = ' + str(torch.cuda.is_available()))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model_conf = {
            "output_size":1, 
            "hidden_layers":1,
            "hidden_unit": [100],
            "dropout": 0.0,
            "activation": nn.ReLU()
            }

    # best
    # model_conf = {
    #         "output_size":1, 
    #         "hidden_layers":1,
    #         "hidden_unit": [100],
    #         "dropout": 0.0,
    #         "activation": nn.ReLU()
    #         }


    features = ["floor_area_sqm", "age", "town_psqm", "regional_psqm", "nearest_mrt_dist",
                "nearest_mall_dist", "near_mall_count", "near_mrt_count", "near_school_count", "nearest_school_dist", "date"]
    cat_features = ["flat_type", "flat_model", "region", "subzone", "planning_area", "town"]
    target = 'monthly_rent'
    if k_fold_val:
        print('running k-fold')
        res = train_kfold(features, cat_features, train_df_cleaned_final, target, 10, 
                BaseMLPRegressor, model_conf, epoches=epoches, feature_norm='', 
                device=device, lr=0.001, batch_size=128)
        print(res)
    out_df = generate_prediction(features, cat_features, train_final_df, test_final_df, target,
                BaseMLPRegressor, model_conf, epoches=epoches, feature_norm='', 
                device=device, lr=0.001, batch_size=128, verbose=verbose)
    # print(out_df)
    out_df.to_csv(output_name)
    # print(res)

if __name__ == "__main__":
    fire.Fire(run_nn_pipeline)
