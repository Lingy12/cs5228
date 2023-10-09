from data_processing import clean_data, process_data
import pandas as pd
from torch import cat
from torch_models import BaseMLPRegressor
from nn_utils import train_kfold
train_final = './data/train_with_mrt_mall_school.csv'
train_final_df = pd.read_csv(train_final)

train_df_cleaned_final = clean_data(train_final_df)
train_df_cleaned_final = process_data(train_df_cleaned_final)

print(train_df_cleaned_final.columns)


model_conf = {
        "output_size":1, 
        "hidden_layers":1,
        "hidden_unit": [100],
        "dropout": 0.0
        }


features = ["floor_area_sqm", "age", "town_psqm", "regional_psqm", "nearest_mrt_dist",
            "nearest_mall_dist", "near_mall_count", "near_mrt_count", "near_school_count", "nearest_school_dist", "date"]
cat_features = ["flat_type", "flat_model", "region", "subzone", "planning_area", "town"]
target = 'monthly_rent'
train_kfold(features, cat_features, train_df_cleaned_final, target, 10, 
            BaseMLPRegressor, model_conf, epoches=1000, feature_norm='', 
            device='cuda:0', lr=0.001, batch_size=128)

