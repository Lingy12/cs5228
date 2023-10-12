import sys
sys.path.append('..')
import pandas as pd
from preprocessing.data_processing import *
# from tqdm import tqdm
from preprocessing.combine_auxiliary import produce_mall_features, produce_mrt_features, produce_school_features

# tqdm.pandas()

existing_mrt_data = '../data/auxiliary-data/sg-mrt-existing-stations.csv'
school_data = '../data/auxiliary-data/sg-primary-schools.csv'
shopping_mall_data = '../data/auxiliary-data/sg-shopping-malls.csv'
train_data = '../data/train.csv'
test_data = '../data/test.csv'

mrt_df = pd.read_csv(existing_mrt_data)
school_df = pd.read_csv(school_data)
shopping_mall_df = pd.read_csv(shopping_mall_data)

train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)

train_with_mrt = produce_mrt_features(train_df, mrt_df, 1)
train_with_mrt.to_csv('../data/train_with_mrt.csv')
test_with_mrt = produce_mrt_features(test_df, mrt_df, 1)
train_with_mrt.to_csv('../data/test_with_mrt.csv')

if 'list_id' in train_with_mrt.columns:
    train_with_mrt = train_with_mrt.drop(columns=['list_id'])
    test_with_mrt = test_with_mrt.drop(columns=['list_id'])
# train_with_mrt.columns
train_with_mrt_mall = produce_mall_features(train_with_mrt, shopping_mall_df, 1)
train_with_mrt_mall.to_csv('../data/train_with_mrt_mall.csv')
test_with_mrt_mall = produce_mall_features(test_with_mrt, shopping_mall_df, 1)
test_with_mrt_mall.to_csv('../data/test_with_mrt_mall.csv')

if 'list_id' in train_with_mrt_mall:
    train_with_mrt_mall = train_with_mrt_mall.drop(columns=['list_id'])
    test_with_mrt_mall = test_with_mrt_mall.drop(columns=['list_id'])
train_with_mrt_mall.columns
train_with_mrt_mall_school = produce_school_features(train_with_mrt_mall, school_df, 1)
train_with_mrt_mall_school.to_csv('../data/train_with_mrt_mall_school.csv')
test_with_mrt_mall_school = produce_school_features(test_with_mrt_mall, school_df, 1)
test_with_mrt_mall_school.to_csv('../data/test_with_mrt_mall_school.csv')