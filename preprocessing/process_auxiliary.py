import sys
import os
sys.path.append('..')
sys.path.append(os.getcwd())
import pandas as pd
from preprocessing.data_processing import *
# from tqdm import tqdm
from preprocessing.combine_auxiliary import produce_mall_features, produce_mrt_features, produce_school_features, produce_coe_features, produce_stock_features

# tqdm.pandas()

existing_mrt_data = './data/auxiliary-data/sg-mrt-existing-stations.csv'
school_data = './data/auxiliary-data/sg-primary-schools.csv'
shopping_mall_data = './data/auxiliary-data/sg-shopping-malls.csv'
coe_data = './data/auxiliary-data/sg-coe-prices.csv'
stock_data = './data/auxiliary-data/sg-stock-prices.csv'
train_data = './data/train.csv'
test_data = './data/test.csv'

mrt_df = pd.read_csv(existing_mrt_data)
school_df = pd.read_csv(school_data)
shopping_mall_df = pd.read_csv(shopping_mall_data)
coe_df = pd.read_csv(coe_data)
stock_df = pd.read_csv(stock_data)

mrt_radius = [0.3, 0.8, 1.0, 1.5, 2] # from analysis
mall_radius = [0.5, 1.0, 1.5, 2]
school_radius = [0.5, 1.0, 1.5, 2]
train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)

print('generating mrt feature with radius {}'.format(mrt_radius))
if not os.path.exists('./data/train_with_mrt.csv'):
    if 'list_id' in train_df:
        train_df = train_df.drop(columns=['list_id'])
        test_df = test_df.drop(columns=['list_id'])
    train_df = produce_mrt_features(train_df, mrt_df, mrt_radius)
    train_df.to_csv('./data/train_with_mrt.csv')
    test_df = produce_mrt_features(test_df, mrt_df, mrt_radius)
    test_df.to_csv('./data/test_with_mrt.csv')
else:
    train_df, test_df = pd.read_csv('./data/train_with_mrt.csv'), pd.read_csv('./data/test_with_mrt.csv')
    print('mrt feature exists and loaded')
    

print('generating mall feature with radius {}'.format(mall_radius))
if not os.path.exists('./data/train_with_mrt_mall.csv'):
    if 'list_id' in train_df:
        train_df = train_df.drop(columns=['list_id'])
        test_df = test_df.drop(columns=['list_id'])
    # train_with_mrt.columns
    train_df = produce_mall_features(train_df, shopping_mall_df, mall_radius)
    train_df.to_csv('./data/train_with_mrt_mall.csv')
    test_df = produce_mall_features(test_df, shopping_mall_df, mall_radius)
    test_df.to_csv('./data/test_with_mrt_mall.csv')
else:
    train_df, test_df = pd.read_csv('./data/train_with_mrt_mall.csv'), pd.read_csv('./data/test_with_mrt_mall.csv')
    print('mall feature exists and loaded')

print('generating school feature with radius {}'.format(school_radius))
if not os.path.exists('./data/train_with_mrt_mall_school.csv'):
    if 'list_id' in train_df:
        train_df = train_df.drop(columns=['list_id'])
        test_df = test_df.drop(columns=['list_id'])

    train_df = produce_school_features(train_df, school_df, school_radius)
    train_df.to_csv('./data/train_with_mrt_mall_school.csv')
    test_df = produce_school_features(test_df, school_df, school_radius)
    test_df.to_csv('./data/test_with_mrt_mall_school.csv')
else:
    train_df, test_df = pd.read_csv('./data/train_with_mrt_mall_school.csv'), pd.read_csv('./data/test_with_mrt_mall_school.csv')
    print('school feature exists and loaded')

print('generating coe feature')
if not os.path.exists('./data/train_with_mrt_mall_school_coe.csv'):
    if 'list_id' in train_df:
        train_df = train_df.drop(columns=['list_id'])
        test_df = test_df.drop(columns=['list_id'])

    train_df = produce_coe_features(train_df, coe_df)
    train_df.to_csv('./data/train_with_mrt_mall_school_coe.csv')
    test_df = produce_coe_features(test_df, coe_df)
    test_df.to_csv('./data/test_with_mrt_mall_school_coe.csv')
else:
    train_df, test_df = pd.read_csv('./data/train_with_mrt_mall_school_coe.csv'), pd.read_csv('./data/test_with_mrt_mall_school_coe.csv')
    print('coe feature exists and loaded')

'''
Handle stock feature
add stock_price
'''
print('generating stock feature')
if not os.path.exists('./data/train_with_mrt_mall_school_stock.csv'):
    if 'list_id' in train_df:
        train_df = train_df.drop(columns=['list_id'])
        test_df = test_df.drop(columns=['list_id'])

    train_df = produce_stock_features(train_df, stock_df)
    train_df.to_csv('./data/train_with_mrt_mall_school_stock.csv')
    test_df = produce_stock_features(test_df, stock_df)
    test_df.to_csv('./data/test_with_mrt_mall_school_stock.csv')
else:
    train_df, test_df = pd.read_csv('./data/train_with_mrt_mall_school_stock.csv'), pd.read_csv('./data/test_with_mrt_mall_school_stock.csv')
    print('stock feature exists and loaded')

train_df.to_csv('./data/train_final.csv')
test_df.to_csv('./data/test_final.csv')