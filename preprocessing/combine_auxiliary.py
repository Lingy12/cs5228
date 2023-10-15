import pandas as pd
from preprocessing.data_processing import haversine
# import swifter
from tqdm import tqdm
import sys, os
sys.path.append('..')
sys.path.append(os.getcwd())
tqdm.pandas()

existing_mrt_data = './data/auxiliary-data/sg-mrt-existing-stations.csv'
school_data = './data/auxiliary-data/sg-primary-schools.csv'
shopping_mall_data = './data/auxiliary-data/sg-shopping-malls.csv'
train_data = './data/train.csv'
test_data = './data/test.csv'

mrt_df = pd.read_csv(existing_mrt_data)
school_df = pd.read_csv(school_data)
shopping_mall_df = pd.read_csv(shopping_mall_data)

train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)

def produce_mrt_features(df, mrt_df, thresholds):
    df = df.copy()
    df = df.reset_index().rename(columns={"index": "list_id"})
    #find nearest mrt station, number of mrt within threshold t, distance to the nearest mrt
    df_explode = df.merge(mrt_df, how='cross', suffixes=('_hdb', '_mrt'))
    df_explode[f'mrt_dist'] = df_explode.progress_apply(lambda x: haversine(x['latitude_hdb'], x['longitude_hdb'], x['latitude_mrt'], x['longitude_mrt']), axis=1)
    
    # cacluate nearest
    nearest_mrt_dist = df_explode.iloc[df_explode.groupby('list_id')['mrt_dist'].idxmin()][['code', 'name', 'opening_year', 'mrt_dist']].reset_index(drop=True)
    
    # count mrt within threshold
    for threshold in thresholds:
        near_mrt_count = df_explode.groupby('list_id')['mrt_dist'].progress_apply(lambda x: (x <= threshold).sum()).reset_index(name='near_count')
        df[f'near_mrt_count_{threshold}'] = near_mrt_count['near_count']
    nearest_mrt_dist = nearest_mrt_dist.rename(columns={"code": "nearest_code", "name": "nearest_mrt_name", "opening_year": "nearest_open_year", "mrt_dist": "nearest_mrt_dist"})
    
    # appending data
    df = pd.concat([df, nearest_mrt_dist], axis='columns')
    
    return df

def produce_school_features(df, school_df, thresholds):
    df = df.copy()
    df = df.reset_index().rename(columns={"index": "list_id"})
    #find nearest mrt station, number of mrt within threshold t, distance to the nearest mrt
    df_explode = df.merge(school_df, how='cross', suffixes=('_hdb', '_school'))
    df_explode[f'school_dist'] = df_explode.progress_apply(lambda x: haversine(x['latitude_hdb'], x['longitude_hdb'], x['latitude_school'], x['longitude_school']), axis=1)
    
    # cacluate nearest
    nearest_school_dist = df_explode.iloc[df_explode.groupby('list_id')['school_dist'].idxmin()][['name', 'school_dist']].reset_index(drop=True)
    
    # count mrt within threshold
    for threshold in thresholds:
        near_mrt_count = df_explode.groupby('list_id')['school_dist'].progress_apply(lambda x: (x <= threshold).sum()).reset_index(name='near_count')
        df[f'near_school_count_{threshold}'] = near_mrt_count['near_count']
    nearest_school_dist = nearest_school_dist.rename(columns={"name": "nearest_school_name", "school_dist": "nearest_school_dist"})
    
    # appending data
    df = pd.concat([df, nearest_school_dist], axis='columns')
    
    return df

def produce_mall_features(df, mall_df, thresholds):
    df = df.copy()
    df = df.reset_index().rename(columns={"index": "list_id"})
    #find nearest mrt station, number of mrt within threshold t, distance to the nearest mrt
    df_explode = df.merge(mall_df, how='cross', suffixes=('_hdb', '_mall'))
    df_explode[f'mall_dist'] = df_explode.progress_apply(lambda x: haversine(x['latitude_hdb'], x['longitude_hdb'], x['latitude_mall'], x['longitude_mall']), axis=1)
    
    # cacluate nearest
    nearest_mall_dist = df_explode.iloc[df_explode.groupby('list_id')['mall_dist'].idxmin()][['name', 'mall_dist']].reset_index(drop=True)
    
    # count mrt within threshold
    for threshold in thresholds:
        near_mall_count = df_explode.groupby('list_id')['mall_dist'].progress_apply(lambda x: (x <= threshold).sum()).reset_index(name='near_count')
        df[f'near_mall_count_{threshold}'] = near_mall_count['near_count']
    nearest_mall_dist = nearest_mall_dist.rename(columns={"name": "nearest_mall_name", "mall_dist": "nearest_mall_dist"})
    
    # appending data
    df = pd.concat([df, nearest_mall_dist], axis='columns')
    
    return df

def produce_coe_features(df, coe_df):
    df, coe_df = df.copy(), coe_df.copy()

    # df_other['rent_approval_date'] = pd.to_datetime(df_other['rent_approval_date'])
    month_to_num = {'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06', 'july': '07', 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'}
    coe_df['month'] = coe_df['month'].map(month_to_num)
    #
    coe_df['year'] = coe_df['year'].astype(str)  
    coe_df['month'] = coe_df['month'].astype(str)  

    coe_df['rent_approval_date'] = coe_df['year'].str.cat(coe_df['month'], sep='-')
    # print(df['rent_approval_date'])
    result = coe_df.groupby(['year', 'category', 'rent_approval_date']).agg({'price': 'mean', 'quota': 'mean', 'bids': 'mean'}).reset_index()
    result = result.drop(columns = ['category','year','quota','bids'])
    result = result.groupby('rent_approval_date')['price'].mean().reset_index()
    # print(result['rent_approval_date'][12])
    result_merged = pd.merge(df, result, on='rent_approval_date', how='left')
    # result_merged = result_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    return result_merged

'''
Handle stock features data
left table rent house data, right table stock data, they join by month(yyyy-mm)
reutrn stock_price 
'''
def produce_stock_features(df, stock_df):
    df, stock_df = df.copy(), stock_df.copy()
    # Convert day (yyyy-mm-dd) to month (yyyy-mm)
    stock_df['rent_approval_date'] = pd.to_datetime(stock_df['date']).dt.to_period('M')
    stock_df['rent_approval_date'] = stock_df['rent_approval_date'].astype(str)
    # print(stock_df['rent_approval_date'])
    # Rename adjusted_close to price
    stock_df.rename(columns={"adjusted_close": "stock_price"}, inplace=True)
    # Group price by month (yyyy-mm)
    result = stock_df.groupby(['date', 'symbol', 'rent_approval_date']).agg({'stock_price': 'mean'}).reset_index()
    result = result.drop(columns = ['symbol','date'])
    result = result.groupby('rent_approval_date')['stock_price'].mean().reset_index()
    # print(result)
    result_merged = pd.merge(df, result, on='rent_approval_date', how='left')
    return result_merged
    
     
    