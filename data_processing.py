from datetime import datetime
import pandas as pd
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import math
approval_date_f = '%Y-%m'
commence_f = '%Y'

# ## preprocessing

# 1. Convert rent_approval_date and lease_commence_data to datetime format
# 2. make other string entry to be lower case
def clean_data(df):
    df = df.copy()
    string_attr = ['region', 'town', 'planning_area', 'subzone', 'town', 'block', 'street_name', 'flat_type', 'flat_model']

    for string_data in string_attr:
        df[string_data] = df[string_data].map(lambda x: x.lower())

    df.flat_type = df.flat_type.map(lambda x: x if x.split(' ') == 1 else '-'.join(x.split(' '))) # normalize the flat type
    df = df.drop(columns=['furnished', 'elevation'])
    df['flat_model'] = df['flat_model'].replace('2-room', '2-room-model')
   
    return df

def process_data(df, mode = 'train'):
  df = df.copy()
  df['type_model'] = df['flat_model'] + ' ' + df['flat_type']
  if mode == 'train':
    model_le = LabelEncoder()
    model_le.fit(df['flat_model'])
    type_le = LabelEncoder()
    type_le.fit(df['flat_type'])
    df['model_digit'] = model_le.transform(df['flat_model']) 
    df['type_digit'] = type_le.transform(df['flat_type']) 
  
  if mode == 'train':
    df['psqm'] = df['monthly_rent'] / df['floor_area_sqm']
  df['date'] = df['rent_approval_date'].map(lambda x: datetime.strptime(x, approval_date_f).timestamp())
  df['age'] = df.apply(lambda x: (datetime.strptime(x['rent_approval_date'], approval_date_f) - datetime.strptime(str(x['lease_commence_date']), commence_f)).days, axis=1)
  df['age_year'] = df['age'] / 360
  df['age_bin'] = bin_values(df['age_year'], step_size=5, right_bound=30, transformation_function=lambda x: 1.5*x)
  return df

def combined_encoding(train_df, test_df, cat_features, is_val=False):
    norm_columns = ['age', 'town_psqm','regional_psqm','date', 'floor_area_sqm']
    train_df, test_df = train_df.copy(), test_df.copy()
    test_df['monthly_rent'] = 'NA' if not is_val else test_df['monthly_rent']
    test_df['split'] = 'test'
    train_df['split'] = 'train'
    df = pd.concat([train_df, test_df])
    one_hot = pd.get_dummies(df[cat_features], columns=cat_features).astype(int)
    df = pd.concat([df, one_hot], axis='columns')
    
    for col in norm_columns:
        df[col + '_z_norm'] = z_norm_col(col, df)
        df[col + '_min_max_norm'] = min_max_col(col, df)
    # print(len(df))
    return df
    
def generate_features(train_df, test_df):
    train_df, test_df = train_df.copy(), test_df.copy()
    town_map, region_map = get_grouped_psqm(train_df, 'town').to_dict(), get_grouped_psqm(train_df, 'region').to_dict()
    train_df['town_psqm'] = train_df.apply(lambda x: town_map[x['town']][x['rent_approval_date']], axis=1)
    train_df['regional_psqm'] = train_df.apply(lambda x: region_map[x['region']][x['rent_approval_date']], axis=1)
    test_df['town_psqm'] = test_df.apply(lambda x: town_map[x['town']][x['rent_approval_date']], axis=1)
    test_df['regional_psqm'] = test_df.apply(lambda x: region_map[x['region']][x['rent_approval_date']], axis=1)
    return train_df, test_df

def z_norm_col(column, df_z_scaled):
    return (df_z_scaled[column] - df_z_scaled[column].mean()) / df_z_scaled[column].std()

def min_max_col(column, df_min_max_scaled):
    return (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())

def min_max_norm(column, df):
  df_min_max_scaled = df.copy()
  df_min_max_scaled[column + '_min_max_norm'] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())
  return df_min_max_scaled

def z_norm(column, df):
  df_z_scaled = df.copy()
  df_z_scaled[column + '_z_norm'] = (df_z_scaled[column] - df_z_scaled[column].mean()) / df_z_scaled[column].std()
  return df_z_scaled

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the earth in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c  # Distance in km
    return d

def cluster_rental_location(data, n_clusters):

    df = data.copy()

    # Number of clusters (change this based on your requirement)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df[['latitude', 'longitude']])
    df['cluster'] = kmeans.labels_
    return df

def get_grouped_psqm(df, target_attr):
    df = df.copy()
    trained_df_group_avg = df.groupby(['rent_approval_date', target_attr]).mean(numeric_only=True).reset_index()
    pivot_df = trained_df_group_avg.pivot(index='rent_approval_date', columns=target_attr, values='psqm')
    
    # fill na with average
    for col in pivot_df.columns:
        pivot_df[col].fillna(float(pivot_df[col].mean()), inplace=True)
    return pivot_df

def generate_perc(df, col, grouping, target):
    df = df.copy()
    df[col] = df.groupby(grouping)[target].rank(pct=True)
    return df

def bin_values(series, step_size, right_bound=None, transformation_function=lambda x: x):

    
    # Infer min and max from the series
    min_value = series.min()
    if right_bound is None:
        max_value = series.max()
    else:
        max_value = right_bound

    # Create bins
    bins = [min_value]
    while bins[-1] < max_value:
        next_step = transformation_function(step_size)
        if bins[-1] + next_step >= max_value:
            bins.append(max_value)
        else:
            bins.append(bins[-1] + next_step)
        step_size = next_step

    # Use pandas cut function to bin values
    binned_series = pd.cut(series, bins=bins, include_lowest=True, right=False)

    return binned_series





