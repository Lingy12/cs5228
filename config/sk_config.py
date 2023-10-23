from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

# --------- Original data -----------#
lg_0_conf = {
    "features": ['floor_area_sqm'],
    "cat_features": [],
    "target": ['monthly_rent'],
    "pipelines": [StandardScaler(), LinearRegression()]
}

lg_1_conf = {
    "features": ['floor_area_sqm', 'age'],
    "cat_features": [],
    "target": ['monthly_rent'],
    "pipelines": [StandardScaler(), LinearRegression()]
}

lg_2_conf = {
    "features": ['floor_area_sqm', 'age', 'date'],
    "cat_features": [],
    "target": ['monthly_rent'],
    "pipelines": [StandardScaler(), LinearRegression()]
}

lg_3_conf = {
    "features": ['floor_area_sqm',
                         'age',
                         'date',
                         'regional_psqm'],
    "cat_features": [],
    "target": ['monthly_rent'],
    "pipelines": [StandardScaler(), LinearRegression()]
}

lg_4_conf = {
    "features": ['floor_area_sqm',
                         'age',
                         'date',
                         'town_psqm'],
    "cat_features": [],
    "target": ['monthly_rent'],
    "pipelines": [StandardScaler(), LinearRegression()]
}

lg_5_conf = {
    "features": ['floor_area_sqm',
                         'age',
                         'date',
                         'town_psqm','regional_psqm'],
    "cat_features": [],
    "target": ['monthly_rent'],
    "pipelines": [StandardScaler(), LinearRegression()]
}

lg_6_conf = {
    "features": ['floor_area_sqm',
                         'age',
                         'date',
                         'town_psqm','regional_psqm'],
    "cat_features": ['flat_type'],
    "target": ['monthly_rent'],
    "pipelines": [StandardScaler(), LinearRegression()]
}

lg_7_conf = {
    "features": ['floor_area_sqm',
                         'age',
                         'date',
                         'town_psqm','regional_psqm'],
    "cat_features": ['flat_model'],
    "target": ['monthly_rent'],
    "pipelines": [StandardScaler(), LinearRegression()]
}

lg_8_conf = {
    "features": ['floor_area_sqm',
                         'age',
                         'date',
                         'town_psqm','regional_psqm'],
    "cat_features": ['flat_model', 'flat_type'],
    "target": ['monthly_rent'],
    "pipelines": [StandardScaler(), LinearRegression()]
}

lg_9_conf = {
    "features": ['floor_area_sqm',
                         'age',
                         'date',
                         'town_psqm','regional_psqm'],
    "cat_features": ['flat_model', 'flat_type', 'region', 'subzone', 'town', 'planning_area'],
    "target": ['monthly_rent'],
    "pipelines": [StandardScaler(), LinearRegression()]
}



lg_12_conf = {
    "features": ['floor_area_sqm',
                         'age',
                         'date',
                         'town_psqm','regional_psqm'],
    "cat_features": ['flat_model', 'flat_type', 'region', 'subzone', 'town', 'planning_area'],
    "target": ['monthly_rent'],
    'feature_norm': 'z_norm',
    "pipelines": [LinearRegression()]
}


# with additional data
lg_10_conf = {
    "features": ['floor_area_sqm',
                         'age',
                         'date',
                         'town_psqm','regional_psqm', 'nearest_mrt_dist', 'nearest_mall_dist', 'nearest_school_dist'],
    "cat_features": ['flat_model', 'flat_type', 'region', 'subzone', 'town', 'planning_area'],
    "target": ['monthly_rent'],
    "pipelines": [StandardScaler(), LinearRegression()]
}

lg_11_conf = {
    "features": ['floor_area_sqm',
                         'age',
                         'town_psqm',
                         'regional_psqm',  
                         'nearest_mrt_dist',
                          'near_mrt_count_1.0',
                         'near_mall_count_1.0',
                         'near_school_count_1.0', 
                         'nearest_mall_dist',
                         'nearest_school_dist',
                         'date',
                         ],
    "cat_features": ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town'],
    "target": 'monthly_rent',
    "pipelines": [StandardScaler(), LinearRegression()]
}

mlp_1_conf = {
 "features": ['floor_area_sqm',
                         'age',
                         'town_psqm',
                         'regional_psqm',  
                         'nearest_mrt_dist',
                          'near_mrt_count_1.0',
                         'near_mall_count_1.0',
                         'near_school_count_1.0', 
                         'nearest_mall_dist',
                         'nearest_school_dist',
                         'date',
                         'price'
                         ],
    "cat_features": ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town'],
    "target": 'monthly_rent',
    "pipelines": [StandardScaler(), 
                                   MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes=(100,), )]
}

mlp_2_conf = {
 "features": ['floor_area_sqm',
                         'age',
                         'town_psqm',
                         'regional_psqm',  
                         'nearest_mrt_dist',
                          'near_mrt_count_1.0',
                         'near_mall_count_1.0',
                         'near_school_count_1.0', 
                         'nearest_mall_dist',
                         'nearest_school_dist',
                         'date',
                         'price'
                         ],
    "cat_features": ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town'],
    "target": 'monthly_rent',
    "pipelines": [StandardScaler(), 
                                   MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes=(200,), )]
}

mlp_3_conf = {
 "features": ['floor_area_sqm',
                         'age',
                         'town_psqm',
                         'regional_psqm',  
                         'nearest_mrt_dist',
                          'near_mrt_count_1.0',
                         'near_mall_count_1.0',
                         'near_school_count_1.0', 
                         'nearest_mall_dist',
                         'nearest_school_dist',
                         'date',
                         'price'
                         ],
    "cat_features": ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town'],
    "target": 'monthly_rent',
    "pipelines": [StandardScaler(), 
                                   MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes=(500,), )]
}

mlp_4_conf = {
        "features": ['floor_area_sqm',
                         'age',
                         'town_psqm',
                         'regional_psqm',  
                         'nearest_mrt_dist',
                          'near_mrt_count_1.0',
                         'near_mall_count_1.0',
                         'near_school_count_1.0', 
                         'nearest_mall_dist',
                         'nearest_school_dist',
                         'date',
                         'price'
                         ],
    "cat_features": ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town'],
    "target": 'monthly_rent',
    "pipelines": [StandardScaler(), 
                                   MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes=(100,), 
                                                learning_rate='adaptive', early_stopping=True)]
}

mlp_5_conf = {
        "features": ['floor_area_sqm',
                         'age',
                         'town_psqm',
                         'regional_psqm',  
                         'nearest_mrt_dist',
                          'near_mrt_count_1.0',
                         'near_mall_count_1.0',
                         'near_school_count_1.0', 
                         'nearest_mall_dist',
                         'nearest_school_dist',
                         'date',
                         'price'
                         ],
    "cat_features": ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town'],
    "target": 'monthly_rent',
    "pipelines": [StandardScaler(), 
                                   MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes=(200,), 
                                                learning_rate='adaptive', early_stopping=True)]
}

mlp_6_conf = {
        "features": ['floor_area_sqm',
                         'age',
                         'town_psqm',
                         'regional_psqm',  
                         'nearest_mrt_dist',
                          'near_mrt_count_1.0',
                         'near_mall_count_1.0',
                         'near_school_count_1.0', 
                         'nearest_mall_dist',
                         'nearest_school_dist',
                         'date',
                         'price'
                         ],
    "cat_features": ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town'],
    "target": 'monthly_rent',
    "pipelines": [StandardScaler(), 
                                   MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes=(500,), 
                                                learning_rate='adaptive', early_stopping=True)]
}

mlp_7_conf = {
        "features": ['floor_area_sqm',
                         'age',
                         'town_psqm',
                         'regional_psqm',  
                         'nearest_mrt_dist',
                          'near_mrt_count_1.0',
                         'near_mall_count_1.0',
                         'near_school_count_1.0', 
                         'nearest_mall_dist',
                         'nearest_school_dist',
                         'date',
                         'price'
                         ],
    "cat_features": ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town'],
    "target": 'monthly_rent',
    "pipelines": [StandardScaler(), 
                                   MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes=(500,), 
                                                learning_rate='adaptive', early_stopping=True, alpha=0.001)]
}

# base_conf = {
#     "features": ['floor_area_sqm',
#                          'age',
#                          'town_psqm',
#                          'regional_psqm',  
#                          'nearest_mrt_dist',
#                           'near_mrt_count_1.0',
#                          'near_mall_count_1.0',
#                          'near_school_count_1.0', 
#                          'nearest_mall_dist',
#                          'nearest_school_dist',
#                          'date',
#                          'price'
#                          ],
#     "cat_features": ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town'],
#     "target": 'monthly_rent',
#     "pipelines": [StandardScaler(), 
#                                    MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes=(200,), 
#                                                 learning_rate='adaptive', early_stopping=True)]
# }