from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

base_mlp_conf = {
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
                        #  'price'
                         ],
    "norm_features": ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town'],
    "ord_features": [],
    "target": 'monthly_rent',
    "pipelines": {'num_processor': [StandardScaler()], 
                    'ord_cat_processor': [OneHotEncoder(sparse_output=False)], 
                    'norm_cat_processor': [OneHotEncoder(sparse_output=False)], # ensure the concatenation 
                    'model': [MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes=(200,), 
                                                learning_rate='adaptive', early_stopping=True, validation_fraction=0.01)]}
}

# best_mlp_conf = {
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
#                         #  'price'
#                          ],
#     "cat_features": ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town'],
#     "target": 'monthly_rent',
#     "pipelines": [StandardScaler(), 
#                                    MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes=(200,), 
#                                                 learning_rate='adaptive', early_stopping=True, validation_fraction=0.0)]
# }