from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

base_conf = {
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
                         'price',
                         'num_primary_school',
                         ],
    "cat_features": ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town'],
    "target": 'monthly_rent',
    "pipelines": [StandardScaler(), 
                                   MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes=(200,), 
                                                learning_rate='adaptive', early_stopping=True)]
}