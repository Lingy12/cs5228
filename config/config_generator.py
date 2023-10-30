from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import itertools
def get_mlp_hp_conf():
    hidden_layer_lst = [50, 100, 200, 500, 1000] 
    batch_size_lst = [200]
    init_learning_rate_lst = [0.001]
    early_stopping_lst = [True]
    learning_rate_lst = ['adaptive']
    scaler_lst = [None, StandardScaler(), MinMaxScaler()]
    tol_lst = [1e-4]
    validate_fraction_lst = [0.1]

    params = list(itertools.product(*[hidden_layer_lst, batch_size_lst, init_learning_rate_lst, early_stopping_lst, learning_rate_lst, scaler_lst, tol_lst, validate_fraction_lst]))
    configs = []
    names = []

    conf_template = {
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
    "cat_features": ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town'],
    "target": 'monthly_rent',
    "pipelines": []
        }
    
    for param in params:
        pipeline = []
        hidden_layer, batch_size, init_learning_rate, early_stopping, learning_rate, scaler, tol, validate_fraction = param
        if scaler:
            pipeline.append(scaler)
        pipeline.append(MLPRegressor(hidden_layer_sizes=hidden_layer, batch_size=batch_size, learning_rate_init=init_learning_rate, 
                                     early_stopping=early_stopping, learning_rate=learning_rate, tol=tol, validation_fraction=validate_fraction, max_iter=1000))
        conf_template['pipelines'] = pipeline
        configs.append(conf_template.copy())
        names.append('_'.join([str(hidden_layer), str(batch_size), str(learning_rate), str(init_learning_rate), str(early_stopping), str(tol), str(validate_fraction)]))

    return list(zip(configs, names))

def get_scaler_conf():
    model = [LinearRegression(), MLPRegressor(hidden_layer_sizes=200, batch_size=200, learning_rate_init=0.001, early_stopping=True, learning_rate='adaptive', max_iter=1000)]
    scaler_lst = [None, StandardScaler(), MinMaxScaler()]

    params = list(itertools.product(*[model, scaler_lst]))
    configs = []
    names = []

    conf_template = {
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
    "cat_features": ['flat_type', 'flat_model', 'region', 'subzone', 'planning_area', 'town'],
    "target": 'monthly_rent',
    "pipelines": []
        }
    
    for param in params:
        pipeline = []
        model, scaler = param
        if scaler:
            pipeline.append(scaler)
        pipeline.append(model)
        conf_template['pipelines'] = pipeline
        configs.append(conf_template.copy())
        names.append('_'.join([str(model), str(scaler)]))

    return list(zip(configs, names))