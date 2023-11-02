from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import itertools

def get_scaler_conf():
    model = [LinearRegression(), MLPRegressor(random_state=1, hidden_layer_sizes=200, batch_size=200, learning_rate_init=0.001, early_stopping=True, learning_rate='adaptive', max_iter=1000)]
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