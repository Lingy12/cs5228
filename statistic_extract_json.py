import pandas as pd
import fire
import json
import numpy as np

def load_as_df(json_file, name):
    with open(json_file, 'r') as f:
        res_dict = json.load(f)
    
    data_dict = {}

    for setting in res_dict.keys():
        data_dict[setting] = {}
        for key in res_dict[setting].keys():
            avg, std = np.mean(res_dict[setting][key]), np.std(res_dict[setting][key]) 
            data_dict[setting][key] = f"{float(avg):.2f} ± {float(std):.2f}"
    
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df.reset_index().rename(columns={'index': 'conf_name'}).to_excel('{}.xlsx'.format(name))
    # print(df)
if __name__ == '__main__':
    fire.Fire(load_as_df)