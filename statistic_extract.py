import re
import pandas as pd
import fire
def fetch_data_dict(data):
    # Regular expression patterns
    config_block_pattern = r"Start run with config name = (\w+)([\s\S]+?)==============================================Finished=============================================="
    config_details_pattern = r"running with \w+[\s\S]+?'features': (\[[^\]]+\]), 'cat_features': (\[[^\]]+\]), 'pipelines': (\[[^\]]+\])"
    overall_eval_pattern = r"Overall evaluation([\s\S]+?)(?=(Overall evaluation|Finished|$))"
    stat_pattern = r"(\w+): average = ([\d\.]+), std_dev = ([\d\.]+)"

    # Extract blocks for each config
    config_blocks = re.findall(config_block_pattern, data)

    config_to_eval = {}
    for block in config_blocks:
        # Extract config name, details and their respective overall evaluation sessions from each block
        config_name = block[0]
        # config_details = re.search(config_details_pattern, block[1])
        overall_eval = re.search(overall_eval_pattern, block[1])
        # print(config_details)
        # Extract statistics from the overall evaluation session
        stats = re.findall(stat_pattern, overall_eval.group(1))
        print(stats)
        stat_dict = {}
        for stat in stats:
            stat_name, avg, std = stat
            combined = f"{float(avg):.2f} ± {float(std):.2f}"
            stat_dict[f"{stat_name}_average"] = float(avg)
            stat_dict[f"{stat_name}_std_dev"] = float(std)
            stat_dict[f"{stat_name}_combined"] = combined
        
        # Add 'features', 'cat_features', and 'pipelines' details to the dictionary
        # features, cat_features, pipelines = config_details.groups()
        # stat_dict["features"] = features
        # stat_dict["cat_features"] = cat_features
        # stat_dict["pipelines"] = pipelines
        # print(stat_dict)
        config_to_eval[config_name] = stat_dict
    print(config_to_eval.keys())
    return pd.DataFrame.from_dict(config_to_eval, orient='index')



def get_data_table(benchmarks_files, dest):
    with open(benchmarks_files, 'r') as f:
        data = f.read()
    # print(data)
    fetch_data_dict(data).reset_index().rename(columns={'index': 'conf_name'})[['conf_name', 'train_mae_combined', 'r2_train_combined', 'val_mape_combined', 'val_pcc_combined', 'val_mae_combined']].to_excel(dest, index=False)

if __name__ == '__main__':
    fire.Fire(get_data_table)
    