from training.k_fold import run
import config.config_generator as config_gen
import fire
import json
import os

def run_gen(generator, run_name):
    config_func = getattr(config_gen, generator)
    
    res_dict = {}
    configs = config_func()

    for conf, name in configs:
        res = run(conf, is_hp_tune=True)
        res_dict[name] = res 

    with open(os.path.join('logs', run_name + '.json'), 'w') as f:
        json.dump(res_dict, f)
    
if __name__ == '__main__':
    fire.Fire(run_gen)