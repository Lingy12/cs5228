import config.sk_config as confs
from training.k_fold import run
# from inspect import getmembers, isfunction

conf_lst = sorted(list(filter(lambda x: x[-4:] == 'conf', dir(confs))), key=lambda x: (x.split('_')[0], int(x.split('_')[1]))) # order the configuration

print('running benchmark for the folloing'.center(100, '='))
print(conf_lst)

for conf in conf_lst:
    print('Start run with config name = {}'.format(conf).center(100, '='))
    run(conf)
    print('Finished'.center(100, '='))

