from torch import nn

base_model_conf = {
            "output_size":1, 
            "hidden_layers":1,
            "hidden_unit": [100],
            "dropout": 0.0,
            "activation": nn.ReLU()
            }