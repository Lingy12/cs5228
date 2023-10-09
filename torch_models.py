from torch import nn

class BaseMLPRegressor(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_unit, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        assert len(hidden_unit) == hidden_layers 
        self.layers = []
        for i in range(hidden_layers + 1):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_unit[0]))
            elif i == hidden_layers:
                self.layers.append(nn.Linear(hidden_unit[-1], output_size))
            else:
                self.layers.append(nn.Linear(hidden_unit[i-1], hidden_unit[i]))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        output = x
        for i in range(len(self.layers)):
            output = self.layers[i](output)
            if i != len(self.layers) - 1:
                output = self.activation(output)
                output = self.dropout(output)
        return output


