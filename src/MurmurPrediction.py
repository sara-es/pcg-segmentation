import torch.nn as nn

# model definition
class DNN(nn.Module):
    # define model elements
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.hidden = nn.ModuleList()
        self.batches = nn.ModuleList() 

        for in_size, out_size in zip(layers, layers[1:]):
            self.hidden.append(nn.Linear(in_size, out_size))
        
        for i in (layers[1:]):
            self.batches.append(nn.BatchNorm1d(i))

    # forward propagate input
    def forward(self, activation):
        L = len(self.hidden)
        for (i, linear_transform, batch_norm) in zip(range(L), self.hidden, self.batches):
            if (i<L-1):
                activation = batch_norm(nn.functional.relu(linear_transform(activation)))
            else:
                activation = linear_transform(activation)
        
        return activation
