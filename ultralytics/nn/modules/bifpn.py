import torch
import torch.nn as nn

class Concat_BIFPN(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat_BIFPN, self).__init__()
        self.d = dimension
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        if len(x) == 2:
            w = self.w1
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = [weight[0] * x[0], weight[1] * x[1]]
        elif len(x) == 3:
            w = self.w2
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]

        return torch.cat(x, self.d)
