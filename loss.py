import torch
import torch.nn.functional as F
import torch.nn as nn
# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        #print(inputs.size())
        return self.nll_loss(F.log_softmax(inputs,dim=1), targets)

class BCE_Loss(nn.Module):
    def __init__(self):
        super(BCE_Loss,self).__init__()
        self.bce=nn.BCELoss()
    def forward(self,inputs,targets):
        return self.bce(inputs,targets)