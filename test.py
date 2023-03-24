import torch
import numpy as np
from avalanche.models import SimpleMLP, IcarlNet, make_icarl_net, initialize_icarl_net
from benchmark import make_tensors

model: IcarlNet = make_icarl_net(num_classes=25)
model.apply(initialize_icarl_net)
#model.load_state_dict(torch.load('saved_model.pth'))
model.eval()

def loss(outputs, labels):
    batch_size = outputs.size()[0]            # batch_size
    outputs = torch.log_softmax(outputs, dim=1)   # compute the log of softmax values
    outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
    return -torch.sum(outputs)/batch_size

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


torch_data, targets = make_tensors('a')

x = torch_data[:2][0]
y = torch_data[:2][1]

with torch.no_grad():
    out = model(x)
    print(out)
