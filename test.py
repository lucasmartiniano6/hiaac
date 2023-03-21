import torch
from avalanche.models import SimpleMLP
from numpy import genfromtxt

model = SimpleMLP(num_classes=25, input_size=51, hidden_size=512, hidden_layers=2)
model.load_state_dict(torch.load('saved_model.pth'))
model.eval()

def load_file():
    #path = 'PAMAP2_Dataset/Protocol/subject108.dat'
    path = 'a.txt'
    data = torch.from_numpy(genfromtxt(path))

    x = [data[i][3:] for i in range(len(data))]
    y = [data[i][1] for i in range(len(data))]

    x = torch.stack(x).to(torch.float32)
    y = torch.stack(y).to(torch.long)

    assert (torch.any(torch.isnan(x)) == False)

    x = torch.nn.functional.normalize(x, p = 1.0)
    return x,y

def loss(outputs, labels):
    batch_size = outputs.size()[0]            # batch_size
    outputs = torch.log_softmax(outputs, dim=1)   # compute the log of softmax values
    outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
    return -torch.sum(outputs)/batch_size

x, y = load_file()

with torch.no_grad():
    out = model(x)
    print(out)
    print(loss(out, y))
