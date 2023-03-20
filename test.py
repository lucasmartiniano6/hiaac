import torch
from avalanche.models import SimpleMLP
from numpy import genfromtxt

model = SimpleMLP(num_classes=25, input_size=52, hidden_size=512, hidden_layers=2)
model.load_state_dict(torch.load('saved_model.pth'))
model.eval()


#path = 'PAMAP2_Dataset/Protocol/subject108.dat'
path = 'a.txt'
data = torch.from_numpy(genfromtxt(path))

x = [data[i][2:] for i in range(len(data))]
y = [data[i][1] for i in range(len(data))]

y = []
for i in range(len(data)):
    number = int(data[i][1])
    target = torch.zeros(25)
    target[number] = 1
    y.append(target)

x = torch.stack(x).to(torch.float32)
y = torch.stack(y).to(torch.int32)

print(y)
with torch.no_grad():
    out = model(x)
