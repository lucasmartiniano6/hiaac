import torch
from benchmark import make_tensors

torch_data, targets = make_tensors('a')
x = torch_data[:2][0]
y = torch_data[:2][1]

x = x.reshape(x.shape + (1,))

layers = torch.nn.Sequential(
    torch.nn.Conv1d(51, 10, 1, padding='same'),
    torch.nn.BatchNorm1d(num_features=10),
    torch.nn.ReLU(),

    torch.nn.Conv1d(10, 5, 1, padding='same'),
    torch.nn.BatchNorm1d(num_features=5),
    torch.nn.ReLU(),

    torch.nn.Conv1d(5, 1, 1, padding='same'),
    torch.nn.BatchNorm1d(num_features=1),
    torch.nn.ReLU(),

#    torch.nn.AvgPool2d(kernel_size=1),
    torch.nn.Linear(1, 25),
    torch.nn.Softmax(dim=-1)
)

num_classes = 25
out = layers(x).reshape(x.shape[0],num_classes)
print(out)