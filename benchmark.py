import torch
from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.generators import nc_benchmark
from numpy import genfromtxt


def make_tensors(file_id):
    path = f'data/{file_id}.txt'

    data = torch.from_numpy(genfromtxt(path))
    
    x = [data[i][3:] for i in range(len(data))]
    y = [data[i][1] for i in range(len(data))]

    x = torch.stack(x).to(torch.float32)
    y = torch.stack(y).to(torch.long)

    x = torch.nn.functional.normalize(x, p = 1.0)
    
    x = x.reshape(x.shape + (1,)) # reshape for ResNet (conv1d)

    torch_data = torch.utils.data.dataset.TensorDataset(x,y)
    return torch_data, y

def make_benchmark(args=None):
    n_exp = args.n_exp if args is not None else 1

    train_data, targets = make_tensors('a') 
    train_data.targets = targets.tolist()
    train_data = make_classification_dataset(train_data)
    
    test_data, targets = make_tensors('a') 
    test_data.targets = targets.tolist()
    test_data = make_classification_dataset(test_data)

    benchmark = nc_benchmark(train_dataset=train_data, test_dataset=test_data,
                             n_experiences=n_exp, task_labels=False, shuffle=False)
    return benchmark
    

if __name__ == '__main__':
    benchmark =  make_benchmark()
    for exp in benchmark.train_stream:
        print(exp.current_experience)
        print(exp.classes_in_this_experience)

