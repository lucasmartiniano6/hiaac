import torch
from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.training import Naive
from slimrestnet import SlimResNet18
from storage_policy import HerdingSelectionStrategy
from benchmark import make_tensors
from avalanche.benchmarks.utils import AvalancheDataset
from torch.utils.data.dataset import TensorDataset
from resnet import ResNetBaseline
 
def main():
    benchmark = SplitCIFAR10(n_experiences=10, shuffle=True, seed=42)

    adapted_dataset = benchmark.train_stream[0].dataset
    torch_data = adapted_dataset.subset(range(5)) # q-herding

    model = SlimResNet18(10)
    #dataset = TensorDataset(torch.ones(5, 150, 128, 1), torch.ones(5, dtype=torch.long))
    #dataset = TensorDataset(torch.ones(5, 3, 32, 32), torch.ones(5, dtype=torch.long))

#    model = ResNetBaseline(in_channels=51, num_pred_classes=25)
#    x = torch.ones(3, 51, 1), y = torch.ones(3)
#    torch_data = torch.utils.data.dataset.TensorDataset(x,y)
 
    herding = HerdingSelectionStrategy(model, 'feature_extractor')
    strategy = Naive(model, optimizer=torch.optim.SGD(model.parameters(), 0.1), criterion=torch.nn.CrossEntropyLoss())
    examplar_idx = herding.make_sorted_indices(strategy, torch_data)

if __name__ == '__main__':
    main()