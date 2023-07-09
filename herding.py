import torch
from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.training import Naive
from slimrestnet import SlimResNet18
from storage_policy import HerdingSelectionStrategy
from benchmark import make_tensors
from avalanche.benchmarks.utils import AvalancheDataset
from torch.utils.data.dataset import TensorDataset
from resnet import ResNetBaseline
from avalanche.benchmarks.classic import SplitCIFAR100
 
def main():
    benchmark = SplitCIFAR100(n_experiences=20, shuffle=True, seed=42)

    for exp in benchmark.train_stream:
        print(exp.current_experience, len(benchmark.test_stream[:exp.current_experience+1]))

if __name__ == '__main__':
    main()