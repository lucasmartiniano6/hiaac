import torch
from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.training import Naive
from slimrestnet import SlimResNet18
from storage_policy import HerdingSelectionStrategy

def main():
    benchmark = SplitCIFAR10(n_experiences=10, shuffle=True, seed=42)

    adapted_dataset = benchmark.train_stream[0].dataset
    sub = adapted_dataset.subset(range(5)) # q-herding

    model = SlimResNet18(10)
    herding = HerdingSelectionStrategy(model, 'feature_extractor')
    strategy = Naive(model, optimizer=torch.optim.SGD(model.parameters(), 0.1), criterion=torch.nn.CrossEntropyLoss())
    examplar_idx = herding.make_sorted_indices(strategy, sub)
    print(examplar_idx)
  
    # x = [5, 3, 32, 32]
    # out = model(x) # [5, 160, 4, 4]
    # indices = herding(out)  # [0]


if __name__ == '__main__':
    main()