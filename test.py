import torch
from slimrestnet import SlimResNet34
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.storage_policy import ParametricBuffer, RandomExemplarsSelectionStrategy
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader, GroupBalancedDataLoader
from avalanche.benchmarks.utils import AvalancheDataset
import matplotlib.pyplot as plt
import itertools
from types import SimpleNamespace
 
def batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch
 
def main():
    model = SlimResNet34(100)
#    model.load_state_dict(torch.load('pth/saved_model.pth'))

    benchmark = SplitCIFAR100(n_experiences=20, seed=42)

    storage_p = ParametricBuffer(
        max_size=100,
        groupby='class',
        selection_strategy=RandomExemplarsSelectionStrategy()
    )

    for batch, letter in zip(batched(benchmark.test_stream[0].dataset, 8), itertools.cycle(["offline", "online"])):
        inputs, labels, _ = zip(*batch)
        inputs = torch.stack(inputs)
        labels = torch.tensor(labels)

        exp = SimpleNamespace(dataset=torch.utils.data.dataset.TensorDataset(inputs, labels))
        exp.dataset.targets = labels.tolist()

        strategy_state = SimpleNamespace(experience=exp)
        storage_p.update(strategy_state)
        #print(f"Max buffer size: {storage_p.max_size}, current size: {len(storage_p.buffer)}")
        print("Buffer: ", *sorted(storage_p.buffer.targets))
        print("Curr MB: ", *sorted(labels.tolist()))

        curr = torch.utils.data.TensorDataset(inputs, labels)
        curr = AvalancheDataset(curr)
        curr.targets = labels.tolist()

        dl = ReplayDataLoader(curr, storage_p.buffer, batch_size=8, oversample_small_tasks=True)
        print("Batches of REPLAY DATALOADER: ")
        for batch in batched(dl.data, 8):
            inputs, labels = zip(*batch)
            inputs = torch.stack(inputs)
            labels = torch.tensor(labels)
            print(*sorted(labels.tolist())) 
            print(inputs.shape)
        print("\n------")

if __name__ == '__main__':
    main()