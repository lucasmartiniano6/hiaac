import torch
from slimrestnet import SlimResNet34
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.storage_policy import ParametricBuffer, RandomExemplarsSelectionStrategy
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader, GroupBalancedDataLoader
from avalanche.benchmarks.utils import AvalancheDataset
import matplotlib.pyplot as plt
import itertools
from types import SimpleNamespace
import random
 
def batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch
 
def scratch():
    benchmark = SplitCIFAR100(n_experiences=20, seed=42)

    exemplar = []
    q = 8 # exemplars per class

    for exp in benchmark.train_stream:
        print('exp', exp.current_experience)
        if exp.current_experience == 0:
            for batch in batched(exp.dataset, 8):
                    inputs, labels, *_ = zip(*batch)
                    inputs = torch.stack(inputs) # 8, 3, 32, 32
                    labels = torch.tensor(labels) # 8 
                    # TRAIN
            print(*set(exp.dataset.targets))
            # update exemplar set herding selection
            # examplar_idx = herding.make_sorted_indices(exp.dataset)
            # for now lets put random samples
            exemplar_idx = random.sample(range(len(exp.dataset)), q)
            for i in exemplar_idx:
                exemplar.append(exp.dataset[i])
        elif exp.current_experience > 0:
            # two-step learning
            for batch_red, batch_black in zip(batched(exp.dataset, 8), batched(itertools.cycle(exemplar), 8)):
                # Red dot - Modified Cross-Distillation Loss
                inputs_red, labels_red, *_ = zip(*batch_red)
                inputs_red = torch.stack(inputs_red)
                labels_red = torch.tensor(labels_red)
                # print('red: ', inputs_red.shape, labels_red.shape)

                # Black dot - Cross-Entropy Loss
                inputs_black, labels_black, *_ = zip(*batch_black)
                inputs_black = torch.stack(inputs_black)
                labels_black = torch.tensor(labels_black)
                # print('black: ', inputs_black.shape, labels_black.shape)
            
                inputs_balanced = torch.cat((inputs_red, inputs_black), 0)
                labels_balanced = torch.cat((labels_red, labels_black), 0)
                # print('red+black: ', inputs_balanced.shape, labels_balanced.shape)
            # update exemplar set with NCM
            # for now lets put random samples
            exemplar_idx = random.sample(range(len(exp.dataset)), q)
            for i in exemplar_idx:
                exemplar.append(exp.dataset[i])
 

def test_replay():
    benchmark = SplitCIFAR100(n_experiences=20, seed=42)

    exemplar_set = ParametricBuffer(
        max_size=1000,
        groupby='class',
        selection_strategy=RandomExemplarsSelectionStrategy()
    )
    exemplar_set.update(SimpleNamespace(experience=benchmark.train_stream[0]))
 
    experience = benchmark.train_stream[1]

    datas = [v.buffer for v in exemplar_set.buffer_groups.values()]
    datas.append(experience.dataset)
    dataLoader = GroupBalancedDataLoader(
        datas,
        batch_size=100, #CHANGE THIS LATER
        oversample_small_groups=False
    )

    targets = {}
    i = 0
    for batch in dataLoader:
        x, y, *_ = batch
        
        for t in y.tolist():
            if t not in targets.keys():
                targets[t] = 1
            else:
                targets[t] += 1
        i += 1
    targets = {k: v for k, v in sorted(targets.items(), key=lambda item: item[1])} 
    samples = 0
    for i in targets:
        print(i, targets[i])
        samples += targets[i]
    print('samples', samples)
    exit(0)
    # --------------------
    print('Exemplar set: ', *set(sorted(exemplar_set.buffer.targets)))
    print('Current exp: ', *set(sorted(experience.dataset.targets)))
    dataLoader = ReplayDataLoader( # slow
        experience.dataset,
        exemplar_set.buffer,
        batch_size=8,
        oversample_small_tasks=True,
    ) # TODO: BUFFER NO BALANCING WITH PREVIOUS CLASSES data = dataLoader.data
    print('Replay data: ', *set(sorted(dataLoader.data.targets)))
    data = dataLoader.data
    for batch in batched(data, 8):
        inputs, labels, *_ = zip(*batch)
        # print(*sorted(labels)) 

 
def test_model():
    model = SlimResNet34(100)
    model.load_state_dict(torch.load('pth/saved_model.pth'))

    benchmark = SplitCIFAR100(n_experiences=20, seed=42)

    total, correct = 0, 0
    pred, ground = set(), set()
    for batch in batched(benchmark.train_stream[0].dataset, 8):
        inputs, labels, _ = zip(*batch)
        inputs = torch.stack(inputs)
        labels = torch.tensor(labels)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pred.update(predicted.tolist())
        ground.update(labels.tolist())
      
    print("Predicted: ", pred) 
    print("Ground in this experience: ", ground)
    print("Accuracy: ", correct/total)
 
def main():
    model = SlimResNet34(100)
    model.load_state_dict(torch.load('pth/saved_model.pth'))

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


def loss_fn(outputs, labels):
    seen = [0,1,2,3,4]
    labels = torch.tensor([5,6,7,8,9])


    seen = torch.tensor(seen)
    both = torch.cat((seen, labels), 0)
    print(both)
    p = outputs[:, both]
    print(outputs[1])
    print(p)

if __name__ == '__main__':
    scratch()