import torch
from slimrestnet import SlimResNet34
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.benchmarks.utils import AvalancheDataset
import matplotlib.pyplot as plt
import itertools
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
            # update herding
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
            exemplar_idx = random.sample(range(len(exp.dataset)), q)
            for i in exemplar_idx:
                exemplar.append(exp.dataset[i])
 
def main():
    pass

if __name__ == '__main__':
    main()