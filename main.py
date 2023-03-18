import torch
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins.checkpoint import CheckpointPlugin, FileSystemCheckpointStorage
from avalanche.benchmarks.generators import tensors_benchmark
from numpy import genfromtxt
from model import create_strategy, device


def make_tensors(n_exp, file_id):
    data = torch.from_numpy(genfromtxt(f'clean/{file_id}.txt'))
    
    x = [data[i][1:] for i in range(len(data))]
    y = [data[i][0] for i in range(len(data))]

    x = torch.stack(x).to(torch.float32)
    y = torch.stack(y).to(torch.int32)

    sz = x.size()[0] - (x.size()[0] % n_exp) # closest multiple of n_exp
    x = x[:int(sz)]
    y = y[:int(sz)]

    n_samples = int(x.size()[0] / n_exp)
    train_tensor = []
    for i in range(0, x.size()[0], n_samples):
        train_tensor.append((x[i:i+n_samples], y[i:i+n_samples]))
    return train_tensor

def make_benchmark():
    n_exp = 20
    benchmark = tensors_benchmark(
        train_tensors=make_tensors(n_exp, file_id='101')+make_tensors(n_exp, file_id='102'),
        test_tensors=make_tensors(n_exp, file_id='109'),
        task_labels=[0 for _ in range(2*n_exp)],  # for each train exp
    )
    return benchmark

def main():
    RNGManager.set_random_seeds(42)
    check_plugin = CheckpointPlugin(
        FileSystemCheckpointStorage(
            directory='./checkpoints/'
        ),
        map_location=device
    )
    cl_strategy, initial_exp = check_plugin.load_checkpoint_if_exists()    

    if cl_strategy is None:
        cl_strategy = create_strategy(check_plugin)
    
    benchmark = make_benchmark()

    results = []
    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience)
        results.append(cl_strategy.eval(benchmark.test_stream))

    print(results)

if __name__ == '__main__':
    make_tensors(1, '101')
    #main()