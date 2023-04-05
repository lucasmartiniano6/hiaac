from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.plugins.checkpoint import CheckpointPlugin, FileSystemCheckpointStorage
import torch
from model import create_strategy
from benchmark import make_benchmark

class Trainer():
    """
    Handle training and testing processes.
    """

    def __init__(self, args):
        self.args = args
        self.train()

    def train(self):
        RNGManager.set_random_seeds(42)
        check_plugin = CheckpointPlugin(
        FileSystemCheckpointStorage(
            directory='./checkpoints/'
            ),
            map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        )
        cl_strategy, initial_exp = check_plugin.load_checkpoint_if_exists()    

        if cl_strategy is None:
            cl_strategy = create_strategy(self.args)

        print("Creating benchmark...")
        benchmark = make_benchmark(self.args)
        results = []
        print("Strategy: " + self.args.strat)
        for experience in benchmark.train_stream:
            print("EXPERIENCE: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)
            print('Previous Classes: ', experience.previous_classes)
            print('Future Classes: ', experience.future_classes)
            print('Classes seem so far: ', experience.classes_seen_so_far)

            cl_strategy.train(experience)
            results.append(cl_strategy.eval(benchmark.test_stream))

        torch.save(cl_strategy.model.state_dict(), 'pth/saved_model.pth')
