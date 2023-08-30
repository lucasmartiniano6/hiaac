import torch
from model import create_strategy
from avalanche.benchmarks.classic import SplitCIFAR100

class Trainer():
    """
    Handle training and testing processes.
    """

    def __init__(self, args):
        self.args = args
        self.train()

    def train(self):
        cl_strategy = create_strategy(self.args)

        print("Creating benchmark...")
        benchmark = SplitCIFAR100(n_experiences=self.args.n_exp, shuffle=True, seed=42)
        torch.manual_seed(42)

        print("Strategy: " + self.args.strat)
        for experience in benchmark.train_stream:
            print("EXPERIENCE: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)
#            print('Previous Classes: ', experience.previous_classes)
#            print('Future Classes: ', experience.future_classes)

            cl_strategy.train(experience)
            # eval only in previously trained classes
            cl_strategy.eval(benchmark.test_stream, experience.current_experience)

        torch.save(cl_strategy.model.state_dict(), 'pth/saved_model.pth')
        cl_strategy.writer.close()
