import argparse
import time
import train

parser = argparse.ArgumentParser()

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--input_size', default=51, type=int, help="number of features of each data point")
parser.add_argument('--total_classes', default=25, type=int,
                    help='12 for pamap, 12 for hapt, 19 for dsads, 15 for milan, ')
parser.add_argument('--new_classes', default=2, type=int, help='number of new classes per incremental batch')
parser.add_argument('--base_classes', default=12, type=int, help='number of classes in first batch')
parser.add_argument('--epochs', default=1, type=int, help='number of training epochs: 200 for dsads/pamap')
parser.add_argument('--n_exp', default=1, type=int, help='number of experiences in the benchmark. Has to be divisor of the number of classes')
parser.add_argument('--train_mb_size', default=2, type=int, help="train minibatch size")
parser.add_argument('--eval_mb_size', default=2, type=int, help="eval minibatch size")
parser.add_argument('--mem_size', default=5, type=int, help="memory size for ilos replay buffer")

args = parser.parse_args()

def main():
    start_time = time.time()
    train.Trainer(args)
    print(f'\nTotal time: {time.time() - start_time:.3f}s')

if __name__ == '__main__':
    main()