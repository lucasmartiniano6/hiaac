import argparse
import time
import train

parser = argparse.ArgumentParser()

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--total_classes', default=25, type=int,
                    help='12 for pamap, 12 for hapt, 19 for dsads, 15 for milan, ')
parser.add_argument('--new_classes', default=2, type=int, help='number of new classes per incremental batch')
parser.add_argument('--base_classes', default=2, type=int, help='number of classes in first batch')
parser.add_argument('--epochs', default=1, type=int, help='number of training epochs: 200 for dsads/pamap')
parser.add_argument('--n_exp', default=1, type=int, help='number of experiences in the benchmark. Has to be divisible by the number of classes')

args = parser.parse_args()

def main():
    start_time = time.time()
    train.Trainer(args)
    print(f'\nTotal time: {time.time() - start_time:.3f}s')

if __name__ == '__main__':
    main()