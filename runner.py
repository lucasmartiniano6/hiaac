import argparse
import time
import json
import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')

    with open(parser.parse_args().config, 'r', encoding='utf=8') as f:
        config = json.loads(f.read())

    for key in config:
        parser.add_argument('-' + key, default=config[key])

    args = parser.parse_args()

    start_time = time.time()
    train.Trainer(args)
    print(f'\nTotal time: {time.time() - start_time:.3f}s')

if __name__ == '__main__':
    main()