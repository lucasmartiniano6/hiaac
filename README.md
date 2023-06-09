# H.IAAC Unicamp

Repo for Continual Learning in HAR data.

Usage
-----
First, set up the data folder:
```
./setup.sh
```
Then, run the experiment with the desired arguments:
```
./runner.sh
```

Implementation
-----
ILOS strategy is implemented with memory replay and a modified cross-distillation loss with accommodation ratio. The exemplar set is constructed through herding selection as in the original paper, check ```ilos.py``` for how. We also use a ResNet adapted to time-series, check ```resnet.py``` for how. The hyperparameters for the experiment are loaded from ```config.json```.

More information:
* https://arxiv.org/abs/2003.13191
* https://arxiv.org/abs/1611.06455
* https://arxiv.org/abs/2104.09396
* https://avalanche-api.continualai.org/en/v0.3.1/index.html
* https://github.com/srvCodes/continual-learning-benchmark
