# H.IAAC Unicamp

Repo for Continual Learning in HAR data.

Usage
-----
First, download PAMAP2 Dataset and set up the data folder:
```
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip && unzip PAMAP2_Dataset.zip
mkdir clean && python3 clean.py
```
Then, run the experiment with the desired arguments (check runner.py for more):
```
python3 runner.py --epochs 1 --lr 0.001
```

Implementation
-----
Check for more information:
* https://avalanche-api.continualai.org/en/v0.3.1/index.html
* https://arxiv.org/abs/1611.06455
* https://arxiv.org/abs/2104.09396
* https://github.com/srvCodes/continual-learning-benchmark
