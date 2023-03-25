# H.IAAC Unicamp

Repo for Continual Learning in HAR data

Usage
-----
First, download PAMAP2 Dataset and set up the clean/ folder with working data:
```
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip && unzip PAMAP2_Dataset.zip
mkdir clean && python3 clean.py
```
Then, run the experiment with the desired arguments (check runner.py for more):
```
python3 runner.py --epochs 1 --lr 0.001
```
