wget -N -P data/ https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip 
unzip data/PAMAP2_Dataset.zip -d data/
python3 data/clean.py