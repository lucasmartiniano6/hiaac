# Data cleanse
# Excludes 0 activity-Id data and changes any NaN occurence to 0
# Saves everything in data/ folder

import numpy as np
import torch

def clean_file(file_id):
    path = f'data/PAMAP2_Dataset/Protocol/subject{file_id}.dat'
    data = torch.from_numpy(np.genfromtxt(path))

    new = []
    for i in range(len(data)):
        if(data[i][1] == 0):
            continue
        if torch.any(torch.isnan(data[i])): #TODO: optimize look up for NaN
            for nan in np.where(torch.isnan(data[i])):
                data[i][nan] = 0
        new.append(data[i])

    new = np.stack(new)
    np.savetxt(f'data/{file_id}.txt', new, delimiter=' ', newline='\n')

if __name__ == '__main__':
    from tqdm import tqdm
    for i in tqdm(range(101,102)):
        clean_file(str(i))