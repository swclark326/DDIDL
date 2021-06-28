#import torch
#import numpy as np
import os
import pandas as pd

os.makedirs(os.path.join('C:/Users/shane.clark/OneDrive - West Point/Documents/Python Learning/Deep Dive into Deep Learning','data'), exist_ok=True)
data_file=os.path.join('C:/Users/shane.clark/OneDrive - West Point/Documents/Python Learning/Deep Dive into Deep Learning', "house_tiny.csv")
with open(data_file, "w") as f:
    f.write('NumRooms, Alley, Price\n')
    f.write('NA, Pave, 127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

