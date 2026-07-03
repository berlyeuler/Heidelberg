import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt

import glob
import os 
import imageio
ds= yt.load('/scratch/hpc-prf-radmix/hpcbeoe/test_beril/id0/cloud.0100.vtk')
ad= ds.all_data()

def  arrays_temp_den(ds):
    np.linspace(2.0, 3.0, num=5)