import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import yt



ds= yt.load('snap200/id0/cloud.0200.vtk')


ds = yt.load('snap200/id0/cloud.0200.vtk')
ds.field_list # Get the list of native fields
ds.derived_field_list 

#ds.field_list → Yerel alanlar (diskte saklananlar).
#ds.derived_field_list → Tüm türetilmiş alanlar (diğer alanlardan hesaplananlar; 
#       üst küme kapsamında yerel alanları da içerebilir).

def hist_function(ds, field, bins=30):
    ad= ds.all_data()
    field_data=ad["athena", field] #field_data benim ad'dan çektiğim field değerleri
    plt.hist(field_data, bins=bins)
    plt.xlabel(field)
    plt.ylabel('frequency')
    plt.title(f'{field} Histogram')
    plt.show()

hist_function(ds, "density", bins=30)

