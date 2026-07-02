import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import yt



ds= yt.load('/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/cloud.0040.vtk')
ds.field_list # Get the list of native fields
ds.derived_field_list 

#ds.field_list → Yerel alanlar (diskte saklananlar).
#ds.derived_field_list → Tüm türetilmiş alanlar (diğer alanlardan hesaplananlar; 
#       üst küme kapsamında yerel alanları da içerebilir).

def hist_function(ds, field, bins=30):
    ad = ds.all_data()
    field_data = ad["athena", field] 
    
    plt.hist(field_data, bins=bins)
    plt.xlabel(field)
    plt.ylabel('frequency')
    plt.title(f'{field} Histogram')
    
    # plt.show() yerine bunu yazıyoruz:
    output_path = '/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/density_histogram_0040.png'
    plt.savefig(output_path, dpi=300)
    print(f"Grafik başarıyla kaydedildi: {output_path}")

hist_function(ds, "density", bins=30)

hist_function(ds, "density", bins=30)

