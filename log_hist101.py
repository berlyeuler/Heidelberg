import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt

ds= yt.load('/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/cloud.0040.vtk')
ds.field_list # Get the list of native fields

ds.derived_field_list
def hist_function(ds, field, bins=100):
    ad = ds.all_data()
    field_data = ad["athena", field] 

    plt.figure(figsize=(7,5))
    plt.hist(field_data, bins=bins)
    plt.yscale('log')

    plt.xlabel(field)
    plt.ylabel('frequency')
    plt.title(f'{field} Histogram')

    if field == "density":
        output_path = '/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/density_histogram_0040.png'
    elif field == "pressure":
        output_path = '/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/pressure_histogram_0040.png'

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() # Hafıza (memory) sorunları yaşamamak için figürü kapatıyoruz
    print("successfully saved: ", output_path)

hist_function(ds, "density", bins=100)
hist_function(ds, "pressure", bins=100)

