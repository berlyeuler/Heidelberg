import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import yt

ds= yt.load('/scratch/hpc-prf-radmix/hpcbeoe/Mach4_test/id0/cloud.0100.vtk')

ds.derived_field_list

def predicted_histogram(ds, field, bins=30):
    M1 = 8.9887
    gamma = 5 / 3 
    rho1 = 1.0
    T = 10.0

    ad=ds.all_data()
    if field == "density":

        densty=((gamma+1)*(M1**2)) / (2 + (gamma-1) * (M1**2))
        predicted_value= densty* rho1
        field_data= ad[("athena", "density")]
        label_name= f"Predicted Density( {predicted_value:.2f})"
        #.2f → virgülden sonra 2 basamaklı sabit noktalı sayı olarak biçimlendir


    elif field == "pressure":
        press= 1 + (2 * gamma / (gamma + 1)) * (M1**2 - 1)

        press_first= rho1 * T # Başlangıç basıncı hesabı (P1 = rho1 * T)
        predicted_value= press_first * press # Teorik tahmini basınç değeri
        field_data = ad[("athena", "pressure")]
        label_name= f"Predicted Pressure({predicted_value:.2f})"

    elif field == "temperature":
        densty=((gamma+1)*(M1**2)) / (2 + (gamma-1) * (M1**2))
        press= 1 + (2 * gamma / (gamma + 1)) * (M1**2 - 1)

        # Sıcaklık oranı = Basınç Oranı / Yoğunluk Oranı
        T_ratio= press/ densty
        predicted_value= T * T_ratio
        # Hazır veri olmadığı için simülasyonun basınç verisini yoğunluk verisine bölüyoruz:
        field_data = ad["athena", "pressure"]
    
    else:
        print("choose density, pressure or temperature")
        return
    
    #şimdi verileri topladık isimlendirdik gelelim grafik çizmeye

    #gerçek simülasyon verilerinin grafiği
    plt.hist(field_data, bins=bins, alpha= 0.7, color= "pink", label= "Simulation Data")

    #Teorik tahmini gösteren kesikli çizgi
    plt.axvline(x=predicted_value,
                color="crimson",
                linestyle="--",
                linewidth=2.5,
                label=label_name
                )
    
    # Grafiğin süslemeleri
    plt.xlabel(field.capitalize())
    plt.ylabel("Frequency")
    plt.title(f"{field.capitalize()} Histogram & Rankine-Hugoniot Prediction")
    plt.legend()
    plt.show()

predicted_histogram(ds, "density")





