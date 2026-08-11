import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import yt



import glob

import os

import imageio





base_path = "/scratch/hpc-prf-radmix/hpcbeoe/sim_9"

times = []

cold_mass_bg = []

cold_mass_wind = []





def create_plot():

    """

    anlık görüntüleri (snapshot) döngüye alarak soğuk gaz kütlesinin

    zaman içindeki değişimini grafiğe dökmeyi deneyin; ancak bu

    kez her bir anlık görüntüde sıcaklığı yaklaşık 20.000 K'in

    (kod birimleriyle 2,0) altında olan hücrelerin sayısını

    toplayın ve bu değeri zamanın bir fonksiyonu olarak çizin.

   

    """



    """

    Anlık görüntüleri döngüye alarak soğuk gaz kütlesinin (T < 20,000 K)

    zaman içindeki değişimini Background ve Wind olarak iki ayrı çizgi halinde çizer.

    """





    # Çıktı klasörünü hazırlayalım

    output_directory= os.path.join(base_path, "cold_gas_plots")

    os.makedirs(output_directory, exist_ok=True)

    print(f"Output directory created: {output_directory}")

   

    # yt ile tüm vtk dosyalarını sırayla bir seriye alıyoruz

    snap_series= yt.DatasetSeries((f"{base_path}/id0/cloud*.vtk"))

   



   

    # snap_series içindeki her bir snapshot'ı tek tek gezeceğiz

    for snap in snap_series:

        # A) Zamanı alıp sepetimize ekliyoruz

        # snap.current_time bize simülasyon zamanını verir

        sim_time = float(snap.current_time)

        times.append(sim_time)



        # B) Soğuk gaz hücrelerini sayıyoruz

        snap_data = snap.all_data() # B) yt içindeki tüm veriyi (hücreleri) seçiyoruz



        real_temperature = snap_data["temperature"] / 6.97436478913788e-09

        # breakpoint() Debugging için bir breakpoint ekledik, kodu durdurur ve değişkenleri inceleyebilirsiniz.

        # C) Sıcaklığı 2.0'dan küçük olan hücreleri filtreliyoruz

        # yt'de bu filtreleme "grid" veya "all_data" üzerinden böyle yapılır:

       

        cold_cells_filter = real_temperature < 2.0



        #maskeyi yazalım

        density = np.array(snap_data[("athena", "density")], dtype= np.float64)



        # hem soğuk hem de arka plan gazı olanların ( s < 0.5) kütlesini toplayalım

        bg_filter = cold_cells_filter & (density < 600)

        mass_bg= snap_data["gas", "mass"][bg_filter].sum()

        cold_mass_bg.append(float(mass_bg))



        wind_filter = cold_cells_filter & (density >= 600)

        mass_w = snap_data["gas", "mass"][wind_filter].sum()

        cold_mass_wind.append(float(mass_w))



        numb_of_coldcells= int(cold_cells_filter.sum())

        # cold_gas_mass.append(numb_of_coldcells) #boş sepetimize bulduğumuz sonucu ekleyelim

        print(f"TİME: {sim_time:.2f} | BG mass: {mass_bg:.2f} | WIND mass {mass_w:.2f}")

   

    plt.figure(figsize=(8,5))



    plt.plot(times, cold_mass_bg, marker='o', linestyle='-', color='orange', label='Background Gas')

    plt.plot(times, cold_mass_wind, marker='s', linestyle='-', color='deeppink', label='Wind')

    plt.xlabel("TIME")

    plt.ylabel("COLD GAS MASS")

    plt.title("Cold Gas Mass Evolution (Temperature < 20,000 K)")

    plt.grid(True)

    plt.legend(loc="upper right")

   

    plot_save_path = os.path.join(output_directory, "cold_gas_evolution.png")

    plt.savefig(plot_save_path)

    plt.show()

    print(f"Grafik başarıyla kaydedildi: {plot_save_path}")



if __name__ == "__main__":

    create_plot()