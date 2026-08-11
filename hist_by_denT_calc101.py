import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yt
import os
import glob

yt.funcs.mylog.setLevel(50) #Bilgisayar programları, arkada çalışırken ne yaptığını sana anlatmak için sürekli kendilerine bir "günlük" (log) tutar.
#Düşük derece (Örn: 10 veya 20): En ufak ayrıntıyı bile yazar (Dedikodu gibi, her şeyi anlatır).
#En yüksek derece (50): Yalnızca çok büyük krizleri ve felaketleri yazar.

SIMULATIONS = {
    "sim_3": "/scratch/hpc-prf-radmix/hpcbeoe/sim_3",
    "sim_4": "/scratch/hpc-prf-radmix/hpcbeoe/sim_4",
    "sim_9": "/scratch/hpc-prf-radmix/hpcbeoe/sim_9"

}

DENSITY_CUT: 600.0
TEMP_CUT_CODE: 2.0 #Real temperature < 20,000 Kelvin'in karşılığı

#Fonksiyon tanımlarken parantez içine yazdığımız o kelimelere parametre (değişken) denir. Bunlara karar vermek aslında çok mantıklı ve basit bir kurala dayanır:
#Altın Kural: "Bu fonksiyon her çalıştığında dışarıdan değişebilecek / faklılaşabilecek bilgi ne?"
def extract_filtered_cell_data(sim_name, sim_path):
    pattern = os.path.join(sim_path, "id0", "cloud*.vtk") #Aradığımız dosyanın adres tarifini çıkarıyor.
    #Günlük hayat benzetmesi: Bir görevliye "Bana 'Bulut' etiketi taşıyan tüm mavi klasörleri bulacağız, adres şablonumuz bu" demek gibidir.
    vtk_files= sorted(glob.glob(pattern)) #glob.glob(...) kısmı: Şablona uyan ne kadar dosya varsa (örneğin cloud001.vtk, cloud002.vtk gibi) hepsini bilgisayarda aratıp tek bir torbaya toplar.
    #Günlük hayat benzetmesi: Görevlinin raftan bulduğu tüm "Bulut" klasörlerini masaya getirip 1, 2, 3... diye numarasına göre dizmesidir.


    if not vtk_files:
        print(f"{sim_name} VTK file couldnt be find")
        return
    #Fonksiyon arkada o işi kafa yorup yapar. İşte return komutu tam o anda devreye girer:
    #İşi Bitirir: Fonksiyona "Artık görevin bitti, çalışmayı durdur" der.
    #Sonucu Geri Verir: Elde ettiği cevabı veya ürünü, kendisini çağıran kişiye (yani sana veya kodun devamına) teslim eder.

    filtered_densities =[]
    filtered_temperatures = []

    print(f"\n==========================================")
    print(f"🚀 Extracting Histogram Data: {sim_name}")
    print(f"==========================================")

    #Şimdi bütün snapshotlardaki filtreye uyan hücrelerin değelerini toplayalım 
    for vtk_file in vtk_files:
        snap_name= os.path.basename(vtk_file) #Ne yapıyor? Bilgisayardaki dosya adresleri çok uzundur 
        #(Örn: C:/Kullanicilar/Masaustu/Projeler/Simulasyon/id0/cloud001.vtk). Bu komut, baştaki o uzun klasör 
        # yollarını budar ve sana sadece dosyanın adını (cloud001.vtk) verir.
        print(f"--> Reading: {snap_name}", flush=True) 
        #Ekrana --> Reading: cloud001.vtk şeklinde bir mesaj yazar. Böylece sen arkada kodun durmadığını, 
        # hangi dosyayla uğraştığını canlı canlı görürsün.
        #flush=True kısmı: Bilgisayara "Bu mesajı hafızada bekletme, hemen anında ekrana bas!" der. 
        # Bazen bilgisayarlar mesajları biriktirip toplu gösterir; bu komut mesajın gecikmeden, 
        # o saniyede ekranda görünmesini sağlar

        ds = yt.load(vtk_file)
        ad= ds.all_data()

        temp_yt= np.array(ad["temperature"], dtype= np.float64)
        real_temp= temp_yt/ 6.97436478913788e-09
        dens_yt= np.array(ad[("athena", "density")], dtype= np.float64)
        #------------------------------- burda kaldım----------------------------------------
        
        











        



