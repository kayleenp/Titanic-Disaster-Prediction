# Titanic-Disaster-Prediction

Angellica 01082170031


Kayleen Priscilia 01082170009


Raysa Gohtami 01082170033



Dalam kasus laporan ini, kami membuat suatu machine learning dengan bahasa python dan menggunakan aplikasi Spyder berbasis Anaconda. 
Program yang sudah kami buat berjudul “Titanic Disaster Prediction” yang mengenai prediksi- prediksi korban dan yang selamat di kapal 
Titanic. 


Machine learning menjaga suatu agar tetap sederhana, sebuah algoritma dikembangkan untuk mencatat perubahan dalam data dan berevolusi dalam desain itu untuk mengakomodasi temuan baru. 
Seperti diterapkan untuk analisis prediktif, fitur ini memiliki dampak luas mulai pada kegiatan yang biasanya dilakukan untuk mengembangkan, menguji, dan memperbaiki algoritma untuk tujuan tertentu. 
Tujuan dari pembuatan program ini adalah untuk mengetahui klasifikasi korban berdasarkan kategori- kategori seperti: umur, jenis kelamin, tipe kelas, dan lain- lain secara otomatis dan kategoris.



- Library yang digunakan dalam program ini adalah sebagai berikut:


   *dash


    dash_core_components as dcc


    dash_html_components as html


    dash_table


    pandas as pd


    math, time, random, datetime 


    matplotlib.pyplot


    seaborn

*



- Metode yang dilakukan pertama adalah import seluruh library yang diperlukan, begitu juga meng-install components yang dibutuhkan pada Anaconda prompt.

Setelah itu, kita meng-import data csv “train” yang nantinya akan data training. Setelah itu, kita membuat fungsi data fitting untuk data- datanya. 

Setelah itu, kita membuat suatu fungsi untuk menerjemah apa yang user upload dan dapat meng-upload multiple file.

Lalu, berdasarkan apa yang user upload, kita menggunakan fungsi parse_contents_prediction untuk proses fitting data-nya ke Decision tree setiap kategori yang ada pada dalam file.

Lalu, data- data yang terpilih setelah di fitting ke Decision tree akan ditampilkan dalam bentuk tabel. Untuk metode yang kedua memakai seaborn, kita mengeluarkan output grafik- grafik klasifikasi kategori.




