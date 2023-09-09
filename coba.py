import streamlit as st
import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import randint
# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image



data = pd.read_csv('Data5.csv')
data.head()
data.shape
data['KelayakanHalal'].value_counts()
X = data.drop (columns='KelayakanHalal',axis=1)
Y = data['KelayakanHalal']
print(X)
print(Y)
scaler = StandardScaler()
scaler.fit(X)
standarized__data = scaler.transform(X)
print(standarized__data)
X = standarized__data
Y = data['KelayakanHalal']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, stratify=Y, random_state=42)
print(X.shape, X_train.shape, X_test.shape)
X = data.iloc[:, :-1]  # Features
Y = data.iloc[:, -1]   # Labels
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, Y_train)
X_train_prediction = nb_classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy training:", training_data_accuracy)
X = data.iloc[:, :-1]  # Features
Y = data.iloc[:, -1]   # Labels
nb_classifier = GaussianNB()
nb_classifier.fit(X_test, Y_test)
X_test_prediction = nb_classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Akurasi testing:', test_data_accuracy)

# Title
st.title('Prediksi Kecepatan Penerbitan Sertifikat Halal')
session_state = st.session_state

def Analisa():
    st.header('Prediksi Kecepatan Penerbitan Sertifikat Halal')
# Elemen antarmuka pengguna untuk input
Bahan1 = st.number_input('Masukkan Bahan1', min_value=0.0, max_value=2.0, value=0.0, step=0.0)

Bahan2 = st.number_input('Masukkan Bahan2', min_value=0.0, max_value=2.0, value=0.0, step=0.0)

Bahan3 = st.number_input('Masukkan Bahan3', min_value=0.0, max_value=2.0, value=0.0, step=0.0)

Bahan4 = st.number_input('Masukkan Bahan4', min_value=0.0, max_value=2.0, value=0.0, step=0.0)

Bahan5 = st.number_input('Masukkan Bahan5', min_value=0.0, max_value=2.0, value=0.0, step=0.0)

Bahan6 = st.number_input('Masukkan Bahan6', min_value=0.0, max_value=2.0, value=0.0, step=0.0)

Bahan7 = st.number_input('Masukkan Bahan7', min_value=0.0, max_value=2.0, value=0.0, step=0.0)

Bahan8 = st.number_input('Masukkan Bahan8', min_value=0.0, max_value=2.0, value=0.0, step=0.0)

Bahan9 = st.number_input('Masukkan Bahan9', min_value=0.0, max_value=2.0, value=0.0, step=0.0)

# Tombol untuk melakukan prediksi
if st.button('Prediksi'):
        input_data = np.array([Bahan1, Bahan2, Bahan3, Bahan4, Bahan5, Bahan6, Bahan7, Bahan8, Bahan9])
        session_state.input_data_beranda = input_data  # Store the input data in session state
        # Melakukan reshape data input
        input_data_as_numpy_array = input_data.reshape(1, -1)
    
        # Melakukan transformasi dengan scaler
        std_data = scaler.transform(input_data_as_numpy_array)
    
        # Melakukan prediksi dengan model yang telah dilatih
        prediction = nb_classifier.predict(std_data)

        if 2 in input_data:
            hasil_prediksi = 'Tidak Layak'
        else:
            prediction = nb_classifier.predict(std_data)
            hasil_prediksi = 'Layak'

        st.write('Hasil Prediksi:', hasil_prediksi)

# Fungsi untuk halaman Tentang
def Plot():
        st.header('Pie Chart Hasil Input')

        if hasattr(session_state, 'input_data_beranda'):

            st.subheader('Grafik Berdasarkan Inputan Beranda')
labels = 'Bahan1', 'Bahan2', 'Bahan3', 'Bahan4', 'Bahan5', 'Bahan6', 'Bahan7', 'Bahan8', 'Bahan9'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)
# Fungsi untuk halaman Kontak
def File():
    st.header('Download File Analisa Dan File Data')
    file_path = "C:/Users/DINAR/Documents/project/Data5.csv"
    st.write('download file Excel')
    if st.button("Download"):
        with open(file_path, "rb") as f:
            file_data = f.read()
        st.markdown(f'<a onclick="data:application/octet-stream;base64,{base64.b64encode(file_data).decode()}" download="Data5.csv"><button>Click here to download EXCEL</button></a>', unsafe_allow_html=True)
        file_path = "C:/Users/DINAR/Documents/project/NAIVE BAYES.ipynb"
    st.write('download file Ipynb')
    if st.button("DownLoad"):
        with open(file_path, "rb") as f:
            file_data = f.read()
        st.markdown(f'<a onclick="data:application/octet-stream;base64,{base64.b64encode(file_data).decode()}" download="NAIVE BAYES.ipynb"><button>Click here to download .IPYNB</button></a>', unsafe_allow_html=True)
selected_menu = st.sidebar.selectbox('Pilih Halaman:', ['Analisa', 'Plot', 'File'])

if selected_menu == 'Analisa':
    Analisa()
elif selected_menu == 'Plot':
    Plot()
elif selected_menu == 'File':
    File()