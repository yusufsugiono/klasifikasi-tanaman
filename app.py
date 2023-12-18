# Import library yang dibutuhkan
import streamlit as st
import pickle
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# Untuk membuat judul halaman web dan juga iconnya
st.set_page_config(
    page_title="Sistem Klasifikasi Tanaman",
    page_icon="🌾"
)

# Memberikan Title / Heading webnya
st.title("Sistem Klasifikasi Tanaman")
st.write("Halo, selamat datang! Bagaimana kabarmu hari ini? 😄")

dataset = pd.read_csv('dataset/new_crop_recommendation.csv')

encoder = LabelEncoder()
encoder.fit_transform(dataset['label'])


# Membuat form input nilai yang akan digunakan sebagai input klasifikasi
with st.form('input_data'):
    nitrogen = st.number_input('Kadar Nitrogen (N)')
    fosfor = st.number_input('Kadar Fosfor (P)')
    kalium = st.number_input('Kadar Kalium (K)')
    temperatur = st.number_input('Temperatur')
    kelembaban = st.number_input('Kelembaban')
    pH = st.number_input('Tingkat Keasaman (pH)')
    curah_hujan = st.number_input('Curah Hujan')
    submit = st.form_submit_button('Hasil Klasifikasi')

# Jika sudah klik tombol submit, maka lakukan prediksi
if submit:

    # Scaling data input
    data = [[nitrogen, fosfor, kalium, temperatur, kelembaban, pH, curah_hujan]]

    # Load model yang telah ditraining di Google Colab
    pickled_model = pickle.load(open('model/best_rf_model.pkl', 'rb'))

    # Klasifikasikan data nilai tersebut
    classification_result = pickled_model.predict(data)[0]
    # Menampilkan hasil klasifikasi
    st.write("Tanaman yang mungkin cocok ditanam:")
    # result_string = f"Hasil Klasifikasi : **{encoder.classes_[classification_result].title()}**"
    result_string = f"## {encoder.classes_[classification_result].title()}"
    st.markdown(result_string)
