# Import library yang dibutuhkan
import streamlit as st
import openpyxl
import xlrd
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import pickle
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@st.cache_data
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.close()
    processed_data = output.getvalue()
    return processed_data


# Untuk membuat judul halaman web dan juga iconnya
st.set_page_config(
    page_title="Sistem Klasifikasi Tanaman",
    page_icon="🌾",
    layout="wide"
)

# Memberikan Title / Heading webnya
st.title("Sistem Klasifikasi Tanaman")
st.write("Selamat datang pada Sistem Klasifikasi Tanaman Berdasarkan Kondisi Lingkungan 🌾")
st.write("Dengan sistem ini Anda dapat menginputkan data kondisi lingkungan untuk mendapatkan hasil klasifikasi tanaman yang cocok untuk ditanam")


dataset = pd.read_csv('dataset/new_crop_recommendation.csv')

encoder = LabelEncoder()
encoder.fit_transform(dataset['label'])

# Load model yang telah ditraining di Google Colab
pickled_model = pickle.load(open('model/best_rf_model.pkl', 'rb'))
print(pickled_model)

col1, col2 = st.columns((3, 1))


with col1:
    tab1, tab2 = st.tabs(
        ["Klasifikasi Data Tunggal", "Klasifikasi Banyak Data"])

    with tab1:
        st.header("Klasifikasi Data Tunggal")
        # Membuat form input nilai yang akan digunakan sebagai input klasifikasi
        with st.form('input_data'):
            nitrogen = st.number_input('Presentase Kadar Nitrogen (N)')
            fosfor = st.number_input('Presentase Kadar Phospor (P)')
            kalium = st.number_input('Presentase Kadar Kalium (K)')
            temperatur = st.slider(label="Temperatur (&#8451;)",
                                   min_value=0.0, max_value=100.0, step=0.1)
            kelembaban = st.slider(label="Kelembaban (%)",
                                   min_value=0.0, max_value=100.0, step=0.1)
            pH = st.slider(label="Tingkat Keasaman (pH)",
                           min_value=0.0, max_value=14.0, step=0.1)
            curah_hujan = st.number_input('Curah Hujan (mm)')
            submit = st.form_submit_button('Lihat Hasil Klasifikasi')

        # Jika sudah klik tombol submit, maka lakukan klasifikasi
        if submit:

            # Scaling data input
            data = [[nitrogen, fosfor, kalium,
                    temperatur, kelembaban, pH, curah_hujan]]

            # Klasifikasikan data nilai tersebut
            classification_result = pickled_model.predict(data)[0]
            # Menampilkan hasil klasifikasi
            st.write("Tanaman yang mungkin cocok ditanam:")
            result_string = f"## {encoder.classes_[classification_result].title()}"
            st.markdown(result_string)

    with tab2:
        st.header("Klasifikasi Banyak Data")
        st.write("Untuk mengklasifikasikan data dalam jumlah yang banyak secara sekaligus, maka Anda dapat menggunakan template di bawah ini")
        with open("template/template.xlsx", "rb") as file:
            btn = st.download_button(
                label="Download template",
                data=file,
                file_name="template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        st.write(
            "Setelah itu isi data sesuai format dan petunjuk yang telah disediakan, kemudian upload dengan drag-and-drop atau klik tombol di bawah ini")

        uploaded_file = st.file_uploader("Upload Data", type=['xlsx', 'xls'])

        if uploaded_file is not None:

            dataframe = pd.read_excel(uploaded_file)

            hasil_klasifikasi = pickled_model.predict(dataframe)

            dataframe['hasil klasifikasi'] = encoder.inverse_transform(
                hasil_klasifikasi)
            st.write(dataframe)
            st.write(
                "Anda dapat menyimpan hasil klasifikasi dengan menekan tombol di bawah ini.")
            df_xlsx = to_excel(dataframe)
            st.download_button(label='Simpan Hasil Klasifikasi',
                               data=df_xlsx,
                               file_name='hasil_klasifikasi.xlsx')

with col2:
    st.success("""**INFORMASI**
            
Sistem klasifikasi ini menggunakan 7 fitur sebagai data input, diantaranya:
1. Kadar Nitrogen dalam tanah         
2. Kadar Phospor dalam tanah
3. Kadar Kalium dalam tanah
4. Temperatur (Celcius)
5. Presentase kelembaban
6. Tingkat keasaman tanah (pH)
7. Curah hujan (mm)

Sistem ini menggunakan model machine learning yang dibangun dengan
metode Random Forest, sehingga dapat menghasilkan klasifikasi
tanaman yang cocok ditanam berdasarkan data input yang telah diberikan""")

st.markdown(f"<center style='margin-top:50px;'>Dibuat oleh Yusuf Sugiono &copy;{datetime.now().year} Universitas Trunojoyo</center>",
            unsafe_allow_html=True)
