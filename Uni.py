import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import date, timedelta
from PIL import Image

# Tambahkan gambar
img = Image.open('Ihram Adi Pratama.jpg')

# Custom CSS for styling the sidebar image caption
st.markdown("""
    <style>
        .sidebar .stImage > div > div > div > span {
            font-size: 18px;  /* Ukuran font lebih besar */
            font-weight: bold;  /* Menjadikan teks tebal */
            color: #00008B;  /* Mengubah warna teks (optional) */
        }
        .sidebar img {
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
            width: 100%;
            object-fit: cover;  /* Menjaga rasio gambar */
        }
    </style>
""", unsafe_allow_html=True)

# Display image in sidebar with use_container_width=True
st.sidebar.image(img, caption='Ihram Adi Pratama', use_container_width=True)


########## Membuat Fungsi ###############
# Fungsi untuk load dan persiapkan data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=True)
    data = data[['Close', 'Open', 'High', 'Low', 'Volume']]  # Menggunakan lebih banyak kolom untuk analisis
    return data

# Fungsi untuk memproses data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Fungsi untuk membuat dataset untuk NNAR dengan lag
def create_nnar_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])  # Menggunakan lag (60 hari sebelumnya)
        y.append(data[i, 0])  # Nilai yang diprediksi (harga penutupan berikutnya)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Fungsi untuk membangun model NNAR
def build_nnar_model(input_shape):
    model = Sequential()
    model.add(Dense(units=128, activation='relu', input_shape=input_shape))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Fungsi untuk melatih model NNAR
def train_nnar_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Fungsi untuk menghitung MAPE (Mean Absolute Percentage Error)
def calculate_mape(y_actual, y_predicted):
    return np.mean(np.abs((y_actual - y_predicted) / y_actual)) * 100

# Fungsi untuk menampilkan candlestick chart
def plot_candlestick(data):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

# Fungsi untuk menampilkan moving average pada harga saham
def plot_moving_average(data, window=30):
    data['MA'] = data['Close'].rolling(window=window).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(data['MA'], label=f'{window}-Day Moving Average', color='orange')
    plt.title(f'{window}-Day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot()

# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    st.title("Peramalan Waktu Menggunakan Beberapa Model")

    # Deskripsi singkat tentang model
    st.markdown("""
    **By Ihram Adi Pratama**.

    **LSTM (Long Short-Term Memory):** Model ini efektif untuk data deret waktu karena kemampuannya untuk mengingat informasi jangka panjang. LSTM digunakan untuk memprediksi harga saham berdasarkan pola yang berkembang seiring waktu.

    **NNAR (Neural Network Autoregressive):** Model ini menggunakan data historis dengan *lag* untuk memprediksi harga saham masa depan. Dengan menggunakan nilai-nilai sebelumnya (misalnya, harga penutupan 60 hari terakhir), model ini membuat prediksi harga di masa depan.
    """)

    # Sidebar untuk input pengguna
    model_choice = st.sidebar.selectbox("Pilih Model Peramalan", options=["LSTM", "NNAR"])
    
    # Pilih durasi data yang diinginkan
    duration = st.sidebar.selectbox("Pilih Durasi", options=["3 Tahun", "5 Tahun", "7 Tahun", "9 Tahun"], index=0)
    
    # Tentukan tahun mulai dan selesai berdasarkan durasi
    if duration == "3 Tahun":
        start_year = date.today().year - 3
        end_year = date.today().year
    elif duration == "5 Tahun":
        start_year = date.today().year - 5
        end_year = date.today().year
    elif duration == "7 Tahun":
        start_year = date.today().year - 7
        end_year = date.today().year
    elif duration == "9 Tahun":
        start_year = date.today().year - 9
        end_year = date.today().year
    
    # Pilih bulan dan hari untuk peramalan
    month = st.sidebar.selectbox("Bulan", options=list(range(1, 13)), index=0, format_func=lambda x: f'{x:02d}')
    days_to_predict = st.sidebar.slider("Hari untuk Diramalkan", min_value=1, max_value=60, value=30)
    
    # Menambahkan input untuk kontrol model hyperparameters
    st.sidebar.markdown("### Hyperparameter Model")
    epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=50, value=10)
    batch_size = st.sidebar.number_input("Ukuran Batch", min_value=16, max_value=128, value=32)

    # Mengatur format tanggal
    start_date = f"{start_year}-{month:02d}-01"
    end_date = f"{end_year}-{month:02d}-{pd.Timestamp(end_year, month, 1).days_in_month}"
    
    st.sidebar.markdown('<div style="background-color:#f0f0f5;padding:10px;border-radius:8px;">', unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Menampilkan feedback pemilihan model dan tanggal
    st.write(f"**Model yang dipilih**: {model_choice}")
    st.write(f"**Data dari**: {start_date} ke {end_date}")
    st.write(f"**Hari yang akan diramalkan**: {days_to_predict}")

    # Load and preprocess data
    st.write(f"Mengambil data dari {start_date} ke {end_date}...")
    data = load_data("AAPL", start_date, end_date)  # Menggunakan AAPL sebagai default ticker
    st.write("Preview Dataset:", data.head())
    
    # Plot data aktual sebelum prediksi
    st.subheader("Data Aktual")
    fig = plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Harga Aktual', color='blue')
    plt.title(f"Harga Saham Aktual (AAPL)")
    plt.xlabel('Tanggal')
    plt.ylabel('Harga')
    plt.legend()
    st.pyplot(fig)

    # Preprocessing Data
    scaled_data, scaler = preprocess_data(data)
    
    # Train-Test Split
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Buat dataset menggunakan lag untuk NNAR
    X_train, y_train = create_nnar_dataset(train_data)
    X_test, y_test = create_nnar_dataset(test_data)

    # Ubah bentuk X_train dan X_test menjadi tiga dimensi untuk LSTM (samples, timesteps, features)
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Inisialisasi forecast_actual untuk semua model
    forecast_actual = None

    # Pilih model peramalan
    if model_choice == "LSTM":
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=1))  # Output layer
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Melatih model LSTM dengan hyperparameters
        model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Prediksi
        predictions = model.predict(X_test_lstm)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        forecast_actual = predictions  # Menetapkan hasil prediksi LSTM
        
    elif model_choice == "NNAR":
        # Build NNAR model using MLP
        model = build_nnar_model((X_train.shape[1],))
        model = train_nnar_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size)
        
        # Prediksi
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        forecast_actual = predictions  # Menetapkan hasil prediksi NNAR

    # Tampilkan grafik hasil peramalan dan data aktual
    st.subheader(f"{model_choice} - Peramalan vs Harga Aktual")

    # Buat figure baru untuk grafik
    fig = plt.figure(figsize=(10, 6))

    # Plot data harga aktual
    plt.plot(data.index, data['Close'], label='Harga Aktual', color='blue', alpha=0.5)

    # Plot harga yang diprediksi
    plt.plot(data.index[-len(forecast_actual):], forecast_actual, label=f'Prediksi Harga ({model_choice})', color='red', alpha=0.7)

    # Set judul dan label
    plt.title(f'Perbandingan Harga Peramalan dan Aktual menggunakan {model_choice}')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga')
    plt.legend()

    # Tampilkan grafik di Streamlit
    st.pyplot(fig)

    # Menampilkan error metrics
    mae = mean_absolute_error(y_test_actual, forecast_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, forecast_actual))
    mape = calculate_mape(y_test_actual, forecast_actual)

    st.subheader("Error Metrics")
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Prediksi untuk n hari ke depan
    st.subheader(f"Peramalan untuk {days_to_predict} Hari Ke Depan")
    future_forecast = []

    if model_choice == "LSTM":
        # Mulai dengan urutan terakhir dari X_test_lstm
        last_sequence = X_test_lstm[-1]  # Urutan input terakhir di X_test_lstm
        
        # Lakukan prediksi untuk n hari ke depan
        for _ in range(days_to_predict):
            # Lakukan prediksi
            next_prediction = model.predict(last_sequence.reshape(1, last_sequence.shape[0], 1)).flatten()[0]
            future_forecast.append(next_prediction)
            
            # Update last_sequence: tambahkan prediksi baru dan hapus nilai pertama
            last_sequence = np.append(last_sequence[1:], [[next_prediction]], axis=0)

    else:  # Untuk model NNAR
        last_sequence = X_test[-1]
        for _ in range(days_to_predict):
            next_prediction = model.predict(last_sequence.reshape(1, last_sequence.shape[0])).flatten()[0]
            future_forecast.append(next_prediction)
            last_sequence = np.append(last_sequence[1:], next_prediction)

    future_forecast_actual = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))
    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_to_predict)

    fig = plt.figure(figsize=(12, 6))
    plt.plot(future_dates, future_forecast_actual, label='Peramalan Masa Depan', color='red')
    plt.title(f'Peramalan Harga {days_to_predict} Hari Ke Depan')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Tampilkan nilai peramalan masa depan
    st.subheader(f"Peramalan Harga untuk {days_to_predict} Hari Ke Depan")
    
    # Tampilkan DataFrame yang berisi tanggal dan harga peramalan
    forecast_df = pd.DataFrame({
        'Tanggal': future_dates,
        'Prediksi Harga': future_forecast_actual.flatten()
    })
    
    # Tampilkan tabel nilai peramalan
    st.write(forecast_df)

if __name__ == '__main__':
    main()