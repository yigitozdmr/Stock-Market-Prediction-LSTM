import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# --- AYARLAR ---
TICKER = "SPY"  # S&P 500
START_DATE = "2005-01-01" # Daha uzun vade için veriyi biraz geriye çektik
END_DATE = "2023-12-31"
SEQUENCE_LENGTH = 60

# Tahmin Hedefleri (İş Günü Cinsinden)
# Borsa hafta sonu kapalı olduğu için 1 Ay ≈ 20-21 İş Günüdür.
HORIZONS = {
    "1 Hafta (5 Gün)": 5,
    "3 Ay (60 Gün)": 60,
    "6 Ay (120 Gün)": 120
}

# --- GÜNCELLENMİŞ VERİ ÇEKME FONKSİYONU ---
def get_data_with_target(ticker, start, end, horizon_days):
    """Hata korumalı veri çekme fonksiyonu."""
    print(f"{ticker} verisi indiriliyor...")
    
    # timeout parametresi ekleyerek bekleme süresini artırabiliriz ama yfinance bunu bazen desteklemez.
    # En temiz yöntem basitçe indirmeyi denemektir.
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as e:
        print(f"HATA: Veri indirilirken sorun oluştu: {e}")
        return None, None

    # Veri boş mu kontrol et
    if len(df) == 0:
        print("HATA: Yahoo Finance boş veri döndürdü. İnternet bağlantını kontrol et.")
        # Burada programı durdurmamız lazım yoksa aşağıda hata verir
        raise ValueError("Veri indirilemediği için işlem durduruldu.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Özellikler
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    loss[loss == 0] = 0.001
    rs = gain / loss
    df['RSI'] = (100 - (100 / (1 + rs))) / 100.0

    # Trend Uzaklığı (SMA 50)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Trend_Dist'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    
    # Hedef
    df['Target'] = (df['Close'].shift(-horizon_days) > df['Close']).astype(int)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    
    features = df[['Log_Ret', 'RSI', 'Trend_Dist']].values
    target = df['Target'].values
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, target

def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(target[i])
    return np.array(X), np.array(y)

def train_evaluate(period_name, horizon_days):
    print(f"\n{'='*40}")
    print(f"🚀 EĞİTİM BAŞLIYOR: {period_name}")
    print(f"{'='*40}")
    
    # 1. Veri Hazırla
    data, target = get_data_with_target(TICKER, START_DATE, END_DATE, horizon_days)
    X, y = create_sequences(data, target, SEQUENCE_LENGTH)
    
    # Bölme
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Ağırlık Hesapla
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Model
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=True),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Eğit (Daha hızlı sonuç için epoch 15 yaptım, artırabilirsin)
    model.fit(X_train, y_train, epochs=15, batch_size=32, 
              validation_data=(X_test, y_test), 
              class_weight=class_weight_dict, verbose=0) # verbose=0 çıktıyı temiz tutar
    
    # Test Et
    preds = (model.predict(X_test) > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    
    print(f"✅ {period_name} Tamamlandı. Doğruluk: %{acc*100:.2f}")
    return acc

# --- ANA DÖNGÜ ---
results = {}

for name, days in HORIZONS.items():
    acc = train_evaluate(name, days)
    results[name] = acc

print("\n\n################################")
print("📊 SONUÇ ÖZET TABLOSU")
print("################################")
print(f"{'VADE':<20} | {'DOĞRULUK (ACCURACY)':<20}")
print("-" * 40)
for name, acc in results.items():
    print(f"{name:<20} | %{acc*100:.2f}")