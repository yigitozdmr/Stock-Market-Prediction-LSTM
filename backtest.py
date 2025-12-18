import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# --- AYARLAR ---
TICKER = "SPY"
START_DATE = "2010-01-01"
END_DATE = "2023-12-31"
SEQUENCE_LENGTH = 60
HORIZON = 60  # EN BAŞARILI OLDUĞUMUZ 3 AYLIK VADE

def get_data(ticker, start, end, horizon):
    print(f"📥 {ticker} verisi indiriliyor...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    
    if len(df) == 0:
        raise ValueError("Veri indirilemedi.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Özellikler (Feature Engineering)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    loss[loss == 0] = 0.001
    rs = gain / loss
    df['RSI'] = (100 - (100 / (1 + rs))) / 100.0

    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Trend_Dist'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    
    # HEDEF: 60 Gün sonrası
    df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    
    # Backtest için fiyatları sakla
    price_data = df['Close'].copy()
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    
    # İndeksleri eşleyerek fiyat verisini de kes
    price_data = price_data.loc[df.index]
    
    features = df[['Log_Ret', 'RSI', 'Trend_Dist']].values
    target = df['Target'].values
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, target, price_data

# --- ANA AKIŞ ---
try:
    X_scaled, y, prices = get_data(TICKER, START_DATE, END_DATE, HORIZON)

    # Sequence Oluşturma
    X_seq, y_seq = [], []
    test_prices = [] 

    for i in range(SEQUENCE_LENGTH, len(X_scaled)):
        X_seq.append(X_scaled[i-SEQUENCE_LENGTH:i])
        y_seq.append(y[i])
        test_prices.append(prices.iloc[i]) 

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    test_prices = np.array(test_prices)

    # Veriyi Bölme (%80 Eğitim - %20 Backtest)
    split = int(len(X_seq) * 0.8)

    X_train = X_seq[:split]
    y_train = y_seq[:split]

    X_test = X_seq[split:]
    y_test = y_seq[split:]
    price_test = test_prices[split:] 

    # Ağırlıklar
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Model Kurulumu
    print("🧠 Model Eğitiliyor (Bu işlem veriye göre biraz sürebilir)...")
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
    
    # Modeli Eğit
    model.fit(X_train, y_train, epochs=20, batch_size=32, 
              class_weight=class_weight_dict, verbose=0)

    # --- SİMÜLASYON ---
    print("📈 Ticaret Simülasyonu Hesaplanıyor...")
    
    # 1. Tahminler
    predictions = (model.predict(X_test) > 0.5).astype(int).flatten()

    # 2. Gerçek Getiriler (Günlük % Değişim)
    actual_returns = pd.Series(price_test).pct_change().fillna(0)

    # 3. Strateji Getirisi
    # Basit Kural: Tahmin 1 ise o gün hissedsiz, 0 ise nakittesiniz (Getiri 0)
    # Not: Gerçek dünyada işlem maliyeti (komisyon) vardır, burada onu ihmal ediyoruz.
    strategy_returns = actual_returns * predictions

    # 4. Kümülatif Büyüme (Bileşik)
    cumulative_market = (1 + actual_returns).cumprod()
    cumulative_strategy = (1 + strategy_returns).cumprod()

    # --- GRAFİK ---
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_market, label='Piyasa (Al ve Tut)', color='gray', alpha=0.5, linestyle='--')
    plt.plot(cumulative_strategy, label='Yapay Zeka Modeli', color='green', linewidth=2)
    
    plt.title(f"{TICKER} - 3 Aylık Strateji Backtest Sonucu")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylabel("Para Çarpanı (Başlangıç = 1.0)")
    plt.xlabel("İşlem Günleri (Test Dönemi)")
    
    final_market = cumulative_market.iloc[-1]
    final_strategy = cumulative_strategy.iloc[-1]
    
    print(f"\n--- SONUÇLAR ---")
    print(f"Piyasa Getirisi: {final_market:.2f}x (Paranız {final_market:.2f} katına çıkardı)")
    print(f"Yapay Zeka Getirisi: {final_strategy:.2f}x (Paranız {final_strategy:.2f} katına çıkardı)")
    
    if final_strategy > final_market:
        print("✅ BAŞARI: Model piyasayı yendi!")
    else:
        print("⚠️ NOT: Model piyasanın gerisinde kaldı (Riskten kaçınmış olabilir).")
        
    plt.show()

except Exception as e:
    print(f"Hata oluştu: {e}")