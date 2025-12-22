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
VIX_TICKER = "^VIX"
START_DATE = "2010-01-01"
END_DATE = "2023-12-31"
SEQUENCE_LENGTH = 60
HORIZON = 60  
DECISION_THRESHOLD = 0.40 

def generate_dummy_market_data(start, end):
    """
    İnternet yoksa devreye giren 'Akıllı' Sahte Veri Üreticisi.
    VIX ve Fiyat arasındaki ters korelasyonu simüle eder.
    """
    print("⚠️ UYARI: İnternet bağlantısı yok. SENARYO VERİSİ üretiliyor...")
    dates = pd.date_range(start=start, end=end)
    days = len(dates)
    
    # 1. Rastgele Piyasa Hareketleri
    np.random.seed(42)
    market_sentiment = np.random.normal(0, 1, days) # Piyasa ruh hali
    
    # 2. VIX (Korku) Oluşturma
    # Piyasa kötüyse VIX artar (15 taban puan + gürültü)
    vix_close = 15 + (market_sentiment * -5) + np.random.normal(0, 2, days)
    vix_close = np.maximum(vix_close, 10) # VIX 10'un altına düşmesin
    
    # 3. SPY (Fiyat) Oluşturma
    # VIX yüksekse getiri düşer
    returns = (market_sentiment * 0.005) + 0.0002 # Hafif yukarı trend
    price = 100 * np.cumprod(1 + returns)
    
    # 4. DataFrame Birleştirme
    df = pd.DataFrame(index=dates)
    df['Close'] = price
    df['VIX_Close'] = vix_close
    df['Volume'] = np.random.randint(1000000, 5000000, days)
    
    return df

def get_enhanced_data(ticker, vix_ticker, start, end, horizon):
    print(f"📥 {ticker} ve {vix_ticker} verileri indiriliyor...")
    
    df = pd.DataFrame()
    
    try:
        # İNTERNET VARSA:
        df_spy = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        df_vix = yf.download(vix_ticker, start=start, end=end, auto_adjust=True, progress=False)
        
        # Veri inmezse hata fırlat ki 'except' bloğuna düşsün
        if len(df_spy) == 0 or len(df_vix) == 0:
            raise ValueError("Veri boş geldi.")

        if isinstance(df_spy.columns, pd.MultiIndex): df_spy.columns = df_spy.columns.get_level_values(0)
        if isinstance(df_vix.columns, pd.MultiIndex): df_vix.columns = df_vix.columns.get_level_values(0)

        # Birleştirme
        df = df_spy.join(df_vix['Close'], rsuffix='_VIX')
        df.rename(columns={'Close_VIX': 'VIX_Close'}, inplace=True)
        
    except Exception as e:
        print(f"❌ İndirme Hatası: {e}")
        # İNTERNET YOKSA:
        df = generate_dummy_market_data(start, end)
    
    # Boşluk temizliği
    df.dropna(inplace=True)

    # --- ÖZELLİK MÜHENDİSLİĞİ ---
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    loss[loss == 0] = 0.001
    rs = gain / loss
    df['RSI'] = (100 - (100 / (1 + rs))) / 100.0

    # Trend
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Trend_Dist'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    
    # VIX Norm
    df['VIX_Norm'] = df['VIX_Close'] / 80.0 

    # HEDEF
    df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    
    price_data = df['Close'].copy()
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    
    price_data = price_data.loc[df.index]
    
    features = df[['Log_Ret', 'RSI', 'Trend_Dist', 'VIX_Norm']].values
    target = df['Target'].values
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, target, price_data

# --- ANA AKIŞ ---
try:
    X_scaled, y, prices = get_enhanced_data(TICKER, VIX_TICKER, START_DATE, END_DATE, HORIZON)

    # Sequence Oluşturma
    X_seq, y_seq = [], []
    test_prices = [] 

    for i in range(SEQUENCE_LENGTH, len(X_scaled)):
        X_seq.append(X_scaled[i-SEQUENCE_LENGTH:i])
        y_seq.append(y[i])
        test_prices.append(prices.iloc[i]) 

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    test_prices = np.array(test_prices)

    # Bölme
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    price_test = test_prices[split:] 

    # Ağırlıklar
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Model Kurulumu
    print("🧠 Gelişmiş Model (VIX Dahil) Eğitiliyor...")
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
    
    model.fit(X_train, y_train, epochs=25, batch_size=32, 
              class_weight=class_weight_dict, verbose=0)

    # --- SİMÜLASYON ---
    print("📈 Ticaret Simülasyonu Hesaplanıyor...")
    
    pred_probs = model.predict(X_test)
    predictions = (pred_probs > DECISION_THRESHOLD).astype(int).flatten()

    actual_returns = pd.Series(price_test).pct_change().fillna(0)
    strategy_returns = actual_returns * predictions

    cumulative_market = (1 + actual_returns).cumprod()
    cumulative_strategy = (1 + strategy_returns).cumprod()

    # --- GRAFİK ---
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_market, label='Piyasa (SPY)', color='gray', alpha=0.5, linestyle='--')
    plt.plot(cumulative_strategy, label='Yapay Zeka (VIX + Agresif)', color='blue', linewidth=2)
    
    plt.title(f"Gelişmiş AI Stratejisi (Eşik: {DECISION_THRESHOLD} | VIX Dahil)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylabel("Büyüme Çarpanı")
    
    final_market = cumulative_market.iloc[-1]
    final_strategy = cumulative_strategy.iloc[-1]
    
    print(f"\n--- YENİ SONUÇLAR ---")
    print(f"Piyasa: {final_market:.2f}x")
    print(f"Yapay Zeka: {final_strategy:.2f}x")
    
    if final_strategy > final_market:
        print(f"🚀 MÜKEMMEL! Piyasayı %{((final_strategy/final_market)-1)*100:.1f} farkla yendin.")
    else:
        print("⚠️ Model piyasanın gerisinde, fakat güvenli limanda olabilir.")
        
    plt.show()

except Exception as e:
    print(f"Kritik Hata: {e}")