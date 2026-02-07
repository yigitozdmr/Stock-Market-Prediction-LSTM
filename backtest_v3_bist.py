import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# --- TÜRK HİSSELERİ LİSTESİ ---
STOCKS = {
    "1": ("ASELS.IS", "Aselsan"),
    "2": ("TUPRS.IS", "Tüpraş"),
    "3": ("BIMAS.IS", "Bimaş"),
    "4": ("THYAO.IS", "Türk Hava Yolları"),
    "5": ("GARAN.IS", "Garanti BBVA"),
    "6": ("KOZAL.IS", "Koza Altın (Turk Altin Isletmeleri)") 
}

print("\n--- BORSA İSTANBUL ANALİZİ ---")
print("Lütfen analiz etmek istediğiniz hisseyi seçin:")
for key, val in STOCKS.items():
    print(f"{key}: {val[1]} ({val[0]})")

selection = input("Seçiminiz (1-6): ")
selected_stock = STOCKS.get(selection, ("THYAO.IS", "Türk Hava Yolları")) # Varsayılan THY

TICKER = selected_stock[0]
STOCK_NAME = selected_stock[1]
MACRO_TICKER = "TRY=X"  # Türkiye için 'Korku Endeksi' yerine DOLAR/TL kullanıyoruz
START_DATE = "2015-01-01" # BIST verisi için 2015 ideal
END_DATE = "2024-12-30"   # Güncel tarih
SEQUENCE_LENGTH = 60
HORIZON = 60              # 3 Aylık tahmin
DECISION_THRESHOLD = 0.40

print(f"\n✅ SEÇİLEN HİSSE: {STOCK_NAME} ({TICKER})")
print(f"📊 PİYASA GÖSTERGESİ: USD/TRY Kuru ({MACRO_TICKER})")

# --- DATA GENERATOR (OFFLINE MODE) ---
def generate_dummy_bist_data(start, end):
    print("⚠️ İNTERNET YOK: Senaryo verisi üretiliyor...")
    dates = pd.date_range(start=start, end=end)
    days = len(dates)
    np.random.seed(42)
    
    # Dolar Kuru Simülasyonu (Sürekli artan trend)
    usd_trend = np.linspace(3, 30, days) + np.random.normal(0, 0.5, days)
    
    # Hisse Simülasyonu (Enflasyonist yükseliş)
    stock_trend = np.cumprod(1 + np.random.normal(0.001, 0.02, days)) * 10
    
    df = pd.DataFrame(index=dates)
    df['Close'] = stock_trend
    df['Macro_Close'] = usd_trend
    return df

def get_bist_data(ticker, macro_ticker, start, end, horizon):
    print(f"📥 {ticker} ve {macro_ticker} verileri indiriliyor...")
    df = pd.DataFrame()
    
    try:
        # İndirme
        df_stock = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        df_macro = yf.download(macro_ticker, start=start, end=end, auto_adjust=True, progress=False)
        
        if len(df_stock) == 0: raise ValueError("Hisse verisi boş.")
            
        # MultiIndex düzeltme
        if isinstance(df_stock.columns, pd.MultiIndex): df_stock.columns = df_stock.columns.get_level_values(0)
        if isinstance(df_macro.columns, pd.MultiIndex): df_macro.columns = df_macro.columns.get_level_values(0)

        # Birleştirme (Dolar kurunu hisse verisine ekle)
        df = df_stock.join(df_macro['Close'], rsuffix='_MACRO')
        df.rename(columns={'Close_MACRO': 'Macro_Close'}, inplace=True)
        
    except Exception as e:
        print(f"❌ Veri Hatası: {e}")
        df = generate_dummy_bist_data(start, end)
    
    df.dropna(inplace=True)

    # --- ÖZELLİK MÜHENDİSLİĞİ ---
    # 1. Log Getiri
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    loss[loss == 0] = 0.001
    rs = gain / loss
    df['RSI'] = (100 - (100 / (1 + rs))) / 100.0

    # 3. Trend Mesafesi (SMA 50)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Trend_Dist'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    
    # 4. MAKRO VERİ (Dolar Kuru Değişimi)
    # Doların 30 günlük değişimi (Hızlı kur artışı kriz demektir)
    df['Macro_Change'] = df['Macro_Close'].pct_change(periods=30)

    # HEDEF (3 Ay sonra fiyat daha yüksek mi?)
    df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    
    # Fiyatları sakla (Backtest için)
    price_data = df['Close'].copy()
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    price_data = price_data.loc[df.index]
    
    # Girdiler: [Getiri, RSI, Trend, Dolar_Değişimi]
    features = df[['Log_Ret', 'RSI', 'Trend_Dist', 'Macro_Change']].values
    target = df['Target'].values
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, target, price_data

# --- ANA AKIŞ ---
try:
    X_scaled, y, prices = get_bist_data(TICKER, MACRO_TICKER, START_DATE, END_DATE, HORIZON)

    # Sequence
    X_seq, y_seq, test_prices = [], [], []
    for i in range(SEQUENCE_LENGTH, len(X_scaled)):
        X_seq.append(X_scaled[i-SEQUENCE_LENGTH:i])
        y_seq.append(y[i])
        test_prices.append(prices.iloc[i])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    test_prices = np.array(test_prices)

    # Bölme (%80 Eğitim)
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    price_test = test_prices[split:]

    # Ağırlıklar
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Model
    print(f"🧠 {STOCK_NAME} için Model Eğitiliyor...")
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
    model.fit(X_train, y_train, epochs=30, batch_size=32, class_weight=class_weight_dict, verbose=0)

    # Simülasyon
    print("📈 Simülasyon Yapılıyor...")
    pred_probs = model.predict(X_test)
    predictions = (pred_probs > DECISION_THRESHOLD).astype(int).flatten()

    actual_returns = pd.Series(price_test).pct_change().fillna(0)
    strategy_returns = actual_returns * predictions

    cum_market = (1 + actual_returns).cumprod()
    cum_strategy = (1 + strategy_returns).cumprod()

    # Grafik
    plt.figure(figsize=(12, 6))
    plt.plot(cum_market, label=f'{TICKER} (Al-Tut)', color='gray', alpha=0.5, linestyle='--')
    plt.plot(cum_strategy, label='Yapay Zeka Stratejisi', color='red', linewidth=2)
    plt.title(f"{STOCK_NAME} - BIST Yapay Zeka Analizi (Dolar Kuru Destekli)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylabel("Getiri Çarpanı")
    plt.show()

    print(f"\n--- {STOCK_NAME} SONUÇLARI ---")
    print(f"Hisse Getirisi: {cum_market.iloc[-1]:.2f}x")
    print(f"Yapay Zeka: {cum_strategy.iloc[-1]:.2f}x")

except Exception as e:
    print(f"Hata: {e}")
    