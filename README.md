# LSTM Stock Trend Predictor 📈

An advanced Deep Learning project that predicts the directional movement of the S&P 500 (SPY) index using LSTM (Long Short-Term Memory) neural networks. This project utilizes Multi-Timeframe Analysis and incorporates Market Sentiment (VIX) to achieve robust results.

## 🚀 Key Features
* **Deep Learning Architecture:** Custom Stacked LSTM model with Dropout regularization to prevent overfitting.
* **Smart Data Pipeline:**
  * Fetches data via `yfinance`.
  * **Offline Mode:** Includes a synthetic data generator that simulates market correlations (Price vs. VIX) if the internet connection fails.
* **Feature Engineering:**
  * Log Returns
  * RSI (Relative Strength Index)
  * SMA Distance (Trend Following)
  * **VIX Integration:** Uses the "Fear Index" to detect market stress (in V2).
* **Backtesting Engine:**
  * Compares AI strategy vs. Buy & Hold strategy.
  * Visualizes cumulative returns.

## 📂 Project Structure
* `main.py` / `LSTM_GRU_finance.py`: The core training script for the LSTM model.
* `backtest.py`: Standard backtesting engine (Price & Technicals only).
* `backtest_v2.py`: **Advanced** engine with VIX integration, lower decision thresholds, and offline capabilities.
* `backtest_v3_bist.py`: **Borsa Istanbul (BIST) Edition.**
  * Analyzes Turkish stocks (e.g., ASELS, THYAO).
  * Uses **USD/TRY** currency rates instead of VIX as the macro-economic risk indicator.
  * Adapted for high-inflation market dynamics.

## 📊 Performance Highlights
* **Accuracy:** Achieved **~67% accuracy** on 3-Month (Quarterly) trend predictions.
* **Real-World Backtest Results:**
  * **Market Return (Buy & Hold):** 1.19x (+19%)
  * **AI Strategy Return:** 1.32x (+32%)
  * **Alpha:** The AI outperformed the market by **+11.2%** using the VIX-enhanced strategy.

## 🛠️ Installation & Usage

1. **Install Dependencies:**
   ```bash
   pip install numpy pandas yfinance scikit-learn tensorflow matplotlib

## 📜 License / Lisans
**Copyright © 2025 Yiğit Özdemir**

This project is created for **educational and portfolio purposes only**.
* You are free to view, download, and learn from the code.
* You **may not** use this code for commercial purposes.
* You **may not** redistribute or modify this code without explicit permission.

*Bu proje sadece eğitim ve portföy amaçlı oluşturulmuştur.*
* *Kodları inceleyebilir ve öğrenebilirsiniz.*
* *Bu kodu ticari amaçlarla **kullanamazsınız**.*
* *İzin almadan kodu dağıtamaz veya değiştiremezsiniz.*