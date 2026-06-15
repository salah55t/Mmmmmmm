# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
from typing import List, Optional
import uvicorn
import os

app = FastAPI(
    title="Binance Local Analytics Proxy API",
    description="جسر تحليلي متطور يتصل مباشرة بـ Binance API ويحسب المؤشرات محلياً"
)

# نموذج استقبال البيانات لضمان التوافق والأمان
class TickerRequest(BaseModel):
    tickers: List[str]
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

# --- الدوال الرياضية لحساب المؤشرات فريش داخل السيرفر ---
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """حساب مؤشر القوة النسبية RSI بدقة Wilder الرياضية"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """حساب مؤشر قوة الاتجاه ADX بدقة فنية كاملة"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9))
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    return dx.ewm(alpha=1/period, adjust=False).mean()

def fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 100, api_key: Optional[str] = None) -> pd.DataFrame:
    """جلب بيانات الشموع مباشرة من خوادم باينانس الرسمية"""
    url = "https://api.binance.com/api/v3/klines"
    headers = {}
    if api_key:
        headers["X-MBX-APIKEY"] = api_key
        
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    response = requests.get(url, params=params, headers=headers, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Binance API returned status {response.status_code}")
        
    data = response.json()
    # تحويل مصفوفة باينانس إلى DataFrame
    df = pd.DataFrame(data, columns=[
        'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
        'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades',
        'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'
    ])
    
    # تحويل أنواع البيانات إلى أرقام عشرية للحساب الفني
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
        
    return df

@app.get("/")
def health_check():
    return {"status": "online", "message": "Binance Analytics Proxy is fully active!"}

@app.post("/scan")
@app.post("/scan/")
def scan_tickers(payload: TickerRequest):
    print(f"⚙️ جاري معالجة فحص لـ {len(payload.tickers)} عملة مباشرة عبر Binance API...")
    combined_records = []
    
    for t in payload.tickers:
        try:
            # استخراج الرمز الصافي المتوافق مع باينانس (مثال: BTCUSDT)
            symbol_clean = t.replace("BINANCE:", "").replace(":", "")
            
            # جلب البيانات التاريخية من باينانس
            df = fetch_binance_klines(symbol_clean, interval="1h", limit=100, api_key=payload.api_key)
            if df.empty or len(df) < 30:
                continue
                
            # حساب المؤشرات محلياً بدقة متناهية
            df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
            df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
            
            # حساب حزم بولينجر
            sma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            df['BB_lower'] = sma20 - (2 * std20)
            df['BB_upper'] = sma20 + (2 * std20)
            
            # حساب RSI و ADX
            df['RSI'] = calculate_rsi(df['Close'], 14)
            df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'], 14)
            
            # استخراج الشمعة الأخيرة والسابقة
            last_row = df.iloc[-1]
            close_price = float(last_row['Close'])
            open_price = float(last_row['Open'])
            change_pct = ((close_price - open_price) / open_price) * 100
            
            record = {
                "ticker": t,
                "close": close_price,
                "change": change_pct,
                "volume": float(last_row['Volume']) if pd.notna(last_row['Volume']) else 0.0,
                "RSI": float(last_row['RSI']) if pd.notna(last_row['RSI']) else None,
                "ADX": float(last_row['ADX']) if pd.notna(last_row['ADX']) else None,
                "EMA9": float(last_row['EMA9']) if pd.notna(last_row['EMA9']) else None,
                "EMA21": float(last_row['EMA21']) if pd.notna(last_row['EMA21']) else None,
                "BB.lower": float(last_row['BB_lower']) if pd.notna(last_row['BB_lower']) else None,
                "BB.upper": float(last_row['BB_upper']) if pd.notna(last_row['BB_upper']) else None
            }
            combined_records.append(record)
        except Exception as e:
            # تخطي أي عملة تفشل في الجلب لمنع توقف الرادار
            print(f"⚠️ فشل تحليل {t}: {e}")
            continue
            
    if not combined_records:
        raise HTTPException(status_code=404, detail="No data could be fetched from Binance API")
        
    return {"success": True, "data": combined_records}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
