# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import asyncio
import httpx
from typing import List, Optional
import uvicorn
import os

app = FastAPI(
    title="Binance Async Analytics Proxy API",
    description="جسر تحليلي خارق السرعة يجلب بيانات مئات العملات بالتوازي من باينانس خلال ثوانٍ معدودة"
)

class TickerRequest(BaseModel):
    tickers: List[str]
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

# --- الدوال الرياضية لحساب المؤشرات فريش داخل السيرفر ---
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
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

# --- جلب بيانات عملة واحدة بشكل غير متزامن فائق السرعة ---
async def fetch_single_ticker(client: httpx.AsyncClient, ticker_raw: str, api_key: Optional[str] = None) -> Optional[dict]:
    symbol_clean = ticker_raw.replace("BINANCE:", "").replace(":", "")
    # تصفية أزواج العملات المستقرة وغير المهمة لتسريع الفحص
    if any(stable in symbol_clean for stable in ["USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT", "EURUSDT"]):
        return None
        
    url = "https://api.binance.com/api/v3/klines"
    headers = {"X-MBX-APIKEY": api_key} if api_key else {}
    params = {
        "symbol": symbol_clean,
        "interval": "1h",
        "limit": "100"
    }
    
    try:
        response = await client.get(url, params=params, headers=headers, timeout=8.0)
        if response.status_code != 200:
            return None
            
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
            'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades',
            'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'
        ])
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
            
        # حساب المؤشرات الفنية محلياً
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['BB_lower'] = sma20 - (2 * std20)
        df['BB_upper'] = sma20 + (2 * std20)
        
        df['RSI'] = calculate_rsi(df['Close'], 14)
        df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'], 14)
        
        last_row = df.iloc[-1]
        close_price = float(last_row['Close'])
        open_price = float(last_row['Open'])
        change_pct = ((close_price - open_price) / open_price) * 100
        
        return {
            "ticker": ticker_raw,
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
    except Exception:
        return None

@app.get("/")
def health_check():
    return {"status": "online", "message": "Binance Async Proxy is active!"}

@app.post("/scan")
@app.post("/scan/")
async def scan_tickers(payload: TickerRequest):
    print(f"⚡ بدء الفحص المتوازي غير المتزامن لـ {len(payload.tickers)} عملة...")
    
    # استخدام سياق اتصال واحد لإرسال مئات الطلبات بالتوازي في جزء من الثانية
    async with httpx.AsyncClient() as client:
        tasks = [fetch_single_ticker(client, t, payload.api_key) for t in payload.tickers]
        results = await asyncio.gather(*tasks)
        
    # تصفية النتائج من القيم الفارغة أو الفاشلة
    combined_records = [r for r in results if r is not None]
    
    print(f"📊 اكتمل الفحص بنجاح! تم جلب وتحليل {len(combined_records)} عملة فريش.")
    
    if not combined_records:
        raise HTTPException(status_code=404, detail="Failed to fetch data from Binance API")
        
    return {"success": True, "data": combined_records}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
