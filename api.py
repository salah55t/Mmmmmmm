# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from binance import AsyncClient  # الاستيراد الرسمي لمكتبة باينانس غير المتزامنة
import pandas as pd
import numpy as np
import asyncio
from typing import List, Optional
import uvicorn
import os

app = FastAPI(
    title="Binance Official Library Proxy API",
    description="جسر تحليلي خارق السرعة يتصل بباينانس عبر مكتبة python-binance الرسمية ويعالج البيانات محلياً"
)

# قراءة مفاتيح باينانس الرسمية والمشفرة من متغيرات بيئة Render
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

class TickerRequest(BaseModel):
    tickers: List[str]

# --- الحسابات الرياضية للمؤشرات الفنية داخل السيرفر ---
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """حساب مؤشر القوة النسبية RSI بدقة Wilder"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """حساب مؤشر الاتجاه ADX بدقة فنية تامة"""
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

# --- جلب شارات وبيانات عملة واحدة باستخدام مكتبة باينانس الرسمية ---
async def fetch_single_ticker_with_binance_lib(client: AsyncClient, ticker_raw: str) -> Optional[dict]:
    symbol_clean = ticker_raw.replace("BINANCE:", "").replace(":", "")
    
    # تصفية أزواج العملات المستقرة والجانبية لتقليص وقت الفحص وتوفير الموارد
    if any(stable in symbol_clean for stable in ["USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT", "EURUSDT"]):
        return None
        
    try:
        # استخدام دالة get_klines الرسمية من مكتبة python-binance لجلب 100 شمعة ساعة فريش
        klines = await client.get_klines(
            symbol=symbol_clean,
            interval=AsyncClient.KLINE_INTERVAL_1HOUR,
            limit=100
        )
        
        if not klines or len(klines) < 30:
            return None
            
        # تحويل مصفوفة باينانس الرسمية إلى DataFrame
        df = pd.DataFrame(klines, columns=[
            'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
            'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades',
            'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'
        ])
        
        # تحويل الأعمدة الرقمية إلى قيم عشرية صالحة للحسابات الفنية
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
            
        # حساب المتوسطات الأسية السريعة والبطيئة محلياً
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        # حساب حزم بولينجر (Bollinger Bands)
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['BB_lower'] = sma20 - (2 * std20)
        df['BB_upper'] = sma20 + (2 * std20)
        
        # حساب مؤشرات القوة والزخم RSI و ADX
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
    key_status = "متصل ومعرّف في النظام بنجاح ✅" if BINANCE_API_KEY else "غير معرّف! يرجى إضافته لمتغيرات ريندر ⚠️"
    return {
        "status": "online", 
        "message": "FastAPI with python-binance AsyncClient is fully operational!",
        "binance_key_status": key_status
    }

@app.post("/scan")
@app.post("/scan/")
async def scan_tickers(payload: TickerRequest):
    print(f"⚡ بدء استدعاء رسمي عبر مكتبة python-binance لـ {len(payload.tickers)} عملة بالتوازي...")
    
    # تهيئة كائن الاتصال الآمن بالمكتبة باستخدام المفاتيح الرقمية المشفرة
    client = await AsyncClient.create(
        api_key=BINANCE_API_KEY if BINANCE_API_KEY else None,
        api_secret=BINANCE_API_SECRET if BINANCE_API_SECRET else None
    )
    
    try:
        # إرسال طلبات مئات العملات دفعة واحدة بالتوازي لإكمالها خلال 1-2 ثانية فقط
        tasks = [fetch_single_ticker_with_binance_lib(client, t) for t in payload.tickers]
        results = await asyncio.gather(*tasks)
    finally:
        # إغلاق جلسة الاتصال بباينانس بشكل آمن لمنع تسريب الذاكرة
        await client.close_connection()
        
    combined_records = [r for r in results if r is not None]
    print(f"📊 اكتمل الفحص الرسمي! تم تحليل {len(combined_records)} عملة فريش بنجاح.")
    
    if not combined_records:
        raise HTTPException(status_code=404, detail="No active data could be analyzed via official Binance Library")
        
    return {"success": True, "data": combined_records}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
