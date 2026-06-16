# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from binance import AsyncClient
import pandas as pd
import numpy as np
import asyncio
from typing import List, Optional
import uvicorn
import os

app = FastAPI(
    title="Cyber-Radar Analytics API Matrix",
    description="الجسر التحليلي المتكامل لحساب المؤشرات الفنية للرادار باستخدام AsyncClient"
)

# سحب مفاتيح البيئة بأمان من إعدادات ريندر
API_KEY = os.environ.get("BINANCE_API_KEY")
API_SECRET = os.environ.get("BINANCE_API_SECRET")

# هيكلة الطلب القادم من الرادار
class TickerRequest(BaseModel):
    tickers: List[str]
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

# --- [1] محرك الحسابات الفنية المتقدمة ---

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """حساب مؤشر القوة النسبية RSI فريش"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """حساب مؤشر معدل الحركة الاتجاهية ADX لقياس قوة الترند"""
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

# --- [2] سحب البيانات الفورية والمعالجة بالتوازي ---

async def process_single_ticker(client: AsyncClient, ticker_raw: str) -> Optional[dict]:
    """جلب الشموع لعملة واحدة، حساب مؤشراتها، وتجهيزها للرادار"""
    # تنظيف الرمز القادم من الرادار (مثال: BINANCE:BTCUSDT يصبح BTCUSDT)
    symbol_clean = ticker_raw.replace("BINANCE:", "").replace(":", "")
    
    # تصفية أزواج العملات المستقرة لتوفير موارد السيرفر وزيادة السرعة
    if any(stable in symbol_clean for stable in ["USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT", "EURUSDT"]):
        return None
    
    try:
        # جلب أحدث 100 شمعة بإطار 15 دقيقة (15m) كما حددت بدقة
        klines = await client.get_klines(symbol=symbol_clean, interval='15m', limit=100)
        if not klines:
            return None
            
        # تحويل البيانات إلى DataFrame للتعامل الرياضي السريع
        df = pd.DataFrame(klines, columns=[
            'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
            'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades',
            'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'
        ])
        
        # تحويل الأعمدة إلى قيم عشرية قابلة للحساب
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
            
        # حساب المتوسطات المتحركة الأسية (EMA) المطلوبة في الاستراتيجية
        df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        # حساب حدود البولينجر باند (Bollinger Bands)
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['BB_lower'] = sma20 - (2 * std20)
        df['BB_upper'] = sma20 + (2 * std20)
        
        # حساب الزخم وقوة الاتجاه (RSI & ADX)
        df['RSI'] = calculate_rsi(df['Close'], 14)
        df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'], 14)
        
        # استخراج السطر الأخير (الشمعة الحالية المغلقة/النشطة)
        last_row = df.iloc[-1]
        close_price = float(last_row['Close'])
        open_price = float(last_row['Open'])
        change_pct = ((close_price - open_price) / open_price) * 100
        
        # حزم البيانات في قاموس نظيف بنفس المفاتيح التي يبحث عنها الرادار واجهتك
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

# --- [3] المسارات البرمجية (Endpoints) ---

@app.get("/")
def health_check():
    """مسار فحص السلامة الأساسي لمنع مشاكل العرض والتأكد من عمل السيرفر"""
    return {"status": "online", "message": "Cyber-Radar Analytics Core is operational!"}

@app.post("/scan")
@app.post("/scan/")
async def scan_tickers(payload: TickerRequest):
    """المستقبل الرئيسي لطلبات الرادار - يقوم بمسح كل العملات دفعة واحدة بالتوازي"""
    print(f"📡 تم استقبال أمر مسح وحساب المؤشرات لـ {len(payload.tickers)} عملة...")
    
    # فتح اتصال واحد آمن باستخدام المفاتيح الممررة أو المخزنة بالسيرفر
    client = await AsyncClient.create(API_KEY, API_SECRET)
    
    try:
        # إطلاق المهام بالتوازي عبر asyncio.gather لسرعة خارقة (أجزاء من الثانية)
        tasks = [process_single_ticker(client, t) for t in payload.tickers]
        results = await asyncio.gather(*tasks)
    finally:
        # إغلاق الاتصال إجبارياً لحماية السيرفر من تجميد المنافذ (Connection Leaks)
        await client.close_connection()
        
    # تصفية المصفوفة من أي عملة فشل جلب بياناتها أو تم استبعادها
    combined_records = [r for r in results if r is not None]
    
    print(f"📊 اكتمل الفحص الحسابي! تم إرسال {len(combined_records)} مصفوفة مؤشرات فنية إلى الرادار.")
    
    if not combined_records:
        raise HTTPException(status_code=404, detail="No active ticker data could be scanned from Binance")
        
    return {"success": True, "data": combined_records}

if __name__ == "__main__":
    # تشغيل السيرفر والتوافق التلقائي مع البورت الخاص بـ Render
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
