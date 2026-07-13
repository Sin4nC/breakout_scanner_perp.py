# breakout_scanner_v3.py
import argparse
import asyncio
import time
import os
import aiohttp
import numpy as np
import pandas as pd
import scipy.stats as stats
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict
import requests

# --- CONFIG MATRIX & WEIGHTS ---
CONFIG_MATRIX = {
    "WEIGHT_VOLUME": 25.0,
    "WEIGHT_MOMENTUM": 20.0,
    "WEIGHT_COILING": 15.0,
    "WEIGHT_TREND": 15.0,
    "WEIGHT_OI": 10.0,
    "WEIGHT_ALPHA": 10.0,
    "WEIGHT_FUNDING": 5.0,
    "HARD_LIQUIDITY_FLOOR_USD": 50_000.0
}

parser = argparse.ArgumentParser()
parser.add_argument("--api-url", default="https://contract.mexc.com")
parser.add_argument("--concurrency-limit", type=int, default=20)
parser.add_argument("--alpha-threshold", type=float, default=30.0)
args = parser.parse_args()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not found.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram Error: {e}")

class VectorizedQuantEngine:
    @staticmethod
    def calculate_wilder_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
        if len(closes) <= period: return np.zeros_like(closes)
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        atr = np.zeros_like(closes)
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, len(closes)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
        return atr

    @classmethod
    def evaluate_market_data(cls, h4_data: dict, daily_data: dict, btc_returns: np.ndarray, ignore_resistance: bool = False) -> Optional[dict]:
        c_h4, h_h4, l_h4, v_h4, o_h4 = h4_data['close'], h4_data['high'], h4_data['low'], h4_data['volume'], h4_data['open']
        oi_h4 = h4_data.get('open_interest', np.zeros_like(c_h4))
        c_d = daily_data['close']
        
        if len(c_h4) < 30 or len(c_d) < 20: return None
        idx = -1
        
        avg_dollar_vol_4d = float(np.mean(v_h4[-24:] * c_h4[-24:]))
        if avg_dollar_vol_4d < CONFIG_MATRIX["HARD_LIQUIDITY_FLOOR_USD"]: return None
            
        if not ignore_resistance:
            recent_max_resistance = float(np.max(h_h4[-30:-1])) if len(h_h4) >= 30 else float(np.max(h_h4[:-1]))
            if c_h4[idx] < recent_max_resistance * 0.95: return None

        atr = cls.calculate_wilder_atr(h_h4, l_h4, c_h4, period=14)
        candle_range = h_h4[idx] - l_h4[idx]
        atr_mult = candle_range / atr[idx-1] if (len(atr) > 1 and atr[idx-1] > 0) else 1.0
        
        historical_vols = v_h4[-120:-1] if len(v_h4) > 120 else v_h4[:-1]
        vol_percentile = float(stats.percentileofscore(historical_vols, v_h4[idx]) / 100.0) if len(historical_vols) > 0 else 0.5
        
        h4_series = pd.Series(c_h4)
        bbw_history = (h4_series.rolling(20).std() * 4.0) / h4_series.rolling(20).mean()
        bbw_percentile = 0.5
        if len(bbw_history) > 0:
            bbw_block = bbw_history.values[-200:-1] if len(bbw_history) > 200 else bbw_history.values[:-1]
            bbw_block = bbw_block[~np.isnan(bbw_block)]
            if len(bbw_block) > 0:
                bbw_percentile = float(stats.percentileofscore(bbw_block, bbw_history.values[-1]) / 100.0)
        
        oi_change_pct = (oi_h4[idx] - oi_h4[idx-4]) / oi_h4[idx-4] if (len(oi_h4) > 4 and oi_h4[idx-4] > 0) else 0.0
        
        asset_returns = np.diff(c_h4[-21:]) / c_h4[-22:-1] if len(c_h4) >= 22 else np.zeros(20)
        alpha_score_metric = 5.0
        if len(asset_returns) == len(btc_returns) and len(btc_returns) > 0:
            try:
                cov = np.cov(asset_returns, btc_returns)[0][1]
                btc_var = np.var(btc_returns)
                beta = cov / btc_var if btc_var > 0 else 1.0
                alpha_raw = np.mean(asset_returns) - (beta * np.mean(btc_returns))
                alpha_score_metric = max(0.0, min(10.0, alpha_raw * 1500.0))
            except: pass

        score_volume = vol_percentile * CONFIG_MATRIX["WEIGHT_VOLUME"]
        score_momentum = min((atr_mult / 3.5), 1.0) * CONFIG_MATRIX["WEIGHT_MOMENTUM"]
        score_coiling = (1.0 - bbw_percentile) * CONFIG_MATRIX["WEIGHT_COILING"]
        
        daily_ema21 = pd.Series(c_d).ewm(span=21, adjust=False).mean().values
        dist_ema21 = (c_d[idx] - daily_ema21[idx]) / daily_ema21[idx] if len(daily_ema21) > 0 else 0.0
        score_trend = min(max(0.0, (dist_ema21 + 0.05) / 0.15), 1.0) * CONFIG_MATRIX["WEIGHT_TREND"]
        
        score_oi = CONFIG_MATRIX["WEIGHT_OI"] if oi_change_pct > 0.02 else CONFIG_MATRIX["WEIGHT_OI"] * 0.5
        score_funding = CONFIG_MATRIX["WEIGHT_FUNDING"]
        score_alpha = (alpha_score_metric / 10.0) * CONFIG_MATRIX["WEIGHT_ALPHA"]
        
        composite_alpha_score = score_volume + score_momentum + score_coiling + score_trend + score_oi + score_funding + score_alpha
        
        if not ignore_resistance and composite_alpha_score < args.alpha_threshold: return None
        
        return {
            "score": round(composite_alpha_score, 1),
            "close": float(c_h4[idx]),
            "rvol_pct": round(vol_percentile * 100, 1),
            "atr_mult": round(atr_mult, 2),
            "oi_growth": round(oi_change_pct * 100, 1),
            "timestamp": datetime.fromtimestamp(int(h4_data['time'][idx]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        }

class ProductionDataPipeline:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.semaphore = asyncio.Semaphore(args.concurrency_limit)

    async def retrieve_active_universe(self, session: aiohttp.ClientSession) -> List[str]:
        url = f"{self.base_url}/api/v1/contract/detail"
        try:
            async with session.get(url, timeout=10) as response:
                payload = await response.json()
                data = payload.get("data") or []
                return [str(item.get("symbol")).upper() for item in data 
                        if str(item.get("settleCoin")).upper() == "USDT" and int(item.get("state", 0)) == 0]
        except Exception: return []

    async def fetch_clean_kline(self, session: aiohttp.ClientSession, symbol: str, interval: str, duration_days: int) -> Optional[dict]:
        end_time = int(time.time())
        start_time = end_time - (duration_days * 86400)
        url = f"{self.base_url}/api/v1/contract/kline/{symbol}"
        params = {"interval": interval, "start": start_time, "end": end_time}
        async with self.semaphore:
            for attempt in range(3):
                try:
                    async with session.get(url, params=params, timeout=12) as response:
                        if response.status == 429:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        if response.status != 200: return None
                        raw = await response.json()
                        d = raw.get("data") or {}
                        if not d or 'time' not in d or len(d['time']) == 0: return None
                        return {
                            'time': np.array(d['time'], dtype=np.int64),
                            'open': np.array(d['open'], dtype=np.float64),
                            'high': np.array(d['high'], dtype=np.float64),
                            'low': np.array(d['low'], dtype=np.float64),
                            'close': np.array(d['close'], dtype=np.float64),
                            'volume': np.array(d['volume'], dtype=np.float64),
                            'open_interest': np.array(d.get('openInterest', np.zeros_like(d['close'])), dtype=np.float64)
                        }
                except: await asyncio.sleep(0.5)
            return None

async def main():
    pipeline = ProductionDataPipeline(args.api_url)
    async with aiohttp.ClientSession() as session:
        universe = await pipeline.retrieve_active_universe(session)
        if not universe: print("Universe Empty"); return
        
        btc_data = await pipeline.fetch_clean_kline(session, "BTC_USDT", "Hour4", duration_days=20)
        btc_returns = np.zeros(20)
        if btc_data and len(btc_data['close']) > 1:
            btc_closes = btc_data['close'][-21:]
            btc_returns = np.diff(btc_closes) / btc_closes[:-1] if len(btc_closes) > 1 else np.zeros(20)

        tasks = []
        for sym in universe:
            async def scan(s=sym):
                h4 = await pipeline.fetch_clean_kline(session, s, "Hour4", duration_days=20)
                d1 = await pipeline.fetch_clean_kline(session, s, "Day1", duration_days=45)
                if h4 and d1:
                    res = VectorizedQuantEngine.evaluate_market_data(h4, d1, btc_returns, ignore_resistance=False)
                    if res: res["symbol"] = s; return res
                return None
            tasks.append(scan())
            
        raw_results = await asyncio.gather(*tasks)
        validated_signals = [res for res in raw_results if res is not None]
        
        is_fallback = False
        if not validated_signals:
            is_fallback = True
            fallback_tasks = []
            for sym in universe:
                async def scan_fb(s=sym):
                    h4 = await pipeline.fetch_clean_kline(session, s, "Hour4", duration_days=20)
                    d1 = await pipeline.fetch_clean_kline(session, s, "Day1", duration_days=45)
                    if h4 and d1:
                        res = VectorizedQuantEngine.evaluate_market_data(h4, d1, btc_returns, ignore_resistance=True)
                        if res: res["symbol"] = s; return res
                    return None
                fallback_tasks.append(scan_fb())
            raw_results = await asyncio.gather(*fallback_tasks)
            validated_signals = [res for res in raw_results if res is not None]

        validated_signals.sort(key=lambda x: x['score'], reverse=True)
        
        if validated_signals:
            mode_title = "Fallback Mode (No Res Filter)" if is_fallback else "Ultra Strategy Mode"
            msg_lines = [f"📊 *MEXC 4H Breakout Scanner V3 ({mode_title})* 🚀\n"]
            for rank, sig in enumerate(validated_signals[:15], 1):
                line = f"{rank}. *{sig['symbol']}* | Score: {sig['score']} | Price: {sig['close']} (RVOL: {sig['rvol_pct']}% | ATR: {sig['atr_mult']})\n"
                msg_lines.append(line)
            
            full_message = "\n".join(msg_lines)
            print(full_message)
            send_telegram_message(full_message)
        else:
            print("No signals found in any mode. Check API Connection.")

if __name__ == "__main__":
    asyncio.run(main())
