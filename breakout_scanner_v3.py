import argparse
import asyncio
import os
import aiohttp
import numpy as np
import pandas as pd
import scipy.stats as stats
from datetime import datetime, timezone
from typing import Optional, List, Dict
import requests

CONFIG_MATRIX = {
    "WEIGHT_VOLUME": 30.0,
    "WEIGHT_MOMENTUM": 20.0,
    "WEIGHT_COILING": 15.0,
    "WEIGHT_TREND": 15.0,
    "WEIGHT_OI": 10.0,
    "WEIGHT_ALPHA": 10.0,
    "WEIGHT_FUNDING": 5.0,
    "HARD_LIQUIDITY_FLOOR_USD": 50_000.0,
    "RESISTANCE_THRESHOLD_PCT": 0.93,
    "ALPHA_THRESHOLD_DEFAULT": 25.0
}

parser = argparse.ArgumentParser()
parser.add_argument("--api-url", default="https://contract.mexc.com")
parser.add_argument("--concurrency-limit", type=int, default=20)
parser.add_argument("--alpha-threshold", type=float, default=CONFIG_MATRIX["ALPHA_THRESHOLD_DEFAULT"])
parser.add_argument("--debug", action="store_true", help="Enable detailed debugging output")
args = parser.parse_args()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception:
        pass

class VectorizedQuantEngine:
    @staticmethod
    def calculate_wilder_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
        if len(closes) <= period: 
            return np.zeros_like(closes)
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        atr = np.zeros_like(closes)
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, len(closes)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
        return atr

    @classmethod
    def evaluate_market_data(cls, symbol: str, h4_data: dict, daily_data: dict, btc_returns: np.ndarray, ignore_resistance: bool = False) -> Optional[dict]:
        c_h4 = h4_data['close']
        h_h4 = h4_data['high']
        l_h4 = h4_data['low']
        v_h4 = h4_data['volume']
        oi_h4 = h4_data.get('open_interest', np.zeros_like(c_h4))
        c_d = daily_data['close']
        
        if len(c_h4) < 30 or len(c_d) < 20: return None
            
        avg_dollar_vol_4d = float(np.mean(v_h4[-24:] * c_h4[-24:]))
        if avg_dollar_vol_4d < CONFIG_MATRIX["HARD_LIQUIDITY_FLOOR_USD"]: return None

        atr = cls.calculate_wilder_atr(h_h4, l_h4, c_h4, period=14)
        atr_mult = (h_h4[-1] - l_h4[-1]) / atr[-2] if (len(atr) > 1 and atr[-2] > 0) else 1.0
        
        vol_percentile = float(stats.percentileofscore(v_h4[-120:], v_h4[-1]) / 100.0)
        
        h4_series = pd.Series(c_h4)
        bbw = (h4_series.rolling(20).std() * 4.0) / h4_series.rolling(20).mean()
        bbw_percentile = float(stats.percentileofscore(bbw.dropna().values[-200:], bbw.values[-1]) / 100.0)
        
        oi_change_pct = (oi_h4[-1] - oi_h4[-5]) / oi_h4[-5] if len(oi_h4) > 5 and oi_h4[-5] > 0 else 0.0
        
        # CORRECTED MATH: Fixed broadcasting mismatch
        prices = c_h4[-21:]
        asset_returns = np.diff(prices) / prices[:-1] 
        
        alpha_score_metric = 5.0
        if len(asset_returns) == len(btc_returns) and len(btc_returns) > 0:
            cov = np.cov(asset_returns, btc_returns)[0][1]
            beta = cov / np.var(btc_returns) if np.var(btc_returns) > 0 else 1.0
            alpha_score_metric = max(0.0, min(10.0, (np.mean(asset_returns) - (beta * np.mean(btc_returns))) * 1500.0))

        score_volume = vol_percentile * CONFIG_MATRIX["WEIGHT_VOLUME"]
        score_momentum = min((atr_mult / 3.5), 1.0) * CONFIG_MATRIX["WEIGHT_MOMENTUM"]
        score_coiling = (1.0 - bbw_percentile) * CONFIG_MATRIX["WEIGHT_COILING"]
        score_trend = min(max(0.0, ((c_d[-1] / pd.Series(c_d).ewm(span=21).mean().values[-1]) - 0.95) / 0.15), 1.0) * CONFIG_MATRIX["WEIGHT_TREND"]
        score_oi = CONFIG_MATRIX["WEIGHT_OI"] if oi_change_pct > 0.02 else CONFIG_MATRIX["WEIGHT_OI"] * 0.5
        score_alpha = (alpha_score_metric / 10.0) * CONFIG_MATRIX["WEIGHT_ALPHA"]
        
        composite_alpha_score = score_volume + score_momentum + score_coiling + score_trend + score_oi + CONFIG_MATRIX["WEIGHT_FUNDING"] + score_alpha
        
        if not ignore_resistance:
            if c_h4[-1] < (np.max(h_h4[-30:-1]) * CONFIG_MATRIX["RESISTANCE_THRESHOLD_PCT"]): return None
            if composite_alpha_score < args.alpha_threshold: return None
        
        return {
            "score": round(composite_alpha_score, 1),
            "close": float(c_h4[-1]),
            "rvol_pct": round(vol_percentile * 100, 1),
            "atr_mult": round(atr_mult, 2),
            "oi_growth": round(oi_change_pct * 100, 1),
            "timestamp": datetime.fromtimestamp(int(h4_data['time'][-1] / 1000), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        }

class ProductionDataPipeline:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.semaphore = asyncio.Semaphore(args.concurrency_limit)

    async def retrieve_active_universe(self, session: aiohttp.ClientSession) -> List[str]:
        try:
            async with session.get(f"{self.base_url}/api/v1/contract/detail", timeout=10) as response:
                return [item["symbol"] for item in (await response.json()).get("data", []) if item["settleCoin"] == "USDT" and item["state"] == 0]
        except: return []

    async def fetch_clean_kline(self, session: aiohttp.ClientSession, symbol: str, interval: str, limit: int = 200) -> Optional[dict]:
        url = f"{self.base_url}/api/v1/contract/kline/{symbol}"
        async with self.semaphore:
            try:
                async with session.get(url, params={"interval": interval, "limit": limit}, timeout=12) as response:
                    raw = await response.json()
                    d = raw.get("data") or {}
                    if not d or 'time' not in d: return None
                    return {k: np.array(v, dtype=np.float64) for k, v in d.items()}
            except: return None

async def main():
    pipeline = ProductionDataPipeline(args.api_url)
    async with aiohttp.ClientSession() as session:
        universe = await pipeline.retrieve_active_universe(session)
        btc_data = await pipeline.fetch_clean_kline(session, "BTC_USDT", "Hour4", limit=200)
        btc_returns = np.diff(btc_data['close'][-21:]) / btc_data['close'][-22:-1] if btc_data else np.zeros(20)

        tasks = [pipeline.fetch_clean_kline(session, s, "Hour4") for s in universe]
        h4_results = await asyncio.gather(*tasks)
        tasks_d1 = [pipeline.fetch_clean_kline(session, s, "Day1") for s in universe]
        d1_results = await asyncio.gather(*tasks_d1)

        validated = []
        for i, s in enumerate(universe):
            if h4_results[i] and d1_results[i]:
                res = VectorizedQuantEngine.evaluate_market_data(s, h4_results[i], d1_results[i], btc_returns)
                if res: 
                    res["symbol"] = s
                    validated.append(res)
        
        validated.sort(key=lambda x: x['score'], reverse=True)
        if validated:
            msg = f"MEXC Scanner Results:\n" + "\n".join([f"{r['symbol']} | Score: {r['score']} | Price: {r['close']:.4f}" for r in validated[:15]])
            send_telegram_message(msg)
            print(msg)

if __name__ == "__main__":
    asyncio.run(main())
