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
        print("Telegram credentials not found. Skipping Telegram transmission.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code != 200:
            print(f"Telegram API Error: {res.text}")
    except Exception as e:
        print(f"Telegram Connection Error: {e}")

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
        o_h4 = h4_data['open']
        oi_h4 = h4_data.get('open_interest', np.zeros_like(c_h4))
        c_d = daily_data['close']
        
        if len(c_h4) < 30 or len(c_d) < 20: 
            if args.debug:
                print(f"  └─ {symbol}: REJECTED - Insufficient history (h4:{len(c_h4)}, d1:{len(c_d)})")
            return None
            
        idx = -1
        avg_dollar_vol_4d = float(np.mean(v_h4[-24:] * c_h4[-24:]))
        
        if avg_dollar_vol_4d < CONFIG_MATRIX["HARD_LIQUIDITY_FLOOR_USD"]:
            if args.debug:
                print(f"  └─ {symbol}: REJECTED - Low liquidity (${avg_dollar_vol_4d:.0f})")
            return None

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
        beta = 1.0
        if len(asset_returns) == len(btc_returns) and len(btc_returns) > 0:
            try:
                cov = np.cov(asset_returns, btc_returns)[0][1]
                btc_var = np.var(btc_returns)
                beta = cov / btc_var if btc_var > 0 else 1.0
                alpha_raw = np.mean(asset_returns) - (beta * np.mean(btc_returns))
                alpha_score_metric = max(0.0, min(10.0, alpha_raw * 1500.0))
            except: 
                pass

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
        
        if args.debug:
            print(f"\n  OK {symbol} ANALYSIS:")
            print(f"    Score: {composite_alpha_score:.1f} | Vol%: {vol_percentile*100:.0f} | ATR: {atr_mult:.2f}x | OI: {oi_change_pct*100:+.1f}%")
            print(f"    Breakdown: Vol={score_volume:.1f} | Mom={score_momentum:.1f} | Coil={score_coiling:.1f} | Trend={score_trend:.1f} | OI={score_oi:.1f} | Alpha={score_alpha:.1f}")
            print(f"    Liquidity: ${avg_dollar_vol_4d:,.0f} | Beta: {beta:.2f}")

        if not ignore_resistance:
            recent_max_resistance = float(np.max(h_h4[-30:-1])) if len(h_h4) >= 30 else float(np.max(h_h4[:-1]))
            resistance_threshold = recent_max_resistance * CONFIG_MATRIX["RESISTANCE_THRESHOLD_PCT"]
            if c_h4[idx] < resistance_threshold:
                if args.debug:
                    print(f"  └─ {symbol}: REJECTED - Below resistance ({c_h4[idx]:.4f} < {resistance_threshold:.4f})")
                return None
        
        if not ignore_resistance and composite_alpha_score < args.alpha_threshold:
            if args.debug:
                print(f"  └─ {symbol}: REJECTED - Below threshold ({composite_alpha_score:.1f} < {args.alpha_threshold})")
            return None
        
        raw_time = int(h4_data['time'][idx])
        if raw_time > 1e11: 
            raw_time = raw_time // 1000
        
        return {
            "score": round(composite_alpha_score, 1),
            "close": float(c_h4[idx]),
            "rvol_pct": round(vol_percentile * 100, 1),
            "atr_mult": round(atr_mult, 2),
            "oi_growth": round(oi_change_pct * 100, 1),
            "timestamp": datetime.fromtimestamp(raw_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        }

class ProductionDataPipeline:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.semaphore = asyncio.Semaphore(args.concurrency_limit)
        self.api_timestamp_mode = None

    async def retrieve_active_universe(self, session: aiohttp.ClientSession) -> List[str]:
        url = f"{self.base_url}/api/v1/contract/detail"
        try:
            async with session.get(url, timeout=10) as response:
                payload = await response.json()
                data = payload.get("data") or []
                symbols = [str(item.get("symbol")).upper() for item in data 
                        if str(item.get("settleCoin")).upper() == "USDT" and int(item.get("state", 0)) == 0]
                if args.debug:
                    print(f"[UNIVERSE] Loaded {len(symbols)} active USDT perpetual pairs")
                return symbols
        except Exception as e:
            print(f"[ERROR] Failed to retrieve universe: {e}")
            return []

    async def fetch_clean_kline(self, session: aiohttp.ClientSession, symbol: str, interval: str, duration_days: int) -> Optional[dict]:
        """
        Adaptive Timestamp Engine with intelligent fallback:
        1. First tries cached working format (if known)
        2. Falls back to seconds, then milliseconds
        3. Caches successful format for future requests
        """
        url = f"{self.base_url}/api/v1/contract/kline/{symbol}"
        now_sec = int(time.time())
        
        time_formats = []
        if self.api_timestamp_mode:
            time_formats.append(self.api_timestamp_mode)
        time_formats.extend(['seconds', 'milliseconds'])
        
        for time_unit in time_formats:
            if time_unit == 'seconds':
                end_time = now_sec
                start_time = now_sec - (duration_days * 86400)
            else:
                end_time = now_sec * 1000
                start_time = (now_sec - (duration_days * 86400)) * 1000

            params = {"interval": interval, "start": start_time, "end": end_time}
            
            async with self.semaphore:
                for attempt in range(3):
                    try:
                        async with session.get(url, params=params, timeout=12) as response:
                            if response.status == 429:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            if response.status != 200:
                                continue
                            
                            raw = await response.json()
                            d = raw.get("data") or {}
                            
                            if d and 'time' in d and len(d['time']) > 0:
                                if not self.api_timestamp_mode:
                                    self.api_timestamp_mode = time_unit
                                    if args.debug:
                                        print(f"[API] Detected working timestamp format: {time_unit}")
                                
                                return {
                                    'time': np.array(d['time'], dtype=np.int64),
                                    'open': np.array(d['open'], dtype=np.float64),
                                    'high': np.array(d['high'], dtype=np.float64),
                                    'low': np.array(d['low'], dtype=np.float64),
                                    'close': np.array(d['close'], dtype=np.float64),
                                    'volume': np.array(d['volume'], dtype=np.float64),
                                    'open_interest': np.array(d.get('openInterest', np.zeros_like(d['close'])), dtype=np.float64)
                                }
                    except Exception as e:
                        if args.debug:
                            print(f"  └─ {symbol} ({interval}): Attempt {attempt+1} failed with {time_unit}")
                        pass
        
        if args.debug:
            print(f"  FAIL {symbol}: Unable to fetch {interval} data")
        return None

async def main():
    pipeline = ProductionDataPipeline(args.api_url)
    async with aiohttp.ClientSession() as session:
        print("=" * 70)
        print("MEXC 4H Breakout Scanner V3 | Adaptive Timestamp Engine")
        print("=" * 70)
        
        print("\n[INIT] Fetching active symbols from MEXC...")
        universe = await pipeline.retrieve_active_universe(session)
        if not universe:
            print("ERROR: Universe Empty. Verification failed.")
            return
        
        print(f"[INIT] Successfully loaded {len(universe)} symbols. Analyzing market cycles...\n")
        
        print("[DATA] Fetching BTC benchmark data...")
        btc_data = await pipeline.fetch_clean_kline(session, "BTC_USDT", "Hour4", duration_days=20)
        btc_returns = np.zeros(20)
        if btc_data and len(btc_data['close']) > 1:
            btc_closes = btc_data['close'][-21:]
            btc_returns = np.diff(btc_closes) / btc_closes[:-1] if len(btc_closes) > 1 else np.zeros(20)
            print(f"  OK: BTC ready ({len(btc_data['close'])} candles, avg_return={np.mean(btc_returns)*100:+.3f}%)")
        else:
            print("  WARNING: BTC data unavailable. Using baseline alpha.")

        print(f"\n[SCAN] Analyzing {len(universe)} assets...")
        tasks = []
        for sym in universe:
            async def scan(s=sym):
                h4 = await pipeline.fetch_clean_kline(session, s, "Hour4", duration_days=20)
                d1 = await pipeline.fetch_clean_kline(session, s, "Day1", duration_days=45)
                if h4 and d1:
                    res = VectorizedQuantEngine.evaluate_market_data(s, h4, d1, btc_returns, ignore_resistance=False)
                    if res: 
                        res["symbol"] = s
                        return res
                return None
            tasks.append(scan())
            
        raw_results = await asyncio.gather(*tasks)
        validated_signals = [res for res in raw_results if res is not None]
        
        is_fallback = False
        if not validated_signals:
            print(f"  └─ No signals in Alpha Mode. Engaging Fallback Scanner...\n")
            is_fallback = True
            fallback_tasks = []
            for sym in universe:
                async def scan_fb(s=sym):
                    h4 = await pipeline.fetch_clean_kline(session, s, "Hour4", duration_days=20)
                    d1 = await pipeline.fetch_clean_kline(session, s, "Day1", duration_days=45)
                    if h4 and d1:
                        res = VectorizedQuantEngine.evaluate_market_data(s, h4, d1, btc_returns, ignore_resistance=True)
                        if res: 
                            res["symbol"] = s
                            return res
                    return None
                fallback_tasks.append(scan_fb())
            raw_results = await asyncio.gather(*fallback_tasks)
            validated_signals = [res for res in raw_results if res is not None]

        validated_signals.sort(key=lambda x: x['score'], reverse=True)
        
        if validated_signals:
            mode_title = "Fallback Mode (No Resistance Filter)" if is_fallback else "Alpha Mode"
            msg_lines = [f"MEXC 4H Breakout Scanner V3 ({mode_title})\n"]
            msg_lines.append(f"Scan Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
            
            for rank, sig in enumerate(validated_signals[:15], 1):
                line = f"{rank}. {sig['symbol']} | Score: {sig['score']} | Price: {sig['close']:.6f}\n"
                line += f"   RVOL: {sig['rvol_pct']}% | ATR: {sig['atr_mult']}x | OI: {sig['oi_growth']:+.1f}%\n"
                msg_lines.append(line)
            
            full_message = "\n".join(msg_lines)
            print("\n" + "=" * 70)
            print("SCAN COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(full_message)
            send_telegram_message(full_message)
        else:
            print("\n" + "=" * 70)
            print("INFO: Scan completed. No assets matched the baseline strategy filters.")
            print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
