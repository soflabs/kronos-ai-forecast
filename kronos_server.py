"""
Kronos AI Forecast Microservice — Render Deployment
Merlijn Signaal Labo — Camelot Finance

Endpoint: GET /forecast?symbol=BTCUSDT
Health:   GET /health
"""

import os
import requests
from flask import Flask, jsonify, request
import pandas as pd
import time

app = Flask(__name__)

# ── Model laden (eenmalig bij opstart) ───────────────────────────────
print("[Kronos] Model laden...")
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    tokenizer  = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model_k    = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor  = KronosPredictor(model_k, tokenizer, max_context=512)
    print("[Kronos] Model geladen")
    KRONOS_OK = True
except Exception as e:
    print(f"[Kronos] Model fout: {e}")
    KRONOS_OK = False

# ── Cache: max 1 forecast per symbol per 30 min ─────────────────────
_cache = {}
CACHE_TTL = 1800

# Symbool mapping naar CryptoCompare formaat
SYMBOL_MAP = {
    'HBARUSDT': ('HBAR', 'USDT'),
    'XRPUSDT':  ('XRP', 'USDT'),
    'VETUSDT':  ('VET', 'USDT'),
    'BTCUSDT':  ('BTC', 'USDT'),
    'PAXGUSDT': ('PAXG', 'USDT'),
    'XAGUSD':   ('PAXG', 'USDT'),
    'XAUUSD':   ('PAXG', 'USDT'),
}

def get_cc_pair(symbol):
    """Vertaal symbol naar CryptoCompare fsym/tsym."""
    s = symbol.upper()
    if s in SYMBOL_MAP:
        return SYMBOL_MAP[s]
    # Probeer automatisch te splitsen
    if s.endswith('USDT'):
        return (s[:-4], 'USDT')
    if s.endswith('USD'):
        return (s[:-3], 'USD')
    return (s, 'USDT')


def fetch_ohlcv(symbol, lookback=450):
    """Haal OHLCV data op via CryptoCompare (geen geo-restricties)."""
    fsym, tsym = get_cc_pair(symbol)
    errors = []

    # CryptoCompare: 4h candles = hourly met aggregate=4
    # Max 2000 per call, we vragen lookback candles
    url = f'https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym={tsym}&limit={lookback * 4}&aggregate=1'
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if data.get('Response') == 'Success' and data.get('Data', {}).get('Data'):
            raw = data['Data']['Data']
            # Groepeer per 4 uur (OHLCV aggregatie)
            candles = []
            for i in range(0, len(raw) - 3, 4):
                chunk = raw[i:i+4]
                if len(chunk) < 4:
                    break
                candles.append({
                    'timestamp': chunk[0]['time'] * 1000,
                    'open': chunk[0]['open'],
                    'high': max(c['high'] for c in chunk),
                    'low': min(c['low'] for c in chunk),
                    'close': chunk[-1]['close'],
                    'volume': sum(c['volumefrom'] for c in chunk),
                })
            if len(candles) >= 50:
                df = pd.DataFrame(candles)
                df['timestamps'] = pd.to_datetime(df['timestamp'], unit='ms')
                print(f"[Kronos] OHLCV via CryptoCompare: {fsym}/{tsym} ({len(df)} 4h candles)")
                return df
            else:
                errors.append(f"CryptoCompare: te weinig data ({len(candles)} candles)")
        else:
            errors.append(f"CryptoCompare: {data.get('Message', 'onbekende fout')}")
    except Exception as e:
        errors.append(f"CryptoCompare: {e}")

    # Fallback: Binance data-api (proxy, soms minder geo-restrictief)
    binance_urls = [
        f'https://data-api.binance.vision/api/v3/klines?symbol={symbol.upper()}&interval=4h&limit={lookback}',
        f'https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval=4h&limit={lookback}',
    ]
    for burl in binance_urls:
        try:
            resp = requests.get(burl, timeout=10)
            if resp.status_code == 200:
                raw = resp.json()
                if isinstance(raw, list) and len(raw) > 0:
                    df = pd.DataFrame(raw, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_vol', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ])
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
                    df['timestamp'] = df['timestamp'].astype(int)
                    df['timestamps'] = pd.to_datetime(df['timestamp'], unit='ms')
                    src = 'Binance-data-api' if 'data-api' in burl else 'Binance'
                    print(f"[Kronos] OHLCV via {src}: {symbol} ({len(df)} candles)")
                    return df
        except Exception as e:
            errors.append(f"Binance: {e}")

    raise Exception(f"Alle data bronnen gefaald voor {symbol}: {'; '.join(errors)}")


def fetch_multi_exchange_price(symbol):
    """Haal huidige prijs op via CryptoCompare (multi-exchange gemiddelde)."""
    fsym, tsym = get_cc_pair(symbol)
    try:
        url = f'https://min-api.cryptocompare.com/data/pricemultifull?fsyms={fsym}&tsyms={tsym}'
        resp = requests.get(url, timeout=10)
        data = resp.json()
        raw = data.get('RAW', {}).get(fsym, {}).get(tsym, {})
        if not raw:
            return None

        price = float(raw.get('PRICE', 0))
        high24 = float(raw.get('HIGH24HOUR', 0))
        low24 = float(raw.get('LOW24HOUR', 0))
        volume = float(raw.get('VOLUME24HOUR', 0))
        market = raw.get('LASTMARKET', 'unknown')

        return {
            'prices': {market: price},
            'avg': round(price, 6),
            'spread': round(high24 - low24, 6),
            'spread_pct': round((high24 - low24) / price * 100, 3) if price > 0 else 0,
            'high24': round(high24, 6),
            'low24': round(low24, 6),
            'volume24': round(volume, 2),
        }
    except Exception as e:
        print(f"[Kronos] Prijs fout {symbol}: {e}")
        return None


def compute_forecast(symbol):
    now = time.time()

    if symbol in _cache and now - _cache[symbol]['ts'] < CACHE_TTL:
        return _cache[symbol]['data']

    if not KRONOS_OK:
        return {'symbol': symbol, 'direction': 'neutral', 'pct': 0.0, 'score': 0}

    try:
        df = fetch_ohlcv(symbol)

        lookback = 400
        pred_len = 24

        if len(df) < lookback + pred_len:
            return {'symbol': symbol, 'direction': 'neutral', 'pct': 0.0, 'score': 0,
                    'error': f'Onvoldoende data: {len(df)} rijen'}

        df = df.iloc[-(lookback + pred_len):].reset_index(drop=True)

        x_df        = df.iloc[:lookback][['open', 'high', 'low', 'close', 'volume']]
        x_timestamp = df.iloc[:lookback]['timestamps']
        y_timestamp = df.iloc[lookback:lookback + pred_len]['timestamps']

        pred_df = predictor.predict(
            df           = x_df,
            x_timestamp  = x_timestamp,
            y_timestamp  = y_timestamp,
            pred_len     = pred_len,
            T            = 1.0,
            top_p        = 0.9,
            sample_count = 3
        )

        current_close  = df.iloc[lookback - 1]['close']
        forecast_close = pred_df['close'].iloc[-1]
        pct_change     = (forecast_close - current_close) / current_close * 100

        raw_score = pct_change * 1.5
        score     = int(max(-15, min(15, round(raw_score))))

        if pct_change > 1.5:
            direction = 'bullish'
        elif pct_change < -1.5:
            direction = 'bearish'
        else:
            direction = 'neutral'

        # Multi-exchange prijsvergelijking
        multi = fetch_multi_exchange_price(symbol)

        result = {
            'symbol':    symbol,
            'direction': direction,
            'pct':       round(pct_change, 2),
            'score':     score,
            'forecast':  round(float(forecast_close), 6),
            'current':   round(float(current_close), 6),
            'exchanges': multi,
        }

        _cache[symbol] = {'ts': now, 'data': result}
        print(f"[Kronos] {symbol}: {direction} {pct_change:+.2f}% score {score:+d}")
        return result

    except Exception as e:
        print(f"[Kronos] Fout {symbol}: {e}")
        return {'symbol': symbol, 'direction': 'neutral', 'pct': 0.0, 'score': 0, 'error': str(e)}


# ── Routes ───────────────────────────────────────────────────────────

@app.route('/forecast')
def forecast():
    symbol = request.args.get('symbol', 'BTCUSDT').upper()
    result = compute_forecast(symbol)
    return jsonify(result)

@app.route('/prices')
def prices():
    symbol = request.args.get('symbol', 'BTCUSDT').upper()
    result = fetch_multi_exchange_price(symbol)
    if result:
        return jsonify({'symbol': symbol, **result})
    return jsonify({'symbol': symbol, 'error': 'Geen prijzen beschikbaar'})

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'kronos_loaded': KRONOS_OK, 'cached': list(_cache.keys()),
                    'data_source': 'CryptoCompare + Binance fallback'})

@app.route('/cache/clear')
def clear_cache():
    _cache.clear()
    return jsonify({'status': 'cache cleared'})

@app.route('/')
def index():
    return jsonify({'service': 'Kronos AI Forecast', 'status': 'ok',
                    'endpoints': ['/forecast?symbol=BTCUSDT', '/prices?symbol=BTCUSDT', '/health']})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
