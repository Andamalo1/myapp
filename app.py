from pathlib import Path
from flask import Flask, render_template, request, jsonify, url_for
from flask_cors import CORS
import time, logging
import yfinance as yf
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("portfolio_app")

TRADING_DAYS = 252

def normalize_tickers(tickers_raw):
    if not tickers_raw: return []
    if isinstance(tickers_raw, str):
        return [s.strip().upper() for s in tickers_raw.split(',') if s.strip()]
    if isinstance(tickers_raw, (list, tuple)):
        return [str(s).strip().upper() for s in tickers_raw if str(s).strip()]
    return []

def get_ticker_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'currency': info.get('currency', 'USD'),
            'exchange': info.get('exchange', 'Unknown'),
            'longName': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
        }
    except Exception as e:
        logger.warning(f"Cannot get info for {ticker}: {e}")
        return {
            'currency': 'USD',
            'exchange': 'Unknown',
            'longName': ticker,
            'sector': 'N/A',
        }

def download_prices_per_ticker(ticker, start=None, end=None):
    try:
        df = None
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, threads=False)
        except Exception as e:
            logger.warning(f"Download failed for {ticker}: {e}")
            df = None
        
        if df is None or df.empty:
            return None, None
        
        if 'Adj Close' in df.columns:
            s = df['Adj Close'].dropna().to_frame()
        elif 'Close' in df.columns:
            s = df['Close'].dropna().to_frame(name='Adj Close')
        else:
            cols = list(df.columns)
            if cols:
                s = df[cols[-1]].dropna().to_frame(name='Adj Close')
            else:
                return None, None
        
        s.index = pd.to_datetime(s.index)
        s.sort_index(inplace=True)
        
        info = get_ticker_info(ticker)
        
        return s[['Adj Close']], info
    except Exception as e:
        logger.exception("download_prices_per_ticker error %s: %s", ticker, e)
        return None, None

def download_prices_bulk(tickers, start=None, end=None):
    result = {}
    tickers = [t for t in tickers if t]
    if not tickers:
        return result
    
    # Prova download bulk
    try:
        df_all = yf.download(tickers, start=start, end=end, progress=False, threads=False)
        if isinstance(df_all.columns, pd.MultiIndex):
            for tk in tickers:
                col = (tk, 'Adj Close')
                if col in df_all.columns:
                    s = df_all[col].dropna().to_frame(name='Adj Close')
                    s.index = pd.to_datetime(s.index)
                    info = get_ticker_info(tk)
                    result[tk] = {'df': s, 'info': info}
    except Exception as e:
        logger.warning(f"Bulk download failed: {e}")
    
    # Per i ticker mancanti, scarica singolarmente
    missing = [tk for tk in tickers if tk not in result]
    for tk in missing:
        s, info = download_prices_per_ticker(tk, start=start, end=end)
        if s is not None and not s.empty:
            result[tk] = {'df': s, 'info': info}
    
    return result

def to_json_prices_dict(dfs):
    out = {}
    for tk, v in dfs.items():
        df = v.get('df') if isinstance(v, dict) else v
        info = v.get('info') if isinstance(v, dict) else get_ticker_info(tk)
        df_sorted = df.sort_index()
        out[tk] = {
            'dates': [d.strftime('%Y-%m-%d') for d in df_sorted.index],
            'adjclose': [float(x) for x in df_sorted['Adj Close'].round(6).tolist()],
            'currency': info.get('currency', 'USD'),
            'exchange': info.get('exchange', 'Unknown'),
            'longName': info.get('longName', tk),
            'sector': info.get('sector', 'N/A'),
        }
    return out

def compute_returns_and_metrics(series, rf=0.0):
    arr = series.dropna().values
    if len(arr) < 2:
        return {
            'cagr': 0.0,
            'ann_vol': 0.0,
            'sharpe': None,
            'max_drawdown': 0.0,
            'downside_ann': 0.0,
            'sortino': None,
            'var95': None,
            'cvar95': None
        }
    
    simple = arr[1:] / arr[:-1] - 1.0
    logrets = np.log(arr[1:] / arr[:-1])
    days = len(simple)
    years = max(days / TRADING_DAYS, 1e-9)
    
    cagr = (arr[-1] / arr[0]) ** (1.0 / years) - 1.0 if arr[0] > 0 else 0.0
    ann_vol = float(np.std(logrets, ddof=1) * np.sqrt(TRADING_DAYS)) if len(logrets) > 1 else 0.0
    
    below = simple[simple < 0]
    downside = float(np.sqrt(np.mean((below - 0.0) ** 2)) * np.sqrt(TRADING_DAYS)) if len(below) > 0 else 0.0
    
    sharpe = float((cagr - rf) / ann_vol) if ann_vol > 0 else None
    sortino = float((cagr - rf) / downside) if downside > 0 else None
    
    peak = arr[0]
    md = 0.0
    for v in arr:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak != 0 else 0.0
        if dd > md:
            md = dd
    
    if len(simple) > 0:
        var95 = float(np.percentile(-simple, 95))
        losses = [-r for r in simple if -r >= var95]
        cvar95 = float(np.mean(losses)) if losses else None
    else:
        var95 = None
        cvar95 = None
    
    return {
        'cagr': cagr,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': md,
        'downside_ann': downside,
        'sortino': sortino,
        'var95': var95,
        'cvar95': cvar95
    }

@app.route('/')
def index():
    css_url = url_for('static', filename='style.css') + f"?v={int(time.time())}"
    return render_template("applicativo_web.html", css_url=css_url)

@app.route('/api/fetch_prices', methods=['POST'])
def api_fetch_prices():
    payload = request.json or {}
    raw = payload.get('tickers', [])
    start = payload.get('start', None)
    end = payload.get('end', None)
    tickers = normalize_tickers(raw)
    
    if not tickers:
        return jsonify({'error': 'Nessun ticker fornito'}), 400
    
    logger.info(f"Fetching tickers: {tickers}")
    fetched = download_prices_bulk(tickers, start=start, end=end)
    failed = [t for t in tickers if t not in fetched]
    
    if not fetched:
        return jsonify({'error': 'Nessun dato valido per i ticker richiesti', 'failed_tickers': failed}), 400
    
    resp = {'prices': to_json_prices_dict(fetched)}
    if failed:
        resp['failed_tickers'] = failed
    
    return jsonify(resp)

@app.route('/api/simulate_from_initial', methods=['POST'])
def api_simulate_from_initial():
    payload = request.json or {}
    prices = payload.get('prices', {})
    weights = payload.get('weights', None)
    initialCapital = float(payload.get('initialCapital', 10000.0))
    rf = float(payload.get('rf', 0.0))
    
    if not prices:
        return jsonify({'error': 'Prezzi richiesti'}), 400
    
    if not weights:
        return jsonify({'error': 'Pesi richiesti'}), 400
    
    keys = list(prices.keys())
    
    try:
        w = np.array(weights, dtype=float)
    except Exception as e:
        return jsonify({'error': 'Pesi non validi'}), 400
    
    if w.size != len(keys):
        return jsonify({'error': f'Weights mismatch'}), 400
    
    ssum = w.sum() or 1.0
    w = w / ssum
    
    try:
        all_dates = set()
        for tk in keys:
            all_dates.update(pd.to_datetime(prices[tk].get('dates', [])))
        all_dates = sorted(list(all_dates))
        
        price_data = {}
        for tk in keys:
            dates = pd.to_datetime(prices[tk].get('dates', []))
            vals = prices[tk].get('adjclose', [])
            df_tk = pd.DataFrame({'price': vals}, index=dates)
            price_data[tk] = df_tk
        
        df_all = pd.DataFrame(index=pd.DatetimeIndex(all_dates))
        for tk in keys:
            df_all[tk] = price_data[tk]['price']
        
        df_all = df_all.fillna(method='ffill')
        df_all = df_all.dropna(how='all')
        
        # ===== FISSO: primo giorno con TUTTI i dati =====
        valid_idx = None
        actual_date = None
        for i, date in enumerate(df_all.index):
            all_valid = True
            for tk in keys:
                val = df_all[tk].iloc[i]
                if pd.isna(val) or val == 0:
                    all_valid = False
                    break
            if all_valid:
                valid_idx = i
                actual_date = date
                break
        
        if valid_idx is None:
            return jsonify({'error': 'Nessun giorno con dati disponibili per tutti i titoli'}), 400
        
        S0_fissa = np.array([df_all[tk].iloc[valid_idx] for tk in keys], dtype=float)
        
        if np.any(np.isnan(S0_fissa)) or np.any(S0_fissa == 0):
            return jsonify({'error': 'Prezzi non disponibili'}), 400
        
        logger.info(f"FISSO - Data: {actual_date.strftime('%Y-%m-%d')}")
        
        shares_fissa = (w * initialCapital) / S0_fissa
        price_matrix = np.vstack([df_all[tk].iloc[valid_idx:].fillna(method='ffill').values for tk in keys]).T
        pv_fissa = (price_matrix * shares_fissa).sum(axis=1)
        
        dates_fissa = [d.strftime('%Y-%m-%d') for d in df_all.index[valid_idx:]]
        values_fissa = pv_fissa.tolist()
        
        clean_dates_fissa = []
        clean_values_fissa = []
        for d, v in zip(dates_fissa, values_fissa):
            try:
                vnum = float(v)
                if np.isfinite(vnum) and vnum > 0:
                    clean_dates_fissa.append(d)
                    clean_values_fissa.append(vnum)
            except Exception:
                continue
        
        series_fissa = pd.Series(clean_values_fissa)
        metrics_fissa = compute_returns_and_metrics(series_fissa, rf=rf)
        total_gain_fissa = clean_values_fissa[-1] - initialCapital if clean_values_fissa else 0
        total_return_pct_fissa = ((clean_values_fissa[-1] / initialCapital) - 1) * 100 if clean_values_fissa else 0
        
        # ===== PROGRESSIVA =====
        first_dates = {}
        for tk in keys:
            dates = pd.to_datetime(prices[tk].get('dates', []))
            first_dates[tk] = dates[0]
        
        sorted_tickers = sorted(keys, key=lambda tk: first_dates[tk])
        
        portfolio_values_prog = []
        portfolio_dates_prog = []
        holdings = {tk: 0.0 for tk in keys}
        
        for date_idx, date in enumerate(df_all.index):
            for i, tk in enumerate(sorted_tickers):
                if holdings[tk] == 0.0 and date >= first_dates[tk]:
                    price_today = df_all[tk].iloc[date_idx]
                    if not pd.isna(price_today) and price_today > 0:
                        capital_to_invest = w[keys.index(tk)] * initialCapital
                        holdings[tk] = capital_to_invest / price_today
            
            portfolio_value = 0.0
            for tk in keys:
                price_today = df_all[tk].iloc[date_idx]
                if holdings[tk] > 0 and not pd.isna(price_today):
                    portfolio_value += holdings[tk] * price_today
            
            if portfolio_value > 0:
                portfolio_values_prog.append(portfolio_value)
                portfolio_dates_prog.append(date.strftime('%Y-%m-%d'))
        
        series_prog = pd.Series(portfolio_values_prog)
        metrics_prog = compute_returns_and_metrics(series_prog, rf=rf)
        total_gain_prog = portfolio_values_prog[-1] - initialCapital if portfolio_values_prog else 0
        total_return_pct_prog = ((portfolio_values_prog[-1] / initialCapital) - 1) * 100 if portfolio_values_prog else 0
        
        return jsonify({
            'fissa': {
                'dates': clean_dates_fissa,
                'values': clean_values_fissa,
                'portfolio': metrics_fissa,
                'stats': {
                    'initial_capital': initialCapital,
                    'final_value': clean_values_fissa[-1] if clean_values_fissa else 0,
                    'total_gain': total_gain_fissa,
                    'total_return_pct': total_return_pct_fissa,
                    'fixed_date': actual_date.strftime('%Y-%m-%d'),
                }
            },
            'progressiva': {
                'dates': portfolio_dates_prog,
                'values': portfolio_values_prog,
                'portfolio': metrics_prog,
                'stats': {
                    'initial_capital': initialCapital,
                    'final_value': portfolio_values_prog[-1] if portfolio_values_prog else 0,
                    'total_gain': total_gain_prog,
                    'total_return_pct': total_return_pct_prog,
                }
            }
        })
        
    except Exception as e:
        logger.exception(f"Simulation error: {e}")
        return jsonify({'error': f'Errore simulazione: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)