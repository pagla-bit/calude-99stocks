import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import time

# Market cap thresholds (in billions)
SMALL_CAP_MAX = 2.0
MID_CAP_MIN = 2.0
MID_CAP_MAX = 10.0
LARGE_CAP_MIN = 10.0

# Initialize session state
if 'applied_short' not in st.session_state:
    st.session_state.applied_short = 5.0
if 'applied_mid' not in st.session_state:
    st.session_state.applied_mid = 10.0
if 'applied_long' not in st.session_state:
    st.session_state.applied_long = 30.0
if 'stock_universe' not in st.session_state:
    st.session_state.stock_universe = None
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_sp500_tickers():
    """Fetch S&P 500 ticker list from Wikipedia"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        df = pd.read_html(str(table))[0]
        tickers = df['Symbol'].str.replace('.', '-').tolist()
        return tickers
    except Exception as e:
        st.error(f"Failed to fetch S&P 500 tickers: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def get_sp400_tickers():
    """Fetch S&P 400 (Mid Cap) ticker list from Wikipedia"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        df = pd.read_html(str(table))[0]
        tickers = df['Symbol'].str.replace('.', '-').tolist()
        return tickers
    except Exception as e:
        st.error(f"Failed to fetch S&P 400 tickers: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def get_russell2000_tickers():
    """Fetch Russell 2000 ticker list from iShares IWM holdings"""
    try:
        # IWM is the Russell 2000 ETF
        url = 'https://www.ishares.com/us/products/239710/ishares-russell-2000-etf'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        # Alternative: Use a predefined list or CSV
        # For now, we'll use a simpler approach - get from yfinance screener
        st.warning("Russell 2000 full list unavailable. Using S&P 600 small cap as proxy.")
        
        # Fetch S&P 600 Small Cap instead
        url_sp600 = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
        response = requests.get(url_sp600, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable sortable'})
        df = pd.read_html(str(table))[0]
        tickers = df['Symbol'].str.replace('.', '-').tolist()
        return tickers[:500]  # Limit to top 500
    except Exception as e:
        st.error(f"Failed to fetch Russell 2000 tickers: {str(e)}")
        return []

def fetch_tickers_by_index(index_name):
    """Fetch tickers based on index selection"""
    if index_name == "S&P 500":
        return get_sp500_tickers()
    elif index_name == "S&P 400":
        return get_sp400_tickers()
    elif index_name == "Russell 2000":
        return get_russell2000_tickers()
    return []

@st.cache_data(ttl=600)
def fetch_stock_data(ticker, period='6mo'):
    """Fetch stock data and info with proper error handling"""
    try:
        data = yf.download(ticker, period=period, progress=False)
        if not data.empty and isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        info = yf.Ticker(ticker).info
        return data, info
    except Exception as e:
        return pd.DataFrame(), {}

def get_market_cap_billions(info):
    """Extract market cap in billions"""
    try:
        market_cap = info.get('marketCap', 0)
        if market_cap:
            return market_cap / 1e9
        return 0
    except:
        return 0

def categorize_by_market_cap(ticker, info):
    """Categorize stock by market cap"""
    market_cap = get_market_cap_billions(info)
    
    if market_cap == 0:
        return None
    elif market_cap < SMALL_CAP_MAX:
        return 'small'
    elif MID_CAP_MIN <= market_cap <= MID_CAP_MAX:
        return 'mid'
    elif market_cap > LARGE_CAP_MIN:
        return 'large'
    return None

def safe_float(value, default=0.0):
    """Convert pandas Series or any value to float safely"""
    try:
        if isinstance(value, pd.Series):
            return float(value.iloc[-1]) if len(value) > 0 else default
        return float(value)
    except (ValueError, TypeError, IndexError):
        return default

def calculate_rsi(data, window=14):
    """Calculate RSI with proper validation"""
    if len(data) < window + 1:
        return 50.0
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return safe_float(rsi, 50.0)

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD histogram with proper validation"""
    if len(data) < slow + signal:
        return 0.0
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return safe_float(histogram, 0.0)

def calculate_volatility(data, window=14):
    """Calculate annualized volatility"""
    if len(data) < window + 1:
        return 0.0
    returns = data['Close'].pct_change()
    vol = returns.std() * np.sqrt(252)
    return safe_float(vol, 0.0)

def calculate_momentum(data, lookback=20):
    """Calculate price momentum (rate of change)"""
    if len(data) < lookback + 1:
        return 0.0
    current = safe_float(data['Close'].iloc[-1])
    past = safe_float(data['Close'].iloc[-lookback])
    if past == 0:
        return 0.0
    return ((current / past) - 1) * 100

def calculate_volume_surge(data, window=20):
    """Calculate current volume vs average volume"""
    if len(data) < window + 1:
        return 1.0
    avg_vol = safe_float(data['Volume'].tail(window).mean())
    current_vol = safe_float(data['Volume'].iloc[-1])
    if avg_vol == 0:
        return 1.0
    return current_vol / avg_vol

def get_profit_potential(data, info, horizon_type='short'):
    """
    Unified profit potential calculation based on multiple factors
    
    Parameters:
    - data: Price/volume dataframe
    - info: Stock info dictionary
    - horizon_type: 'short' (1-3 days), 'mid' (1-2 weeks), 'long' (1-3 months)
    
    Returns:
    - potential: Expected profit potential percentage
    - metrics: Dictionary of contributing metrics for scoring
    """
    if data.empty or len(data) < 20:
        return 0.0, {}
    
    current_price = safe_float(data['Close'].iloc[-1])
    if current_price == 0:
        return 0.0, {}
    
    # Calculate base metrics
    rsi = calculate_rsi(data, window=14)
    macd = calculate_macd(data)
    volatility = calculate_volatility(data, window=14)
    volume_surge = calculate_volume_surge(data, window=20)
    
    # Horizon-specific calculations
    if horizon_type == 'short':
        lookback_days = 5
        if len(data) < lookback_days:
            return 0.0, {}
        
        recent_high = safe_float(data['High'].tail(lookback_days).max())
        momentum = calculate_momentum(data, lookback=5)
        
        upside_to_high = ((recent_high - current_price) / current_price * 100) if current_price > 0 else 0
        
        potential = (
            upside_to_high * 0.4 +
            momentum * 0.3 +
            (volume_surge - 1) * 5 * 0.2 +
            (50 - abs(rsi - 50)) / 10 * 0.1
        )
        
        metrics = {
            'RSI': rsi,
            'Momentum_5D': momentum,
            'Vol_Surge': volume_surge,
            'Volatility': volatility
        }
        
    elif horizon_type == 'mid':
        lookback_days = 14
        if len(data) < lookback_days:
            return 0.0, {}
        
        momentum = calculate_momentum(data, lookback=14)
        macd_signal = (macd / volatility * 10) if volatility > 0.01 else 0
        
        rsi_signal = 0
        if rsi < 30:
            rsi_signal = (30 - rsi) * 0.5
        elif rsi > 70:
            rsi_signal = (70 - rsi) * 0.3
        
        potential = (
            momentum * 0.4 +
            macd_signal * 0.3 +
            rsi_signal * 0.2 +
            (volume_surge - 1) * 3 * 0.1
        )
        
        metrics = {
            'RSI': rsi,
            'MACD': macd,
            'Momentum_14D': momentum,
            'Vol_Surge': volume_surge
        }
        
    else:  # long
        lookback_days = 60
        if len(data) < lookback_days:
            return 0.0, {}
        
        momentum_60 = calculate_momentum(data, lookback=60)
        momentum_20 = calculate_momentum(data, lookback=20)
        beta = safe_float(info.get('beta', 1.0), 1.0)
        
        trend_acceleration = momentum_20 - momentum_60
        beta_factor = 1.0 + (beta - 1.0) * 0.5
        
        potential = (
            momentum_60 * 0.5 * beta_factor +
            trend_acceleration * 0.3 +
            (50 - abs(rsi - 50)) / 5 * 0.2
        )
        
        metrics = {
            'Beta': beta,
            'Momentum_60D': momentum_60,
            'Momentum_20D': momentum_20,
            'RSI': rsi
        }
    
    return safe_float(potential, 0.0), metrics

def calculate_composite_score(potential, metrics, horizon_type):
    """Calculate composite score incorporating potential and additional metrics"""
    if not metrics:
        return potential
    
    score = potential * 0.6
    
    if horizon_type == 'short':
        vol_surge = metrics.get('Vol_Surge', 1.0)
        rsi = metrics.get('RSI', 50.0)
        
        score += (vol_surge - 1) * 10 * 0.2
        if 30 <= rsi <= 45:
            score += 5 * 0.2
        elif 55 <= rsi <= 70:
            score += 3 * 0.2
            
    elif horizon_type == 'mid':
        macd = metrics.get('MACD', 0.0)
        momentum = metrics.get('Momentum_14D', 0.0)
        
        if macd > 0 and momentum > 0:
            score += 5 * 0.4
        
    else:  # long
        momentum_60 = metrics.get('Momentum_60D', 0.0)
        beta = metrics.get('Beta', 1.0)
        
        if momentum_60 > 10:
            score += 10 * 0.3
        if beta > 1.2:
            score += 5 * 0.1
    
    return safe_float(score, 0.0)

def scan_stocks_from_index(index_list, progress_bar=None, status_text=None):
    """
    Scan stocks from index and categorize by market cap
    Returns dict with 'small', 'mid', 'large' categories
    """
    categorized = {'small': [], 'mid': [], 'large': []}
    total = len(index_list)
    
    for idx, ticker in enumerate(index_list):
        if progress_bar:
            progress_bar.progress((idx + 1) / total)
        if status_text:
            status_text.text(f"Scanning {ticker}... ({idx + 1}/{total})")
        
        data, info = fetch_stock_data(ticker, period='3mo')
        
        if data.empty or len(data) < 20:
            continue
        
        # Get average volume
        avg_volume = safe_float(data['Volume'].mean())
        
        # Categorize by market cap
        category = categorize_by_market_cap(ticker, info)
        
        if category:
            categorized[category].append({
                'ticker': ticker,
                'data': data,
                'info': info,
                'avg_volume': avg_volume,
                'market_cap': get_market_cap_billions(info)
            })
    
    # Sort by volume and take top 500 from each category
    for category in categorized:
        categorized[category] = sorted(
            categorized[category], 
            key=lambda x: x['avg_volume'], 
            reverse=True
        )[:500]
    
    return categorized

def scan_stocks_by_screener(market_cap_filter, progress_bar=None, status_text=None):
    """
    Scan stocks using Yahoo Finance screener criteria
    This is a placeholder - Yahoo screener API is limited
    """
    st.warning("Yahoo screener option coming soon. Using index-based scan for now.")
    
    # For now, fall back to index-based scan
    if market_cap_filter == 'small':
        index_list = get_russell2000_tickers()
    elif market_cap_filter == 'mid':
        index_list = get_sp400_tickers()
    else:
        index_list = get_sp500_tickers()
    
    return scan_stocks_from_index(index_list, progress_bar, status_text)

def build_opportunity_table(stock_list, horizon_type, target_threshold):
    """Build dataframe of top opportunities"""
    df_list = []
    
    for stock_info in stock_list:
        ticker = stock_info['ticker']
        data = stock_info['data']
        info = stock_info['info']
        market_cap = stock_info['market_cap']
        
        # Calculate potential and metrics
        potential, metrics = get_profit_potential(data, info, horizon_type)
        
        # Skip if below threshold
        if potential < target_threshold:
            continue
        
        # Calculate composite score
        score = calculate_composite_score(potential, metrics, horizon_type)
        
        current_price = safe_float(data['Close'].iloc[-1])
        
        # Get primary additional metric
        if horizon_type == 'short':
            add_metric_name = 'Vol Surge'
            add_metric_value = metrics.get('Vol_Surge', 0.0)
        elif horizon_type == 'mid':
            add_metric_name = 'RSI'
            add_metric_value = metrics.get('RSI', 50.0)
        else:
            add_metric_name = 'Beta'
            add_metric_value = metrics.get('Beta', 1.0)
        
        df_list.append({
            'Ticker': ticker,
            'Name': info.get('longName', ticker)[:30],
            'Price': round(current_price, 2),
            'Mkt Cap ($B)': round(market_cap, 2),
            'Potential %': round(potential, 2),
            'Score': round(score, 2),
            add_metric_name: round(add_metric_value, 2)
        })
    
    if df_list:
        df = pd.DataFrame(df_list).sort_values('Score', ascending=False)
        return df.head(33)
    
    return pd.DataFrame()

# Streamlit App
st.set_page_config(page_title="Dynamic Stock Scanner", layout="wide")
st.title("🔍 Dynamic Stock Opportunity Scanner")
st.caption("Scan stocks by index or screener criteria, categorized by market cap and time horizon")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Scan method selection
    st.subheader("Stock Discovery Method")
    scan_method = st.radio(
        "Choose scan method:",
        ["Option A: Index-Based", "Option B: Yahoo Screener"],
        help="Option A: Scan full index lists\nOption B: Use Yahoo Finance screener"
    )
    
    st.divider()
    
    # Index selection for Option A
    if scan_method == "Option A: Index-Based":
        st.subheader("Select Indices to Scan")
        scan_russell = st.checkbox("Russell 2000 (Small Cap)", value=True)
        scan_sp400 = st.checkbox("S&P 400 (Mid Cap)", value=True)
        scan_sp500 = st.checkbox("S&P 500 (Large Cap)", value=True)
    else:
        st.subheader("Screener Filters")
        st.info("Yahoo screener integration in development")
    
    st.divider()
    
    # Profit targets
    st.subheader("Profit Target Thresholds")
    temp_short = st.slider("Short-Term (1-3 days)", 0.0, 50.0, float(st.session_state.applied_short), step=0.5)
    temp_mid = st.slider("Mid-Term (1-2 weeks)", 0.0, 50.0, float(st.session_state.applied_mid), step=0.5)
    temp_long = st.slider("Long-Term (1-3 months)", 0.0, 100.0, float(st.session_state.applied_long), step=1.0)
    
    st.info(f"**Current Applied:**\n- Short: {st.session_state.applied_short}%\n- Mid: {st.session_state.applied_mid}%\n- Long: {st.session_state.applied_long}%")
    
    st.divider()
    
    # Scan button
    if st.button("🚀 Start Scan", type="primary", use_container_width=True):
        st.session_state.applied_short = temp_short
        st.session_state.applied_mid = temp_mid
        st.session_state.applied_long = temp_long
        
        # Clear cache
        st.cache_data.clear()
        
        # Start scanning
        with st.spinner("Scanning stocks..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if scan_method == "Option A: Index-Based":
                all_tickers = []
                if scan_russell:
                    all_tickers.extend(get_russell2000_tickers())
                if scan_sp400:
                    all_tickers.extend(get_sp400_tickers())
                if scan_sp500:
                    all_tickers.extend(get_sp500_tickers())
                
                # Remove duplicates
                all_tickers = list(set(all_tickers))
                
                st.session_state.stock_universe = scan_stocks_from_index(
                    all_tickers, progress_bar, status_text
                )
            else:
                # Option B - screener based (placeholder)
                st.session_state.stock_universe = scan_stocks_by_screener(
                    'all', progress_bar, status_text
                )
            
            progress_bar.empty()
            status_text.empty()
            st.session_state.last_scan_time = datetime.now()
            st.success("✅ Scan complete!")
            st.rerun()
    
    # Show last scan time
    if st.session_state.last_scan_time:
        st.caption(f"Last scan: {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.divider()
    st.caption("**Market Cap Definitions:**\n- Small: < $2B\n- Mid: $2B - $10B\n- Large: > $10B")

# Main content
if st.session_state.stock_universe is None:
    st.info("👈 Configure your scan settings and click 'Start Scan' to begin")
    st.markdown("""
    ### How it works:
    1. **Choose discovery method**: Index-based or Yahoo screener
    2. **Set profit targets**: Minimum expected returns for each time horizon
    3. **Click Start Scan**: The app will analyze stocks and categorize them
    4. **Review results**: Top 33 opportunities per category
    
    ### Categories:
    - **Small Cap (<$2B)**: Optimized for 1-3 day trades
    - **Mid Cap ($2B-$10B)**: Optimized for 1-2 week holds
    - **Large Cap (>$10B)**: Optimized for 1-3 month positions
    """)
else:
    # Display results
    col1, col2, col3 = st.columns(3)
    
    universe = st.session_state.stock_universe
    
    # Small Cap Table
    with col1:
        st.subheader("📊 Small Cap (< $2B)")
        st.caption(f"Short-term (1-3 days) • Target: {st.session_state.applied_short}%")
        
        if universe['small']:
            df_small = build_opportunity_table(
                universe['small'], 
                'short', 
                st.session_state.applied_short
            )
            if not df_small.empty:
                st.dataframe(df_small, use_container_width=True, height=400, hide_index=True)
                st.caption(f"Showing {len(df_small)} opportunities")
            else:
                st.warning("No stocks meet criteria")
        else:
            st.info("No small cap stocks found. Try scanning Russell 2000.")
    
    # Mid Cap Table
    with col2:
        st.subheader("📈 Mid Cap ($2B-$10B)")
        st.caption(f"Mid-term (1-2 weeks) • Target: {st.session_state.applied_mid}%")
        
        if universe['mid']:
            df_mid = build_opportunity_table(
                universe['mid'], 
                'mid', 
                st.session_state.applied_mid
            )
            if not df_mid.empty:
                st.dataframe(df_mid, use_container_width=True, height=400, hide_index=True)
                st.caption(f"Showing {len(df_mid)} opportunities")
            else:
                st.warning("No stocks meet criteria")
        else:
            st.info("No mid cap stocks found. Try scanning S&P 400.")
    
    # Large Cap Table
    with col3:
        st.subheader("📉 Large Cap (> $10B)")
        st.caption(f"Long-term (1-3 months) • Target: {st.session_state.applied_long}%")
        
        if universe['large']:
            df_large = build_opportunity_table(
                universe['large'], 
                'long', 
                st.session_state.applied_long
            )
            if not df_large.empty:
                st.dataframe(df_large, use_container_width=True, height=400, hide_index=True)
                st.caption(f"Showing {len(df_large)} opportunities")
            else:
                st.warning("No stocks meet criteria")
        else:
            st.info("No large cap stocks found. Try scanning S&P 500.")
    
    # Chart section
    st.divider()
    all_tickers = [s['ticker'] for s in universe['small']] + \
                  [s['ticker'] for s in universe['mid']] + \
                  [s['ticker'] for s in universe['large']]
    
    if all_tickers:
        selected_ticker = st.selectbox("📊 View Technical Chart", [""] + sorted(all_tickers))
        
        if selected_ticker:
            st.subheader(f"Technical Analysis: {selected_ticker}")
            
            data, info = fetch_stock_data(selected_ticker, '6mo')
            
            if not data.empty:
                # Calculate indicators
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=(f'{selected_ticker} Price', 'Volume'),
                    row_heights=[0.7, 0.3]
                )
                
                # Candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='Price'
                    ),
                    row=1, col=1
                )
                
                # Moving averages
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20',
                               line=dict(color='orange', width=1)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50',
                               line=dict(color='blue', width=1)),
                    row=1, col=1
                )
                
                # Volume
                colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green'
                          for i in range(len(data))]
                fig.add_trace(
                    go.Bar(x=data.index, y=data['Volume'], name='Volume',
                           marker_color=colors, opacity=0.5),
                    row=2, col=1
                )
                
                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    height=700,
                    showlegend=True,
                    hovermode='x unified'
                )
                fig.update_xaxes(title_text="Date", row=2, col=1)
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics
                col_a, col_b, col_c, col_d = st.columns(4)
                potential_short, metrics_short = get_profit_potential(data, info, 'short')
                potential_mid, metrics_mid = get_profit_potential(data, info, 'mid')
                potential_long, metrics_long = get_profit_potential(data, info, 'long')
                
                col_a.metric("Short-Term Potential", f"{potential_short:.2f}%")
                col_b.metric("Mid-Term Potential", f"{potential_mid:.2f}%")
                col_c.metric("Long-Term Potential", f"{potential_long:.2f}%")
                col_d.metric("Current RSI", f"{metrics_short.get('RSI', 50):.1f}")

# Footer
st.divider()
st.caption(f"📡 Data via yfinance | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("⚠️ This is a prototype for educational purposes. Not financial advice.")
