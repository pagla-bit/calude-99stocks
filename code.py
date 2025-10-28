import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# Initialize session state
if 'applied_short' not in st.session_state:
    st.session_state.applied_short = 5.0
if 'applied_mid' not in st.session_state:
    st.session_state.applied_mid = 10.0
if 'applied_long' not in st.session_state:
    st.session_state.applied_long = 30.0
if 'small_cap_tickers' not in st.session_state:
    st.session_state.small_cap_tickers = ""
if 'mid_cap_tickers' not in st.session_state:
    st.session_state.mid_cap_tickers = ""
if 'large_cap_tickers' not in st.session_state:
    st.session_state.large_cap_tickers = ""
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'last_scan_time' not in st.session_state:
    st.session_state.last_scan_time = None

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

def parse_tickers(ticker_string):
    """Parse comma-separated ticker string into list"""
    if not ticker_string or not ticker_string.strip():
        return []
    
    # Split by comma, space, or newline
    tickers = ticker_string.replace('\n', ',').replace(' ', ',').split(',')
    # Clean and uppercase
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)
    return unique_tickers

def scan_tickers(ticker_list, horizon_type, target_threshold, progress_bar=None, status_text=None):
    """Scan list of tickers and return opportunities"""
    df_list = []
    total = len(ticker_list)
    
    for idx, ticker in enumerate(ticker_list):
        if progress_bar:
            progress_bar.progress((idx + 1) / total)
        if status_text:
            status_text.text(f"Scanning {ticker}... ({idx + 1}/{total})")
        
        data, info = fetch_stock_data(ticker, period='6mo')
        
        if data.empty or len(data) < 20:
            continue
        
        # Calculate potential and metrics
        potential, metrics = get_profit_potential(data, info, horizon_type)
        
        # Skip if below threshold
        if potential < target_threshold:
            continue
        
        # Calculate composite score
        score = calculate_composite_score(potential, metrics, horizon_type)
        
        current_price = safe_float(data['Close'].iloc[-1])
        
        # Get market cap
        market_cap = info.get('marketCap', 0)
        market_cap_b = market_cap / 1e9 if market_cap else 0
        
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
            'Mkt Cap ($B)': round(market_cap_b, 2),
            'Potential %': round(potential, 2),
            'Score': round(score, 2),
            add_metric_name: round(add_metric_value, 2)
        })
    
    if df_list:
        df = pd.DataFrame(df_list).sort_values('Score', ascending=False)
        return df.head(33)
    
    return pd.DataFrame()

# Streamlit App
st.set_page_config(page_title="Stock Opportunity Scanner", layout="wide")
st.title("🔍 Stock Opportunity Scanner")
st.caption("Input your tickers, set criteria, and discover top opportunities")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("📝 Input Tickers")
    st.caption("Enter comma-separated tickers (e.g., AAPL, MSFT, GOOGL)")
    
    # Small cap input
    with st.expander("🔹 Small Cap Tickers", expanded=True):
        small_tickers_input = st.text_area(
            "Enter small cap tickers:",
            value=st.session_state.small_cap_tickers,
            height=100,
            key="small_input",
            placeholder="AVAH, STRA, ARLO, LFST, ASGN..."
        )
        if st.button("Load Sample Small Caps"):
            sample_small = "AVAH, STRA, ARLO, LFST, ASGN, ROCK, VOYG, DQ, SYBT, PRGS, DX, EVTC, DV, WLY, RCUS, SDRL, PACS, BELFB, SLDE, CLOV, NTCT, LQDA, DXPE, PLUS, NWN, VERA, ANIP, FIHL, ADNT, TRIP, SCS, CXM, ADEA"
            st.session_state.small_cap_tickers = sample_small
            st.rerun()
    
    # Mid cap input
    with st.expander("🔸 Mid Cap Tickers", expanded=True):
        mid_tickers_input = st.text_area(
            "Enter mid cap tickers:",
            value=st.session_state.mid_cap_tickers,
            height=100,
            key="mid_input",
            placeholder="CMA, IPG, PEN, ORI, AIT..."
        )
        if st.button("Load Sample Mid Caps"):
            sample_mid = "CMA, IPG, PEN, ORI, AIT, KNSL, OTEX, ALGN, RRX, IDCC, CRL, OVV, SARO, AOS, DCI, CUBE, FRHC, PSO, MKSI, SPXC, EDU, BWA, MOS, AUR, LSCC, DDS, FIGR, EGP, FYBR, MDGL, ESTC, UWMC, RGEN"
            st.session_state.mid_cap_tickers = sample_mid
            st.rerun()
    
    # Large cap input
    with st.expander("🔺 Large Cap Tickers", expanded=True):
        large_tickers_input = st.text_area(
            "Enter large cap tickers:",
            value=st.session_state.large_cap_tickers,
            height=100,
            key="large_input",
            placeholder="UBER, ANET, NOW, LRCX, PDD..."
        )
        if st.button("Load Sample Large Caps"):
            sample_large = "UBER, ANET, NOW, LRCX, PDD, ISRG, INTU, BX, ARM, INTC, AMAT, T, C, BLK, HDB, NEE, SONY, SCHW, BKNG, MUFG, BA, APH, VZ, KLAC, TJX, GEV, AMGN, ACN, DHR, UL, TXN, SPGI, BSX"
            st.session_state.large_cap_tickers = sample_large
            st.rerun()
    
    st.divider()
    
    # Profit targets
    st.subheader("🎯 Profit Target Thresholds")
    temp_short = st.slider("Short-Term (1-3 days)", 0.0, 50.0, float(st.session_state.applied_short), step=0.5)
    temp_mid = st.slider("Mid-Term (1-2 weeks)", 0.0, 50.0, float(st.session_state.applied_mid), step=0.5)
    temp_long = st.slider("Long-Term (1-3 months)", 0.0, 100.0, float(st.session_state.applied_long), step=1.0)
    
    st.info(f"**Current Applied:**\n- Short: {st.session_state.applied_short}%\n- Mid: {st.session_state.applied_mid}%\n- Long: {st.session_state.applied_long}%")
    
    st.divider()
    
    # Scan button
    if st.button("🚀 Start Scan", type="primary", use_container_width=True):
        # Update session state
        st.session_state.applied_short = temp_short
        st.session_state.applied_mid = temp_mid
        st.session_state.applied_long = temp_long
        st.session_state.small_cap_tickers = small_tickers_input
        st.session_state.mid_cap_tickers = mid_tickers_input
        st.session_state.large_cap_tickers = large_tickers_input
        
        # Parse tickers
        small_list = parse_tickers(small_tickers_input)
        mid_list = parse_tickers(mid_tickers_input)
        large_list = parse_tickers(large_tickers_input)
        
        if not small_list and not mid_list and not large_list:
            st.error("❌ Please enter at least one ticker in any category!")
        else:
            # Clear cache
            st.cache_data.clear()
            
            # Start scanning
            results = {'small': None, 'mid': None, 'large': None}
            
            with st.spinner("Scanning stocks..."):
                # Scan small caps
                if small_list:
                    st.info(f"Scanning {len(small_list)} small cap stocks...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results['small'] = scan_tickers(
                        small_list, 'short', temp_short, progress_bar, status_text
                    )
                    progress_bar.empty()
                    status_text.empty()
                
                # Scan mid caps
                if mid_list:
                    st.info(f"Scanning {len(mid_list)} mid cap stocks...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results['mid'] = scan_tickers(
                        mid_list, 'mid', temp_mid, progress_bar, status_text
                    )
                    progress_bar.empty()
                    status_text.empty()
                
                # Scan large caps
                if large_list:
                    st.info(f"Scanning {len(large_list)} large cap stocks...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results['large'] = scan_tickers(
                        large_list, 'long', temp_long, progress_bar, status_text
                    )
                    progress_bar.empty()
                    status_text.empty()
            
            st.session_state.scan_results = results
            st.session_state.last_scan_time = datetime.now()
            st.success("✅ Scan complete!")
            st.rerun()
    
    # Show last scan time
    if st.session_state.last_scan_time:
        st.caption(f"Last scan: {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.divider()
    st.caption("""
    **How It Works:**
    1. Enter tickers for each category
    2. Set minimum profit targets
    3. Click "Start Scan"
    4. Review top 33 opportunities
    """)

# Main content
if st.session_state.scan_results is None:
    st.info("👈 Enter your tickers in the sidebar and click 'Start Scan' to begin")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📊 Small Cap
        **Time Horizon:** 1-3 days  
        **Focus:** Short-term momentum, volume surges, quick moves
        
        **Algorithm:**
        - Distance to recent high (40%)
        - 5-day momentum (30%)
        - Volume surge (20%)
        - RSI positioning (10%)
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Mid Cap
        **Time Horizon:** 1-2 weeks  
        **Focus:** Trend strength, technical signals
        
        **Algorithm:**
        - 14-day momentum (40%)
        - MACD trend (30%)
        - RSI signals (20%)
        - Volume (10%)
        """)
    
    with col3:
        st.markdown("""
        ### 📉 Large Cap
        **Time Horizon:** 1-3 months  
        **Focus:** Extended trends, fundamentals
        
        **Algorithm:**
        - 60-day momentum + beta (50%)
        - Trend acceleration (30%)
        - RSI positioning (20%)
        """)
else:
    # Display results
    results = st.session_state.scan_results
    col1, col2, col3 = st.columns(3)
    
    # Small Cap Table
    with col1:
        st.subheader("📊 Small Cap Opportunities")
        st.caption(f"Short-term (1-3 days) • Target: {st.session_state.applied_short}%")
        
        if results['small'] is not None and not results['small'].empty:
            st.dataframe(results['small'], use_container_width=True, height=500, hide_index=True)
            st.caption(f"✅ Showing {len(results['small'])} opportunities")
        else:
            st.warning("⚠️ No stocks meet criteria. Try lowering the target threshold.")
    
    # Mid Cap Table
    with col2:
        st.subheader("📈 Mid Cap Opportunities")
        st.caption(f"Mid-term (1-2 weeks) • Target: {st.session_state.applied_mid}%")
        
        if results['mid'] is not None and not results['mid'].empty:
            st.dataframe(results['mid'], use_container_width=True, height=500, hide_index=True)
            st.caption(f"✅ Showing {len(results['mid'])} opportunities")
        else:
            st.warning("⚠️ No stocks meet criteria. Try lowering the target threshold.")
    
    # Large Cap Table
    with col3:
        st.subheader("📉 Large Cap Opportunities")
        st.caption(f"Long-term (1-3 months) • Target: {st.session_state.applied_long}%")
        
        if results['large'] is not None and not results['large'].empty:
            st.dataframe(results['large'], use_container_width=True, height=500, hide_index=True)
            st.caption(f"✅ Showing {len(results['large'])} opportunities")
        else:
            st.warning("⚠️ No stocks meet criteria. Try lowering the target threshold.")
    
    # Chart section
    st.divider()
    
    # Collect all tickers from results
    all_tickers = []
    if results['small'] is not None and not results['small'].empty:
        all_tickers.extend(results['small']['Ticker'].tolist())
    if results['mid'] is not None and not results['mid'].empty:
        all_tickers.extend(results['mid']['Ticker'].tolist())
    if results['large'] is not None and not results['large'].empty:
        all_tickers.extend(results['large']['Ticker'].tolist())
    
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
            else:
                st.error(f"Unable to load data for {selected_ticker}")

# Footer
st.divider()
st.caption(f"📡 Data via yfinance | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("⚠️ This is a prototype for educational purposes. Not financial advice.")
