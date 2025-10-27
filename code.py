import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Hardcoded universes (expandable)
SMALL_CAP_TICKERS = ['AVAH', 'STRA', 'ARLO', 'LFST', 'ASGN', 'ROCK', 'VOYG', 'DQ', 'SYBT', 'PRGS',
                     'DX', 'EVTC', 'DV', 'WLY', 'RCUS', 'SDRL', 'PACS', 'BELFB', 'SLDE', 'CLOV',
                     'NTCT', 'LQDA', 'DXPE', 'PLUS', 'NWN', 'VERA', 'ANIP', 'FIHL', 'ADNT', 'TRIP',
                     'SCS', 'CXM', 'ADEA']
MID_CAP_TICKERS = ['CMA', 'IPG', 'PEN', 'ORI', 'AIT', 'KNSL', 'OTEX', 'ALGN', 'RRX', 'IDCC',
                   'CRL', 'OVV', 'SARO', 'AOS', 'DCI', 'CUBE', 'FRHC', 'PSO', 'MKSI', 'SPXC',
                   'EDU', 'BWA', 'MOS', 'AUR', 'LSCC', 'DDS', 'FIGR', 'EGP', 'FYBR', 'MDGL',
                   'ESTC', 'UWMC', 'RGEN']
LARGE_CAP_TICKERS = ['UBER', 'ANET', 'NOW', 'LRCX', 'PDD', 'ISRG', 'INTU', 'BX', 'ARM', 'INTC',
                     'AMAT', 'T', 'C', 'BLK', 'HDB', 'NEE', 'SONY', 'SCHW', 'BKNG', 'MUFG',
                     'BA', 'APH', 'VZ', 'KLAC', 'TJX', 'GEV', 'AMGN', 'ACN', 'DHR', 'UL',
                     'TXN', 'SPGI', 'BSX']

# Initialize session state for applied targets
if 'applied_short' not in st.session_state:
    st.session_state.applied_short = 5.0
if 'applied_mid' not in st.session_state:
    st.session_state.applied_mid = 10.0
if 'applied_long' not in st.session_state:
    st.session_state.applied_long = 30.0

@st.cache_data(ttl=300)  # Cache for 5 min to avoid API spam
def fetch_stock_data(ticker, period='6mo'):
    """Fetch stock data and info with proper error handling"""
    try:
        data = yf.download(ticker, period=period, progress=False)
        # Fix for yfinance MultiIndex columns (single ticker -> droplevel)
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
    vol = returns.std() * np.sqrt(252)  # Annualized
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
        # Short-term: Focus on momentum, volume, and recent volatility
        lookback_days = 5
        if len(data) < lookback_days:
            return 0.0, {}
        
        recent_high = safe_float(data['High'].tail(lookback_days).max())
        momentum = calculate_momentum(data, lookback=5)
        
        # Distance to recent high
        upside_to_high = ((recent_high - current_price) / current_price * 100) if current_price > 0 else 0
        
        # Weighted potential based on multiple factors
        potential = (
            upside_to_high * 0.4 +           # 40% weight on distance to recent high
            momentum * 0.3 +                  # 30% weight on momentum
            (volume_surge - 1) * 5 * 0.2 +   # 20% weight on volume surge
            (50 - abs(rsi - 50)) / 10 * 0.1  # 10% weight on RSI extremes
        )
        
        metrics = {
            'RSI': rsi,
            'Momentum_5D': momentum,
            'Vol_Surge': volume_surge,
            'Volatility': volatility
        }
        
    elif horizon_type == 'mid':
        # Mid-term: Focus on trend strength, MACD, and moderate momentum
        lookback_days = 14
        if len(data) < lookback_days:
            return 0.0, {}
        
        momentum = calculate_momentum(data, lookback=14)
        
        # MACD contribution (normalized by volatility)
        macd_signal = (macd / volatility * 10) if volatility > 0.01 else 0
        
        # RSI oversold/overbought signal
        rsi_signal = 0
        if rsi < 30:  # Oversold - potential upside
            rsi_signal = (30 - rsi) * 0.5
        elif rsi > 70:  # Overbought - potential pullback risk
            rsi_signal = (70 - rsi) * 0.3
        
        potential = (
            momentum * 0.4 +                  # 40% weight on 2-week momentum
            macd_signal * 0.3 +               # 30% weight on MACD trend
            rsi_signal * 0.2 +                # 20% weight on RSI positioning
            (volume_surge - 1) * 3 * 0.1     # 10% weight on volume
        )
        
        metrics = {
            'RSI': rsi,
            'MACD': macd,
            'Momentum_14D': momentum,
            'Vol_Surge': volume_surge
        }
        
    else:  # long
        # Long-term: Focus on extended trends, beta, and fundamental momentum
        lookback_days = 60
        if len(data) < lookback_days:
            return 0.0, {}
        
        momentum_60 = calculate_momentum(data, lookback=60)
        momentum_20 = calculate_momentum(data, lookback=20)
        beta = safe_float(info.get('beta', 1.0), 1.0)
        
        # Trend acceleration (is momentum increasing?)
        trend_acceleration = momentum_20 - momentum_60
        
        # Beta-adjusted potential
        beta_factor = 1.0 + (beta - 1.0) * 0.5  # Moderate beta impact
        
        potential = (
            momentum_60 * 0.5 * beta_factor +  # 50% weight on long-term momentum
            trend_acceleration * 0.3 +          # 30% weight on acceleration
            (50 - abs(rsi - 50)) / 5 * 0.2     # 20% weight on RSI positioning
        )
        
        metrics = {
            'Beta': beta,
            'Momentum_60D': momentum_60,
            'Momentum_20D': momentum_20,
            'RSI': rsi
        }
    
    return safe_float(potential, 0.0), metrics

def calculate_composite_score(potential, metrics, horizon_type):
    """
    Calculate composite score incorporating potential and additional metrics
    
    Higher score = better opportunity
    """
    if not metrics:
        return potential
    
    score = potential * 0.6  # 60% weight on potential
    
    if horizon_type == 'short':
        # Bonus for volume surge and favorable RSI
        vol_surge = metrics.get('Vol_Surge', 1.0)
        rsi = metrics.get('RSI', 50.0)
        
        score += (vol_surge - 1) * 10 * 0.2  # Volume surge bonus
        if 30 <= rsi <= 45:  # Oversold but not extreme
            score += 5 * 0.2
        elif 55 <= rsi <= 70:  # Overbought momentum
            score += 3 * 0.2
            
    elif horizon_type == 'mid':
        # Bonus for MACD and momentum alignment
        macd = metrics.get('MACD', 0.0)
        momentum = metrics.get('Momentum_14D', 0.0)
        
        if macd > 0 and momentum > 0:  # Aligned signals
            score += 5 * 0.4
        
    else:  # long
        # Bonus for strong trend and beta
        momentum_60 = metrics.get('Momentum_60D', 0.0)
        beta = metrics.get('Beta', 1.0)
        
        if momentum_60 > 10:  # Strong uptrend
            score += 10 * 0.3
        if beta > 1.2:  # High beta for leverage
            score += 5 * 0.1
    
    return safe_float(score, 0.0)

# Streamlit App
st.set_page_config(page_title="99 Stocks Dashboard", layout="wide")
st.title("99 Stocks Dashboard: Segmented Opportunities by Market Cap & Horizon")

# Sidebars for sliders and on-demand refresh
with st.sidebar:
    st.header("Profit Target Adjustments")
    st.write("**Minimum Profit Potential %**")
    temp_short = st.slider("Short-Term (1-3 days)", 0.0, 50.0, float(st.session_state.applied_short), step=0.5)
    temp_mid = st.slider("Mid-Term (1-2 weeks)", 0.0, 50.0, float(st.session_state.applied_mid), step=0.5)
    temp_long = st.slider("Long-Term (1-3 months)", 0.0, 100.0, float(st.session_state.applied_long), step=1.0)
    
    st.info(f"**Current Applied:**\n- Short: {st.session_state.applied_short}%\n- Mid: {st.session_state.applied_mid}%\n- Long: {st.session_state.applied_long}%")
    
    if st.button("Apply Changes & Refresh", type="primary"):
        st.session_state.applied_short = temp_short
        st.session_state.applied_mid = temp_mid
        st.session_state.applied_long = temp_long
        st.cache_data.clear()
        st.rerun()
    
    if st.button("Force Full Refresh"):
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    st.write("**Algorithm Info:**")
    st.caption("• Short: Recent momentum + volume + RSI\n• Mid: Trend strength + MACD + positioning\n• Long: Extended trends + beta + acceleration")

# Selected stock for chart
all_tickers = SMALL_CAP_TICKERS + MID_CAP_TICKERS + LARGE_CAP_TICKERS
selected_ticker = st.sidebar.selectbox("View Candlestick Chart", [""] + all_tickers)

col1, col2, col3 = st.columns(3)

# Function to build and display table
@st.cache_data(ttl=300)
def build_table_df(tickers, cap_type, horizon_type, target_threshold):
    """
    Build dataframe of stocks meeting criteria
    
    Parameters:
    - tickers: List of ticker symbols
    - cap_type: 'Small', 'Mid', or 'Large' for display
    - horizon_type: 'short', 'mid', or 'long' for calculation
    - target_threshold: Minimum profit potential to include
    """
    df_list = []
    
    for ticker in tickers:
        data, info = fetch_stock_data(ticker)
        if data.empty or len(data) < 20:
            continue
        
        # Calculate potential and metrics
        potential, metrics = get_profit_potential(data, info, horizon_type)
        
        # Skip if below threshold
        if potential < target_threshold:
            continue
        
        # Calculate composite score
        score = calculate_composite_score(potential, metrics, horizon_type)
        
        # Get current price
        current_price = safe_float(data['Close'].iloc[-1])
        
        # Get primary additional metric based on horizon
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
            'Name': info.get('longName', ticker)[:30],  # Truncate long names
            'Price': round(current_price, 2),
            'Potential %': round(potential, 2),
            'Score': round(score, 2),
            add_metric_name: round(add_metric_value, 2)
        })
    
    if df_list:
        df = pd.DataFrame(df_list).sort_values('Score', ascending=False)
        # Return top 33 or all if less than 33
        return df.head(33)
    
    return pd.DataFrame()

def display_table(tickers, cap_type, horizon_type, target, col):
    """Display table with stocks meeting criteria"""
    horizon_label = {
        'short': '1-3 days',
        'mid': '1-2 weeks', 
        'long': '1-3 months'
    }
    
    with col:
        st.subheader(f"{cap_type} Cap • {horizon_label[horizon_type]}")
        st.caption(f"Minimum Potential: {target}%")
        
        df = build_table_df(tickers, cap_type, horizon_type, target)
        
        if not df.empty:
            # Color-code the Potential % column
            st.dataframe(
                df,
                use_container_width=True,
                height=400,
                hide_index=True
            )
            st.caption(f"Showing {len(df)} of {len(tickers)} stocks")
        else:
            st.warning(f"⚠️ No stocks meet the {target}% threshold. Try lowering the slider.")

# Build tables using applied targets
display_table(SMALL_CAP_TICKERS, 'Small', 'short', st.session_state.applied_short, col1)
display_table(MID_CAP_TICKERS, 'Mid', 'mid', st.session_state.applied_mid, col2)
display_table(LARGE_CAP_TICKERS, 'Large', 'long', st.session_state.applied_long, col3)

# Candlestick Chart
if selected_ticker:
    st.divider()
    st.header(f"📊 Technical Chart: {selected_ticker}")
    
    data, info = fetch_stock_data(selected_ticker, '6mo')
    
    if not data.empty:
        # Calculate indicators for overlay
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
            go.Scatter(
                x=data.index, 
                y=data['SMA_20'], 
                name='SMA 20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['SMA_50'], 
                name='SMA 50',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Volume
        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                  for i in range(len(data))]
        fig.add_trace(
            go.Bar(
                x=data.index, 
                y=data['Volume'], 
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
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
        
        # Show metrics
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

# Heatmap of All 99
st.divider()
st.header("🔥 Profit Potential Heatmap (All 99 Stocks)")

heatmap_data = []
for ticker in all_tickers:
    data, info = fetch_stock_data(ticker)
    if not data.empty:
        # Use mid-term for balanced view
        potential, _ = get_profit_potential(data, info, 'mid')
        heatmap_data.append({'Ticker': ticker, 'Potential %': potential})

if heatmap_data:
    df_heat = pd.DataFrame(heatmap_data)
    
    if not df_heat.empty:
        # Create chunks for better visualization
        chunk_size = 33
        chunks = [df_heat.iloc[i:i+chunk_size] for i in range(0, len(df_heat), chunk_size)]
        
        fig, axes = plt.subplots(len(chunks), 1, figsize=(16, 4*len(chunks)))
        if len(chunks) == 1:
            axes = [axes]
        
        for idx, chunk in enumerate(chunks):
            pivot = chunk.set_index('Ticker')['Potential %'].to_frame().T
            sns.heatmap(
                pivot, 
                annot=True, 
                cmap='RdYlGn', 
                center=0, 
                fmt='.1f',
                ax=axes[idx],
                cbar_kws={'label': 'Potential %'},
                vmin=-20,
                vmax=40
            )
            cap_label = ['Small Cap', 'Mid Cap', 'Large Cap'][idx] if idx < 3 else ''
            axes[idx].set_title(f'{cap_label} Stocks', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Insufficient data for heatmap.")
else:
    st.warning("Unable to generate heatmap—check API connection.")

# Footer
st.divider()
st.caption(f"📡 Data via yfinance | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Cache TTL: 5 minutes")
st.caption("⚠️ This is a prototype for educational purposes. Not financial advice.")
