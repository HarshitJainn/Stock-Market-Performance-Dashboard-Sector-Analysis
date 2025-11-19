# app.py (improved interactive layout & spacing)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

st.set_page_config(layout="wide", page_title="NIFTY50 Sector Dashboard", initial_sidebar_state="expanded")
st.title("NIFTY50 Sector Dashboard")

# ----------------------
# Helpers & Loaders
# ----------------------
@st.cache_data
def load_constituents():
    p = DATA_DIR / "nifty50_constituents.csv"
    if not p.exists():
        return pd.DataFrame(columns=["Ticker","Company","Sector"])
    return pd.read_csv(p)

@st.cache_data
def load_market_caps():
    p = DATA_DIR / "market_caps.csv"
    if p.exists():
        df = pd.read_csv(p)
        if 'market_cap' in df.columns:
            df['market_cap'] = pd.to_numeric(df['market_cap'], errors='coerce')
        return df
    return pd.DataFrame(columns=['Ticker','yahoo_ticker','market_cap','market_cap_date'])

@st.cache_data
def load_sector_mcap_daily_returns():
    p = DATA_DIR / "sector_mcap_daily_returns.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None

@st.cache_data
def compute_equal_weight_sector_returns():
    try:
        from sector_aggregation import compute_sector_returns
    except Exception:
        return None, None
    const = load_constituents()
    raw_returns, sector_eq = compute_sector_returns(equal_weight=True, constituents_df=const, market_caps_df=None)
    return raw_returns, sector_eq

@st.cache_data
def compute_mcap_weighted_returns():
    # prefer precomputed file
    p = DATA_DIR / "sector_mcap_daily_returns.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        return df
    try:
        from sector_aggregation_weighted import compute_marketcap_weighted_sector_returns
    except Exception:
        return None
    const = load_constituents()
    mcap_df = load_market_caps()
    _, _, sector_mcap = compute_marketcap_weighted_sector_returns(constituents_df=const, market_caps_df=mcap_df, market_caps_history=None, rebalance_freq='M')
    return sector_mcap

# ----------------------
# Sidebar controls
# ----------------------
st.sidebar.header("Controls")
mode = st.sidebar.radio("Sector aggregation method", options=["Equal-weighted", "Market-cap weighted"])

# Date controls (quick ranges)
end_default = datetime.today()
start_default = end_default - timedelta(days=365*5)
range_choice = st.sidebar.selectbox("Quick range", options=["5y","3y","1y","YTD","Custom"], index=0)

if range_choice == "5y":
    start_date = st.sidebar.date_input("Start date", value=start_default)
    end_date = st.sidebar.date_input("End date", value=end_default)
elif range_choice == "3y":
    start_date = st.sidebar.date_input("Start date", value=end_default - timedelta(days=365*3))
    end_date = st.sidebar.date_input("End date", value=end_default)
elif range_choice == "1y":
    start_date = st.sidebar.date_input("Start date", value=end_default - timedelta(days=365))
    end_date = st.sidebar.date_input("End date", value=end_default)
elif range_choice == "YTD":
    start_date = st.sidebar.date_input("Start date", value=datetime(end_default.year,1,1))
    end_date = st.sidebar.date_input("End date", value=end_default)
else:
    start_date = st.sidebar.date_input("Start date", value=start_default)
    end_date = st.sidebar.date_input("End date", value=end_default)

# Chart options
with st.sidebar.expander("Chart options", expanded=False):
    show_legend = st.checkbox("Show legend", value=False)
    use_log_scale = st.checkbox("Log scale (cumulative chart)", value=False)
    max_lines = st.slider("Max sectors to display (for clarity)", min_value=3, max_value=20, value=8)

# Sector selection area (multi-select)
const = load_constituents()
sector_list_all = []
if not const.empty:
    sector_list_all = sorted(const['Sector'].unique())
else:
    sector_list_all = []

st.sidebar.markdown("### Sector filter")
selected_sectors = st.sidebar.multiselect("Pick sectors (max {})".format(max_lines), options=sector_list_all, default=sector_list_all[:min(8, len(sector_list_all))])

# Benchmark option
bench_choice = st.sidebar.selectbox("Benchmark for deep-dive", options=["^NSEI (Nifty 50 index)"], index=0)

# ----------------------
# Load data per chosen mode
# ----------------------
if mode == "Equal-weighted":
    raw_returns, sector_returns = compute_equal_weight_sector_returns()
    if sector_returns is None:
        st.error("Equal-weight sector returns not available. Make sure sector_aggregation.py exists.")
        st.stop()
else:
    sector_returns = compute_mcap_weighted_returns()
    if sector_returns is None:
        st.error("Market-cap weighted sector returns not available. Make sure sector_aggregation_weighted.py exists or precomputed parquet present.")
        st.stop()

# Ensure index is datetime and sorted
sector_returns.index = pd.to_datetime(sector_returns.index)
sector_returns = sector_returns.sort_index()

# filter date range
sector_returns = sector_returns.loc[(sector_returns.index >= pd.to_datetime(start_date)) & (sector_returns.index <= pd.to_datetime(end_date))]

if sector_returns.empty:
    st.warning("No data in the selected date range. Try expanding the range or run the pipeline to update data.")
    st.stop()

# If user didn't select sectors, choose top by cumulative
if not selected_sectors:
    cum_all = (1 + sector_returns).cumprod().iloc[-1].sort_values(ascending=False)
    selected_sectors = cum_all.head(min(max_lines, len(cum_all))).index.tolist()

# limit number of sectors to avoid clutter
selected_sectors = selected_sectors[:max_lines]

# ----------------------
# Tabs for neat separation
# ----------------------
# create tabs including new Correlation & Network tab
tabs = st.tabs(["Market Overview", "Sector Deep Dive", "Monthly Heatmap", "Risk-Return", "Correlation & Network", "Download / Audit"])
tab_overview, tab_deep, tab_heat, tab_rr, tab_corr, tab_down = tabs



# ----------------------
# Market Overview Tab
# ----------------------
with tab_overview:
    st.subheader("Market Overview")
    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    # compute cumulative returns for selected sectors
    cum = (1 + sector_returns[selected_sectors]).cumprod() - 1
    last_cum = cum.iloc[-1]

    # Annualized vol & sharpe
    from metrics_india import annualized_volatility, sharpe_ratio
    ann_vol = sector_returns[selected_sectors].apply(lambda s: annualized_volatility(s.dropna()))
    sharpe = sector_returns[selected_sectors].apply(lambda s: sharpe_ratio(s.dropna(), risk_free_rate=0.06))

    # Show top / bottom in KPIs (YTD-like but using selected timeframe)
    best_sector = last_cum.idxmax()
    worst_sector = last_cum.idxmin()
    col1.metric("Best (selected)", f"{best_sector}", f"{last_cum[best_sector]:.1%}")
    col2.metric("Worst (selected)", f"{worst_sector}", f"{last_cum[worst_sector]:.1%}")
    # show median sector vol + average sharpe as extra KPIs
    col3.metric("Median Annual Vol (selected)", f"{ann_vol.median():.1%}")
    col4.metric("Avg Sharpe (selected)", f"{sharpe.mean():.2f}")

    # Cumulative returns chart
    fig = px.line(cum[selected_sectors], x=cum.index, y=selected_sectors, title="Cumulative returns (selected sectors)", labels={"value":"Cumulative return","index":"Date"})
    fig.update_layout(
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40,r=20,t=60,b=40),
        height=520
    )
    if not show_legend:
        fig.update_layout(showlegend=False)
    if use_log_scale:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)

    # Annual returns bar (last year in range)
    st.markdown("#### Annual returns (last available year)")
    annual = (1 + sector_returns[selected_sectors]).resample("Y").prod() - 1
    if not annual.empty:
        last_year_idx = annual.index[-1]
        df_bar = annual.loc[last_year_idx].reset_index()
        df_bar.columns = ['Sector','AnnualReturn']
        df_bar = df_bar.sort_values(by='AnnualReturn', ascending=False)
        fig_bar = px.bar(df_bar, x='AnnualReturn', y='Sector', orientation='h', title=f"Annual returns {last_year_idx.year}")
        fig_bar.update_layout(height=380, margin=dict(l=120,r=20,t=40,b=40))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Not enough data to compute annual returns for the selected range.")

# ----------------------
# Sector Deep Dive Tab
# ----------------------
with tab_deep:
    st.subheader("Sector Deep Dive")
    # select one sector for deep analysis
    sector_choice = st.selectbox("Select sector for deep dive", options=sector_list_all, index=0)
    sec_returns = sector_returns[sector_choice].dropna()
    if sec_returns.empty:
        st.info("No returns for selected sector in this range.")
    else:
        colA, colB = st.columns([2,1])
        with colA:
            # sector vs benchmark (constructed equal-weighted NIFTY like)
            bench = None
            try:
                import yfinance as yf
                idx = yf.download("^NSEI", start=sector_returns.index.min().strftime("%Y-%m-%d"), end=sector_returns.index.max().strftime("%Y-%m-%d"), progress=False, auto_adjust=False)
                bench = idx['Adj Close'].pct_change().loc[sector_returns.index].fillna(0)
            except Exception:
                bench = None
            df_plot = pd.DataFrame({sector_choice: (1+sec_returns).cumprod()-1})
            if bench is not None:
                df_plot['NIFTY'] = (1+bench).cumprod()-1
            fig2 = px.line(df_plot, x=df_plot.index, y=df_plot.columns, title=f"{sector_choice} vs NIFTY")
            fig2.update_layout(hovermode='x unified', legend=dict(orientation="h", y=1.02, x=1), height=420, margin=dict(l=40,r=20,t=40,b=40))
            st.plotly_chart(fig2, use_container_width=True)

        with colB:
            # volatility chart: rolling 60-day vol
            from metrics_india import rolling_volatility
            roll_vol = rolling_volatility(sec_returns, window=60)
            figv = px.line(roll_vol, x=roll_vol.index, y=roll_vol, title=f"Rolling 60-day vol — {sector_choice}")
            figv.update_layout(height=300, margin=dict(l=10,r=10,t=35,b=20))
            st.plotly_chart(figv, use_container_width=True)

        # show top stocks table and concentration (from price panel)
        with st.expander("Top/bottom stocks & concentration (sector)"):
            # top contributors
            from metrics_india import top_contributors
            price_panel = None
            try:
                from sector_aggregation import load_price_panel
                tickers = load_constituents()['Ticker'].tolist()
                price_panel = load_price_panel(tickers)
            except Exception:
                price_panel = None
            if price_panel is not None:
                tick_in_sec = load_constituents()[load_constituents()['Sector']==sector_choice]['Ticker'].tolist()
                if tick_in_sec:
                    contrib = top_contributors(price_panel[tick_in_sec], top_n=10)
                    st.write("Top contributors (cumulative returns):")
                    st.dataframe(contrib.to_frame(sector_choice).rename(columns={0:'CumulativeReturn'}).style.format("{:.2%}"))
                else:
                    st.info("No tickers found for this sector in constituents file.")
            else:
                st.info("Price panel not available to compute contributors.")

# ----------------------
# Risk-Return Tab (NEW)
# ----------------------
with tab_rr:
    st.subheader("Risk–Return Scatter (Sectors)")

    # Controls
    col1, col2, col3 = st.columns([2,2,2])
    with col1:
        rr_start = st.date_input("Start date (risk-return)", value=pd.to_datetime(start_date))
        rr_end = st.date_input("End date (risk-return)", value=pd.to_datetime(end_date))
    with col2:
        sectors_for_rr = st.multiselect("Sectors to include", options=sector_list_all, default=selected_sectors[:min(len(selected_sectors),8)])
    with col3:
        bubble_scale = st.slider("Bubble size scale", min_value=10, max_value=200, value=60)

    # Subset the returns to selected timeframe
    rr_returns = sector_returns.loc[(sector_returns.index >= pd.to_datetime(rr_start)) & (sector_returns.index <= pd.to_datetime(rr_end))]
    if rr_returns.empty:
        st.info("No data in the chosen timeframe.")
    else:
        # restrict to selected sectors
        rr_returns = rr_returns[sectors_for_rr]

        # compute metrics per sector
        from metrics_india import annualized_volatility, sharpe_ratio, cumulative_returns_from_returns

        # Annualized return approximation from daily returns: ((1+daily).prod())**(252/ndays)-1
        def annualized_return_from_returns(returns):
            # returns: pd.Series daily returns
            returns = returns.dropna()
            if returns.empty:
                return float('nan')
            total_period = (1 + returns).prod() - 1
            # compute annualized from period length
            ndays = returns.shape[0]
            if ndays <= 0:
                return float('nan')
            years = ndays / 252.0
            if years <= 0:
                return float('nan')
            return (1 + total_period) ** (1/years) - 1

        ann_rets = rr_returns.apply(lambda s: annualized_return_from_returns(s))
        ann_vols = rr_returns.apply(lambda s: annualized_volatility(s.dropna()))
        sharpes = rr_returns.apply(lambda s: sharpe_ratio(s.dropna(), risk_free_rate=0.06))
        cumrets = (1 + rr_returns).cumprod().iloc[-1] - 1

        # Build DataFrame for plotting
        df_rr = pd.DataFrame({
            "sector": ann_rets.index,
            "annual_return": ann_rets.values,
            "annual_vol": ann_vols.values,
            "sharpe": sharpes.values,
            "cum_return": cumrets.values
        }).dropna(subset=["annual_return","annual_vol"])

        # Bubble size scaling (make positive)
        df_rr['size'] = (df_rr['cum_return'].abs() + 0.01) * bubble_scale

        # Plotly scatter
        fig_rr = px.scatter(
            df_rr,
            x='annual_vol',
            y='annual_return',
            size='size',
            color='sharpe',
            hover_name='sector',
            hover_data={
                'annual_vol': ':.2%',
                'annual_return': ':.2%',
                'sharpe': ':.2f',
                'cum_return': ':.1%'
            },
            labels={'annual_vol':'Annualized Volatility','annual_return':'Annualized Return','sharpe':'Sharpe'},
            color_continuous_scale='RdYlGn',
            title=f"Risk–Return scatter ({rr_start} → {rr_end})"
        )

        # Add a vertical line at median vol and horizontal line at 0 return
        median_vol = df_rr['annual_vol'].median()
        fig_rr.add_vline(x=median_vol, line_dash="dash", line_color="grey", annotation_text="Median vol", annotation_position="top left")
        fig_rr.add_hline(y=0, line_dash="dash", line_color="black")

        fig_rr.update_layout(height=650, margin=dict(l=40,r=20,t=60,b=40), coloraxis_colorbar=dict(title="Sharpe"))
        fig_rr.update_traces(marker=dict(opacity=0.85, line=dict(width=0.5, color='DarkSlateGrey')))

        st.plotly_chart(fig_rr, use_container_width=True)

        # Table summary below
        st.markdown("### Table: Risk & Return summary")
        df_show = df_rr.copy()
        df_show['annual_vol'] = df_show['annual_vol'].map(lambda x: f"{x:.2%}")
        df_show['annual_return'] = df_show['annual_return'].map(lambda x: f"{x:.2%}")
        df_show['cum_return'] = df_show['cum_return'].map(lambda x: f"{x:.1%}")
        df_show['sharpe'] = df_show['sharpe'].map(lambda x: f"{x:.2f}")
        st.dataframe(df_show.set_index('sector').rename(columns={
            'annual_vol':'Ann Vol','annual_return':'Ann Ret','cum_return':'Cumulative','sharpe':'Sharpe'
        }), height=300)

# ----------------------
# Correlation & Network Tab (NEW)
# ----------------------
with tab_corr:
    st.subheader("Correlation & Network — Sector Relationships")

    # quick controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2,2,2])
    with col_ctrl1:
        corr_method = st.selectbox("Correlation method", options=["pearson","spearman"], index=0)
    with col_ctrl2:
        agg_freq = st.selectbox("Resample frequency for correlation (useful to smooth)", options=["D","W","M"], index=1)  # weekly default
    with col_ctrl3:
        corr_threshold = st.slider("Network edge threshold (abs correlation)", min_value=0.3, max_value=0.95, value=0.6, step=0.05)

    # compute resampled returns if needed
    # sector_returns is daily returns DataFrame
    # resample returns to chosen freq then compute pct-change aggregated returns
    if agg_freq == "D":
        corr_returns = sector_returns.copy()
    else:
        # compute period returns: product(1+returns)-1 across period
        corr_returns = (1 + sector_returns).resample(agg_freq).prod() - 1

    # compute correlation matrix
    corr_mat = corr_returns.corr(method=corr_method).fillna(0)

    st.markdown("### Correlation heatmap")
    # display heatmap
    fig_heat = px.imshow(corr_mat,
                        labels=dict(x="Sector", y="Sector", color="Correlation"),
                        x=corr_mat.columns,
                        y=corr_mat.index,
                        zmin=-1, zmax=1,
                        color_continuous_scale='RdBu_r',
                        aspect="auto",
                        title=f"Sector correlation matrix ({corr_method}, resample={agg_freq})")
    fig_heat.update_layout(height=650, margin=dict(l=60,r=60,t=60,b=60))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Rolling correlation explorer
    st.markdown("### Rolling correlation (two sectors)")
    col_a, col_b = st.columns([2,3])
    with col_a:
        sector_a = st.selectbox("Sector A", options=sector_list_all, index=0)
        sector_b = st.selectbox("Sector B", options=sector_list_all, index=1)
    with col_b:
        window_days = st.slider("Rolling window (days)", min_value=20, max_value=252, value=60, step=10)

    if sector_a and sector_b:
        series_a = sector_returns[sector_a].dropna()
        series_b = sector_returns[sector_b].dropna()
        # align indexes
        joined = pd.concat([series_a, series_b], axis=1).dropna()
        if joined.empty:
            st.info("Not enough overlapping data for these two sectors.")
        else:
            rolling_corr = joined[sector_a].rolling(window_days).corr(joined[sector_b])
            fig_rc = px.line(x=rolling_corr.index, y=rolling_corr.values, labels={'x':'Date','y':'Rolling correlation'}, title=f"Rolling correlation ({sector_a} vs {sector_b}) — window={window_days} days")
            fig_rc.update_layout(yaxis=dict(range=[-1,1]), height=300, margin=dict(l=40,r=20,t=30,b=20))
            st.plotly_chart(fig_rc, use_container_width=True)

    # Network graph: build edges where abs(corr) >= threshold
    st.markdown("### Sector correlation network (edges show |corr| >= threshold)")
    import networkx as nx
    G = nx.Graph()
    cols = corr_mat.columns.tolist()

    # node attributes: cumulative return (for size) and average volatility
    cum_all = (1 + sector_returns).cumprod().iloc[-1] - 1
    ann_vol_all = sector_returns.apply(lambda s: s.std() * (252**0.5))
    # add nodes
    for c in cols:
        G.add_node(c, cum_return=float(cum_all.get(c, 0)), vol=float(ann_vol_all.get(c, 0)))

    # edges for pairs above threshold
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            val = corr_mat.at[a,b]
            if abs(val) >= corr_threshold:
                G.add_edge(a,b, weight=float(val))

    if G.number_of_edges() == 0:
        st.info("No edges above threshold — reduce threshold to see relationships.")
    else:
        # compute layout (spring layout) using correlation magnitude as inverse distance
        pos = nx.spring_layout(G, k=0.8, weight='weight', seed=42)

        # build Plotly traces for edges and nodes
        edge_x = []
        edge_y = []
        edge_width = []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_width.append(abs(d['weight']))

        node_x = []
        node_y = []
        node_text = []
        node_size = []
        for n, d in G.nodes(data=True):
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            # tooltip
            node_text.append(f"{n}<br>Cumulative: {d.get('cum_return',0):.1%}<br>AnnVol: {d.get('vol',0):.1%}")
            # size scaled by cumulative return magnitude (plus a minimum)
            size = max(8, 40 * (abs(d.get('cum_return',0)) + 0.01))
            node_size.append(size)

        # edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[n for n in G.nodes()],
            textposition="top center",
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=False,
                color=[d['cum_return'] for _,d in G.nodes(data=True)],
                size=node_size,
                colorbar=dict(title="Cum return"),
                line_width=1
            )
        )

        fig_net = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title=f"Sector correlation network (threshold={corr_threshold})",
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                height=650,
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                            ))
        st.plotly_chart(fig_net, use_container_width=True)


# ----------------------
# Monthly Heatmap Tab
# ----------------------
with tab_heat:
    st.subheader("Monthly returns heatmap")
    # pick sector to view heatmap
    sel = st.selectbox("Sector for heatmap", options=sector_list_all, index=0)
    monthly = (1 + sector_returns[sel]).resample("M").prod() - 1
    if monthly.empty:
        st.info("No monthly data for selected sector.")
    else:
        dfm = monthly.reset_index()
        date_col = dfm.columns[0]
        dfm[date_col] = pd.to_datetime(dfm[date_col])
        dfm['Year'] = dfm[date_col].dt.year
        dfm['Month'] = dfm[date_col].dt.month
        pivot = dfm.pivot(index='Year', columns='Month', values=sel).fillna(0)
        pivot.columns = [datetime(2000, m, 1).strftime('%b') for m in pivot.columns]
        fig3 = px.imshow(pivot, text_auto=".1%", aspect="auto", title=f"Monthly returns — {sel}")
        fig3.update_layout(height=480, margin=dict(l=40,r=20,t=40,b=40))
        st.plotly_chart(fig3, use_container_width=True)

# ----------------------
# Download / Audit Tab
# ----------------------
with tab_down:
    st.subheader("Data & audit")
    st.write("Files in data/ (precomputed or saved by pipeline):")
    files = list((DATA_DIR).glob("*"))
    files = [f.name for f in files if f.is_file()]
    st.dataframe(pd.DataFrame({"files":files}).sort_values("files"), height=300)
    st.markdown("**Quick checks**:")
    st.write("- If graphs look empty, re-run `python data_pipeline_india.py` to refresh data.")
    if (DATA_DIR / "sector_mcap_daily_returns.parquet").exists():
        st.download_button("Download sector_mcap_daily_returns.parquet", data=open(DATA_DIR / "sector_mcap_daily_returns.parquet","rb"), file_name="sector_mcap_daily_returns.parquet")
    if (DATA_DIR / "sector_mcap_cum_last.csv").exists():
        st.download_button("Download sector_mcap_cum_last.csv", data=open(DATA_DIR / "sector_mcap_cum_last.csv","rb"), file_name="sector_mcap_cum_last.csv")

st.caption("Tip: Use the 'Quick range' selector and change 'Max sectors to display' to reduce clutter and avoid overlapping lines.")
