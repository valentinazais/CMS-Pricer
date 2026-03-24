import streamlit as st
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import norm

st.set_page_config(layout="wide", page_title="CMS Spread Option Pricer")

def build_zero_curve(tenors, yields):
    t = np.concatenate([[0.0], tenors])
    y = np.concatenate([[yields[0]], yields])
    return CubicSpline(t, y, bc_type='natural')

def disc(T, zc):
    if T <= 0:
        return 1.0
    return np.exp(-float(zc(T)) * T)

def forward_swap_rate(T0, tenor, zc, freq=0.5):
    n = int(tenor / freq)
    times = [T0 + freq * (i + 1) for i in range(n)]
    ann = sum(freq * disc(t, zc) for t in times)
    if ann < 1e-12:
        return 0.0
    return (disc(T0, zc) - disc(times[-1], zc)) / ann

def price_cms_spread_normal(S1, S2, vol1, vol2, rho, K, T, df, is_call):
    spread = S1 - S2
    vol_spread = np.sqrt(vol1**2 + vol2**2 - 2 * rho * vol1 * vol2)
    if vol_spread * np.sqrt(T) < 1e-12:
        if is_call:
            return df * max(spread - K, 0.0)
        else:
            return df * max(K - spread, 0.0)
    std = vol_spread * np.sqrt(T)
    d = (spread - K) / std
    if is_call:
        return df * (std * norm.pdf(d) + (spread - K) * norm.cdf(d))
    else:
        return df * (std * norm.pdf(d) - (spread - K) * norm.cdf(-d))

st.title("CMS Spread Option Pricer")

with st.sidebar:
    st.header("Yield Curve")
    r1y = st.slider("1Y", 0.0, 10.0, 3.5, 0.05)
    r2y = st.slider("2Y", 0.0, 10.0, 3.3, 0.05)
    r5y = st.slider("5Y", 0.0, 10.0, 3.0, 0.05)
    r10y = st.slider("10Y", 0.0, 10.0, 2.8, 0.05)
    r20y = st.slider("20Y", 0.0, 10.0, 2.7, 0.05)
    r30y = st.slider("30Y", 0.0, 10.0, 2.6, 0.05)

    st.header("CMS Legs")
    tenor1 = st.selectbox("CMS Tenor 1 (Y)", [2, 5, 10, 20, 30], index=3)
    tenor2 = st.selectbox("CMS Tenor 2 (Y)", [2, 5, 10, 20, 30], index=1)

    st.header("Option")
    T_expiry = st.slider("Expiry (Y)", 0.25, 10.0, 1.0, 0.25)
    T_pay = st.slider("Payment (Y)", 0.25, 12.0, T_expiry, 0.25)
    K_bps = st.slider("Strike (bps)", -200, 300, 0, 5)
    is_call = st.radio("Type", ["Call", "Put"]) == "Call"
    notional = st.number_input("Notional", value=10_000_000, step=1_000_000)

    st.header("Volatility (Normal, bps)")
    vol1_bps = st.slider(f"Vol CMS{tenor1}Y", 10, 200, 80, 5)
    vol2_bps = st.slider(f"Vol CMS{tenor2}Y", 10, 200, 70, 5)
    rho_spread = st.slider("Correlation", -0.5, 0.99, 0.85, 0.01)

tenors = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
yields = np.array([r1y, r2y, r5y, r10y, r20y, r30y]) / 100.0
zc = build_zero_curve(tenors, yields)

cms1 = forward_swap_rate(T_expiry, tenor1, zc)
cms2 = forward_swap_rate(T_expiry, tenor2, zc)
spread_fwd = cms1 - cms2
K = K_bps / 10000.0
vol1_n = vol1_bps / 10000.0
vol2_n = vol2_bps / 10000.0
df_pay = disc(T_pay, zc)

price = price_cms_spread_normal(cms1, cms2, vol1_n, vol2_n, rho_spread, K, T_expiry, df_pay, is_call)
price_total = price * notional

st.subheader("Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"CMS{tenor1}Y Fwd", f"{cms1*10000:.1f} bps")
c2.metric(f"CMS{tenor2}Y Fwd", f"{cms2*10000:.1f} bps")
c3.metric("Spread Fwd", f"{spread_fwd*10000:.1f} bps")
c4.metric("Discount", f"{df_pay:.6f}")

spread_vol = np.sqrt(vol1_n**2 + vol2_n**2 - 2*rho_spread*vol1_n*vol2_n)
d1, d2, d3 = st.columns(3)
d1.metric("Spread Vol (bps)", f"{spread_vol*10000:.1f}")
d2.metric("Unit Price (bps)", f"{price*10000:.2f}")
d3.metric("Total Price", f"{price_total:,.0f}")

st.subheader("Strike Ladder")
strikes_bps = list(range(-150, 175, 25))
rows = []
for sb in strikes_bps:
    k = sb / 10000.0
    pc = price_cms_spread_normal(cms1, cms2, vol1_n, vol2_n, rho_spread, k, T_expiry, df_pay, True)
    pp = price_cms_spread_normal(cms1, cms2, vol1_n, vol2_n, rho_spread, k, T_expiry, df_pay, False)
    rows.append({"Strike (bps)": sb, "Call (bps)": round(pc*10000, 2), "Put (bps)": round(pp*10000, 2)})
st.dataframe(rows, use_container_width=True)

st.subheader("Price vs Correlation")
import pandas as pd
rhos = np.linspace(-0.5, 0.99, 30)
prices_rho = [price_cms_spread_normal(cms1, cms2, vol1_n, vol2_n, r, K, T_expiry, df_pay, is_call)*10000 for r in rhos]
st.line_chart(pd.DataFrame({"rho": rhos, "Price (bps)": prices_rho}).set_index("rho"))
