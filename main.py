import streamlit as st
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.integrate import quad

st.set_page_config(layout="wide", page_title="CMS Spread Option Pricer")

# ─── Yield Curve ───

def build_zero_curve(tenors, yields):
    t = np.concatenate([[0.0], tenors])
    y = np.concatenate([[yields[0]], yields])
    return CubicSpline(t, y, bc_type='natural')

def discount(T, zc):
    if T <= 0:
        return 1.0
    return np.exp(-float(zc(T)) * T)

def forward_swap_rate(T0, tenor, zc, freq=0.5):
    n = int(tenor / freq)
    times = [T0 + freq * (i + 1) for i in range(n)]
    annuity = sum(freq * discount(t, zc) for t in times)
    if annuity < 1e-12:
        return 0.0
    return (discount(T0, zc) - discount(times[-1], zc)) / annuity

def annuity_value(T0, tenor, zc, freq=0.5):
    n = int(tenor / freq)
    times = [T0 + freq * (i + 1) for i in range(n)]
    return sum(freq * discount(t, zc) for t in times)

# ─── SABR ───

def sabr_normal_vol(F, K, T, alpha, beta, rho, nu):
    if T <= 0:
        return alpha
    if abs(F - K) < 1e-8:
        FK_mid = F
        factor1 = alpha * FK_mid ** (beta - 1.0)
        term1 = ((beta - 1.0) ** 2 / 24.0) * alpha ** 2 * FK_mid ** (2 * beta - 2)
        term2 = 0.25 * rho * beta * nu * alpha * FK_mid ** (beta - 1)
        term3 = (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
        vol_n = alpha * F ** beta * (1.0 + (term1 + term2 + term3) * T)
        return vol_n
    FK = F * K
    if FK <= 0:
        FK = 1e-10
    FK_beta = FK ** ((1 - beta) / 2)
    log_FK = np.log(F / K)
    z = (nu / alpha) * FK_beta * log_FK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
    if abs(x_z) < 1e-10:
        x_z = 1e-10
    prefix = alpha / (FK_beta * (1 + ((1 - beta) ** 2 / 24) * log_FK ** 2 + ((1 - beta) ** 4 / 1920) * log_FK ** 4))
    body = z / x_z
    correction = 1 + (((1 - beta) ** 2 / 24) * (alpha ** 2 / FK ** (1 - beta)) + 0.25 * rho * beta * nu * alpha / FK_beta + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T
    lognormal_vol = prefix * body * correction
    normal_vol = lognormal_vol * F
    return normal_vol

# ─── Convexity Adjustment (Hagan-style) ───

def cms_convexity_adjustment(S_fwd, T_expiry, tenor, sigma_atm_n, zc, freq=0.5):
    A = annuity_value(T_expiry, tenor, zc, freq)
    if A < 1e-12:
        return 0.0
    n = int(tenor / freq)
    dA_dS = 0.0
    for i in range(n):
        ti = T_expiry + freq * (i + 1)
        dA_dS += -freq * ti * discount(ti, zc)
    d2A_dS2 = 0.0
    for i in range(n):
        ti = T_expiry + freq * (i + 1)
        d2A_dS2 += freq * ti ** 2 * discount(ti, zc)
    G_prime = -dA_dS / A
    G_double_prime = d2A_dS2 / A - (dA_dS / A) ** 2
    var = (sigma_atm_n ** 2) * T_expiry
    adjustment = S_fwd ** 2 * G_double_prime * var / (2 * A)
    return adjustment

# ─── CMS Spread Option Pricing (Gaussian Copula) ───

def price_cms_spread_option(cms1, cms2, vol1, vol2, rho, K, T, df_pay, is_call=True, n_points=200):
    if T <= 0:
        return 0.0
    std1 = vol1 * np.sqrt(T)
    std2 = vol2 * np.sqrt(T)
    if std1 < 1e-10 or std2 < 1e-10:
        spread = cms1 - cms2
        if is_call:
            return df_pay * max(spread - K, 0)
        else:
            return df_pay * max(K - spread, 0)

    x = np.linspace(-5, 5, n_points)
    w = np.diff(norm.cdf(x))
    x_mid = 0.5 * (x[:-1] + x[1:])

    price = 0.0
    for i in range(len(x_mid)):
        z1 = x_mid[i]
        s1 = cms1 + std1 * z1
        cond_mean = cms2 + rho * (std2 / std1) * (s1 - cms1)
        cond_std = std2 * np.sqrt(max(1 - rho ** 2, 0))

        if is_call:
            if cond_std < 1e-12:
                ev = max(s1 - cond_mean - K, 0)
            else:
                d = (s1 - cond_mean - K) / cond_std
                ev = (s1 - cond_mean - K) * norm.cdf(d) + cond_std * norm.pdf(d)
        else:
            if cond_std < 1e-12:
                ev = max(K - s1 + cond_mean, 0)
            else:
                d = (s1 - cond_mean - K) / cond_std
                ev = (K - s1 + cond_mean) * norm.cdf(-d) + cond_std * norm.pdf(d)

        price += ev * w[i]

    return df_pay * price

# ─── Streamlit UI ───

st.title("CMS Spread Option Pricer")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Yield Curve")
    tenors_default = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
    yields_default = [3.80, 3.75, 3.60, 3.45, 3.35, 3.25, 3.20, 3.15, 3.10, 3.08, 3.05]
    curve_text = st.text_area("Tenors,Yields (one per line)",
                              value="\n".join(f"{t},{y}" for t, y in zip(tenors_default, yields_default)),
                              height=200)
    tenors, yields_input = [], []
    for line in curve_text.strip().split("\n"):
        parts = line.split(",")
        if len(parts) == 2:
            tenors.append(float(parts[0]))
            yields_input.append(float(parts[1]))
    tenors = np.array(tenors)
    yields_arr = np.array(yields_input) / 100.0
    zc = build_zero_curve(tenors, yields_arr)

    st.subheader("CMS Legs")
    T_expiry = st.slider("Option Expiry (Y)", 0.5, 10.0, 5.0, 0.25)
    tenor1 = st.slider("CMS Tenor 1 (Y)", 1, 30, 10, 1)
    tenor2 = st.slider("CMS Tenor 2 (Y)", 1, 30, 2, 1)
    strike_bps = st.slider("Strike (bps)", -200, 200, 0, 5)
    K = strike_bps / 10000.0
    is_call = st.radio("Type", ["Call (spread > K)", "Put (spread < K)"]) == "Call (spread > K)"
    notional = st.number_input("Notional (M)", value=100.0, step=10.0) * 1e6

    st.subheader("SABR (CMS Leg 1)")
    alpha1 = st.slider("α₁", 0.005, 0.10, 0.03, 0.001)
    beta1 = st.slider("β₁", 0.0, 1.0, 0.5, 0.05)
    rho1 = st.slider("ρ₁", -0.8, 0.8, -0.2, 0.05)
    nu1 = st.slider("ν₁", 0.05, 1.0, 0.4, 0.05)

    st.subheader("SABR (CMS Leg 2)")
    alpha2 = st.slider("α₂", 0.005, 0.10, 0.03, 0.001)
    beta2 = st.slider("β₂", 0.0, 1.0, 0.5, 0.05)
    rho2 = st.slider("ρ₂", -0.8, 0.8, -0.2, 0.05)
    nu2 = st.slider("ν₂", 0.05, 1.0, 0.4, 0.05)

    st.subheader("Correlation")
    rho_spread = st.slider("ρ(CMS1, CMS2)", -1.0, 1.0, 0.85, 0.01)

with col_right:
    S1_fwd = forward_swap_rate(T_expiry, tenor1, zc)
    S2_fwd = forward_swap_rate(T_expiry, tenor2, zc)

    vol1_n = sabr_normal_vol(S1_fwd, S1_fwd, T_expiry, alpha1, beta1, rho1, nu1)
    vol2_n = sabr_normal_vol(S2_fwd, S2_fwd, T_expiry, alpha2, beta2, rho2, nu2)

    ca1 = cms_convexity_adjustment(S1_fwd, T_expiry, tenor1, vol1_n, zc)
    ca2 = cms_convexity_adjustment(S2_fwd, T_expiry, tenor2, vol2_n, zc)

    cms1 = S1_fwd + ca1
    cms2 = S2_fwd + ca2

    df_pay = discount(T_expiry, zc)

    price_unit = price_cms_spread_option(cms1, cms2, vol1_n, vol2_n, rho_spread, K, T_expiry, df_pay, is_call)
    price_total = price_unit * notional
    price_bps = price_unit * 10000

    st.subheader("Forward Rates & CMS Adjustments")
    res = {
        f"Fwd Swap {tenor1}Y": f"{S1_fwd * 10000:.1f} bps",
        f"Fwd Swap {tenor2}Y": f"{S2_fwd * 10000:.1f} bps",
        f"Convexity Adj {tenor1}Y": f"{ca1 * 10000:.2f} bps",
        f"Convexity Adj {tenor2}Y": f"{ca2 * 10000:.2f} bps",
        f"CMS {tenor1}Y": f"{cms1 * 10000:.1f} bps",
        f"CMS {tenor2}Y": f"{cms2 * 10000:.1f} bps",
        "Fwd Spread": f"{(S1_fwd - S2_fwd) * 10000:.1f} bps",
        "CMS Spread (adj)": f"{(cms1 - cms2) * 10000:.1f} bps",
    }
    for k, v in res.items():
        st.metric(k, v)

    st.subheader("Pricing")
    c1, c2, c3 = st.columns(3)
    c1.metric("Price (bps running)", f"{price_bps:.2f}")
    c2.metric(f"Price (notional {notional / 1e6:.0f}M)", f"{price_total:,.0f}")
    c3.metric("Discount Factor", f"{df_pay:.6f}")

    st.subheader("Vol & Distribution")
    v1, v2 = st.columns(2)
    v1.metric(f"ATM Normal Vol {tenor1}Y (bps)", f"{vol1_n * 10000:.1f}")
    v2.metric(f"ATM Normal Vol {tenor2}Y (bps)", f"{vol2_n * 10000:.1f}")

    spread_vol = np.sqrt(vol1_n ** 2 + vol2_n ** 2 - 2 * rho_spread * vol1_n * vol2_n)
    st.metric("Implied Spread Vol (bps)", f"{spread_vol * 10000:.1f}")

    st.subheader("Strike Ladder")
    strikes_bps = np.arange(-150, 175, 25)
    ladder = []
    for sb in strikes_bps:
        k = sb / 10000.0
        pc = price_cms_spread_option(cms1, cms2, vol1_n, vol2_n, rho_spread, k, T_expiry, df_pay, True)
        pp = price_cms_spread_option(cms1, cms2, vol1_n, vol2_n, rho_spread, k, T_expiry, df_pay, False)
        ladder.append({"Strike (bps)": sb, "Call (bps)": round(pc * 10000, 2), "Put (bps)": round(pp * 10000, 2),
                        "Call-Put (bps)": round((pc - pp) * 10000, 2)})
    st.dataframe(ladder, use_container_width=True)

    st.subheader("Sensitivity to Correlation")
    rhos = np.linspace(-0.5, 0.99, 30)
    prices_rho = []
    for r in rhos:
        p = price_cms_spread_option(cms1, cms2, vol1_n, vol2_n, r, K, T_expiry, df_pay, is_call)
        prices_rho.append(p * 10000)
    chart_data = {"ρ": rhos, "Price (bps)": prices_rho}
    st.line_chart(chart_data, x="ρ", y="Price (bps)")

    st.subheader("Spread Distribution at Expiry")
    z = np.linspace(-4, 4, 500)
    spread_mean = cms1 - cms2
    spread_std = spread_vol * np.sqrt(T_expiry)
    spread_vals = (spread_mean + spread_std * z) * 10000
    density = norm.pdf(z)
    st.line_chart({"Spread (bps)": spread_vals, "Density": density}, x="Spread (bps)", y="Density")
