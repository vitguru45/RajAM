"""
RajAM  —  Hardness & microstructure predictor for laser additive manufacturing of steels
(SS316L, 18Ni-300 maraging, 17-4PH, H13) made by LPBF/SLM and L-DED.

Developed by Dr. Gururaja S, Centre for Innovative Manufacturing Research (CIMR), VIT Vellore.
Run:  streamlit run app.py
"""
import datetime as dt
import hashlib
import io
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import physics as ph
import models as M

APP_DIR = os.path.dirname(os.path.abspath(__file__))
COUNTER_FILE = os.path.join(APP_DIR, "rajam_counter.json")
LOG_FILE = os.path.join(APP_DIR, "rajam_usage_log.csv")
MODEL_NAMES = ["Analytical", "Random Forest", "Gradient Boosting", "Hybrid GPR", "Ensemble"]
COLORS = {"Analytical": "#2a6fdb", "Random Forest": "#1f9d55", "Gradient Boosting": "#e07b00",
          "Hybrid GPR": "#8e44ad", "Ensemble": "#c0392b"}

st.set_page_config(page_title="RajAM | AM Steel Hardness Predictor", page_icon="🔩", layout="wide")

# ----------------------------------------------------------------------------- style
st.markdown("""
<style>
.raj-header{background:linear-gradient(90deg,#0b3d91 0%,#1565c0 55%,#00897b 100%);
  padding:18px 24px;border-radius:14px;color:#fff;margin-bottom:10px}
.raj-header h1{margin:0;font-size:2.0rem;color:#fff}
.raj-header p{margin:4px 0 0 0;opacity:.92}
.raj-card{border:1px solid rgba(128,128,128,.25);border-radius:12px;padding:12px 14px;margin-bottom:8px}
.raj-badge{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:600;font-size:.85rem;color:#fff}
.raj-foot{text-align:center;opacity:.75;font-size:.85rem;margin-top:12px}
.raj-counter{display:flex;justify-content:center;gap:14px;flex-wrap:wrap;margin-top:36px;padding-top:16px;
  border-top:1px solid rgba(128,128,128,.25)}
.raj-counter .box{background:linear-gradient(90deg,#0b3d91,#00897b);color:#fff;border-radius:10px;
  padding:8px 18px;text-align:center;min-width:130px}
.raj-counter .num{font-size:1.45rem;font-weight:700;font-family:monospace;letter-spacing:2px}
.raj-counter .lab{font-size:.75rem;opacity:.9}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- counter + log
def _read_counter():
    try:
        with open(COUNTER_FILE) as f:
            return json.load(f)
    except Exception:
        return {"visits": 0, "predictions": 0, "since": dt.date.today().isoformat()}


def _write_counter(c):
    try:
        tmp = COUNTER_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(c, f)
        os.replace(tmp, COUNTER_FILE)
    except Exception:
        pass


def bump(key):
    c = _read_counter()
    c[key] = c.get(key, 0) + 1
    today = dt.date.today().isoformat()
    if key == "visits":
        if c.get("today_date") != today:
            c["today_date"], c["today_visits"] = today, 0
        c["today_visits"] = c.get("today_visits", 0) + 1
    _write_counter(c)
    return c


if "visited" not in st.session_state:
    st.session_state.visited = True
    counter = bump("visits")
else:
    counter = _read_counter()


def log_use(row):
    try:
        pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)
    except Exception:
        pass


# ----------------------------------------------------------------------------- models (cached)
@st.cache_resource(show_spinner="Training analytical calibration + ML models (first load only)…")
def get_bundle(extra_csv_hash=None, extra_csv=None):
    extra = pd.read_csv(io.StringIO(extra_csv)) if extra_csv else None
    df = M.load_dataset(os.path.join(APP_DIR, "rajam_dataset.csv"), extra)
    return M.ModelBundle(df)


extra_csv = st.session_state.get("extra_csv")
bundle = get_bundle(hashlib.md5(extra_csv.encode()).hexdigest() if extra_csv else None, extra_csv)
ph.CAL.update(bundle.cal)

# ----------------------------------------------------------------------------- header
st.markdown("""
<div class="raj-header">
<h1>🔩 RajAM — Hardness & Microstructure Predictor for Laser-AM Steels</h1>
<p>SS316L · 18Ni-300 maraging · 17-4PH · H13 &nbsp;|&nbsp; LPBF/SLM and L-DED &nbsp;|&nbsp;
analytical physics + machine learning, checked against published experiments</p>
</div>""", unsafe_allow_html=True)
h1, h2, h3, h4 = st.columns([2, 1, 1, 1])
with h1:
    user = st.text_input("Your name (optional — appears in your report)", key="user_name",
                         placeholder="e.g. Priya, 23BME1234")
h2.metric("👥 Visits", f"{counter.get('visits', 0):,}")
h3.metric("🔮 Predictions", f"{counter.get('predictions', 0):,}")
h4.metric("📚 Experimental records", len(bundle.df))

# ----------------------------------------------------------------------------- sidebar inputs
sb = st.sidebar
sb.header("① Process")
proc_label = sb.radio("Select the AM process", ["LPBF / SLM (powder bed)", "L-DED (laser directed energy deposition)"])
proc = "LPBF" if proc_label.startswith("LPBF") else "LDED"

sb.header("② Material")
mat_key = sb.selectbox("Steel", M.MATS, format_func=lambda k: ph.MATERIALS[k]["label"])

sb.header("③ Process parameters")
if proc == "LPBF":
    P = sb.slider("Laser power P (W)", 50, 500, 200, 5)
    v = sb.slider("Scan speed v (mm/s)", 100, 3000, 900, 10)
    hatch = sb.slider("Hatch spacing h (µm)", 30, 200, 90, 5)
    layer = sb.slider("Layer thickness t (µm)", 20, 120, 40, 5)
    spot = sb.slider("Laser spot diameter (µm)", 40, 200, 80, 5)
    feed = 10.0
    pre = sb.slider("Build-plate preheat (°C)", 25, 500, 25 if mat_key != "H13" else 200, 5)
else:
    P = sb.slider("Laser power P (W)", 150, 3000, 800, 10)
    v = sb.slider("Traverse speed v (mm/s)", 1.0, 40.0, 10.0, 0.5)
    feed = sb.slider("Powder feed rate (g/min)", 2.0, 40.0, 10.0, 0.5)
    spot = sb.slider("Laser spot diameter (µm)", 300, 3000, 1000, 50)
    hatch = sb.slider("Hatch / track spacing (µm)", 150, 2500, int(0.6 * 1000), 25)
    d_tmp = ph.defects(mat_key, proc, P, v, hatch, np.nan, spot, feed, 25)
    h_dep_um = float(np.ravel(d_tmp["h_dep"])[0]) * 1e6
    layer = sb.slider("Layer height (µm)", 100, 2000, int(np.clip(round(h_dep_um / 10) * 10, 100, 2000)), 10,
                      help=f"Mass-balance estimate of single-track height at these settings ≈ {h_dep_um:.0f} µm")
    pre = sb.slider("Substrate / inter-layer temperature (°C)", 25, 600, 25, 5)

sb.header("④ Post-processing")
ht_opts = ph.HT_OPTIONS[mat_key]
ht = sb.selectbox("Heat treatment", list(ht_opts.keys()), format_func=lambda k: ht_opts[k])
T_def, t_def = ph.HT_DEFAULT_T[mat_key]
if ht in ("DA", "SA", "DT", "QT"):
    T_lo, T_hi = {"18Ni300": (400, 650), "17-4PH": (450, 650), "H13": (400, 700)}[mat_key]
    T_ht = sb.slider("Aging / tempering temperature (°C)", T_lo, T_hi, T_def, 5)
    t_ht = sb.slider("Time (h)", 0.25, 24.0, t_def, 0.25)
else:
    T_ht, t_ht = T_def, t_def

sb.header("⑤ Goal & advanced")
target = sb.number_input("Target hardness (HV) — optional", 0, 900, 0, 5,
                         help="Leave 0 to skip. Used for suggestions and inverse design.")
target = target if target > 0 else None
adv = sb.expander("Advanced physics settings")
A_abs = adv.slider("Absorptivity A", 0.15, 0.80, ph.ABSORPTIVITY[proc], 0.01)
kh_thr = adv.slider("Keyhole threshold ΔH/hs (King et al.)", 10.0, 50.0, 30.0, 1.0)
ball_thr = adv.slider("Balling limit L/W (Plateau–Rayleigh)", 3.0, 8.0, 5.0, 0.1)
comp = None
if mat_key == "SS316L":
    adv.caption("316L composition (wt%) for Creq/Nieq")
    comp = dict(Cr=adv.number_input("Cr", 15.0, 20.0, 17.0, 0.1), Ni=adv.number_input("Ni", 9.0, 15.0, 12.0, 0.1),
                Mo=adv.number_input("Mo", 1.5, 3.5, 2.5, 0.1), Mn=adv.number_input("Mn", 0.0, 2.0, 1.5, 0.1),
                Si=adv.number_input("Si", 0.0, 1.0, 0.5, 0.05), C=adv.number_input("C", 0.0, 0.05, 0.02, 0.005),
                N=adv.number_input("N", 0.0, 0.15, 0.05, 0.01), Nb=0.0, Ti=0.0, Cu=0.0)

# ----------------------------------------------------------------------------- predictions
res = bundle.predict_all(mat_key, proc, P, v, hatch, layer, spot, feed, pre, ht, T_ht, t_ht, kh_thr, A_abs, ball_thr)
phy = res["physics"]
f = lambda k: float(np.ravel(phy[k])[0])
preds = {m: float(np.ravel(res[m])[0]) for m in MODEL_NAMES}
gpr_sd = float(np.ravel(res["gpr_sd"])[0])
n_same, n_same_ht = bundle.domain(mat_key, proc, ht)
sub_mask = (bundle.loo.material == mat_key) & (bundle.loo.process == proc)
met_sub = bundle.metrics(sub_mask) if sub_mask.sum() >= 2 else None
met_all = bundle.metrics()
err_tbl = (met_sub if met_sub is not None else met_all).set_index("Model")

# count a prediction once per distinct input set
sig = (mat_key, proc, P, v, hatch, layer, spot, feed, pre, ht, T_ht, t_ht)
if st.session_state.get("last_sig") != sig:
    st.session_state.last_sig = sig
    counter = bump("predictions")
    log_use(dict(time=dt.datetime.now().isoformat(timespec="seconds"), user=user, material=mat_key, process=proc,
                 P=P, v=v, hatch=hatch, layer=layer, spot=spot, feed=feed, preheat=pre, ht=ht, T=T_ht, t=t_ht,
                 **{f"HV_{k}": round(val, 1) for k, val in preds.items()}))

tabs = st.tabs(["🔮 Predictions", "🗺️ Process map", "🛠️ Suggestions & inverse design", "🌡️ Melt pool & thermal",
                "📊 Model accuracy", "📚 Data, models & references"])

# =============================================================================  TAB 1
with tabs[0]:
    if n_same == 0:
        st.warning(f"No experimental record for **{mat_key} + {proc}** in the database yet → the ML models are "
                   "extrapolating from other alloys/processes. Trust the **Analytical** and **Hybrid GPR** values "
                   "(the GPR falls back to physics and widens its uncertainty).")
    elif n_same_ht == 0:
        st.info(f"{n_same} records exist for {mat_key} + {proc}, but none with this heat treatment — "
                "the heat-treatment effect comes mainly from the analytical model.")

    best = "Hybrid GPR"
    hv = preds[best]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hardness (Hybrid GPR)", f"{hv:.0f} HV", f"± {1.96 * gpr_sd:.0f} HV (95 %)", delta_color="off")
    hrc = float(ph.hv_to_hrc(hv))
    c2.metric("≈ Rockwell C", "—" if np.isnan(hrc) else f"{hrc:.1f} HRC")
    c3.metric("Est. UTS", f"{ph.hv_to_uts(hv):.0f} MPa", f"YS ≈ {ph.hv_to_ys(hv):.0f} MPa", delta_color="off")
    c4.metric("Relative density (analytical)", f"{f('rel_density'):.1f} %")
    if proc == "LPBF":
        ved, led, _ = ph.energy_densities(P, v, hatch, layer, spot)
        c5.metric("VED", f"{float(ved):.0f} J/mm³", f"LED {float(led):.2f} J/mm", delta_color="off")
    else:
        c5.metric("Linear energy", f"{P / v:.0f} J/mm", f"{P / (v * spot * 1e-3):.0f} J/mm² areal", delta_color="off")

    # regime badge
    lof, kh, ball = f("lof"), f("keyhole"), f("balling")
    if lof > 1:
        badge = ("Lack of fusion", "#c0392b")
    elif kh > 1:
        badge = ("Keyhole", "#8e44ad")
    elif ball > 1 and proc == "LPBF":
        badge = ("Balling risk", "#e07b00")
    else:
        badge = ("Conduction mode — dense", "#1f9d55")
    st.markdown(f"Process regime: <span class='raj-badge' style='background:{badge[1]}'>{badge[0]}</span> "
                f"&nbsp; LoF index {lof:.2f} · keyhole ratio {kh:.2f} · balling ratio {ball:.2f}",
                unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Hardness from each model")
        rows = []
        for m in MODEL_NAMES:
            e = err_tbl.loc[m]
            rows.append({"Model": m, "Predicted HV": round(preds[m]), "≈ HRC": ("–" if np.isnan(ph.hv_to_hrc(preds[m])) else f"{float(ph.hv_to_hrc(preds[m])):.1f}"),
                         "Expected error (LOO MAE, HV)": round(e.MAE_HV, 1),
                         "Closeness to experiments": f"{max(0, 100 - e.MAPE_pct):.1f} %",
                         "R² (LOO)": None if np.isnan(e.R2) else round(e.R2, 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        scope = f"{mat_key} + {proc} records (N={int(err_tbl.N.iloc[0])})" if met_sub is not None else "all records (no subset data)"
        st.caption(f"Expected error & closeness = leave-one-out cross-validation against published experiments, for {scope}. "
                   "Closeness = 100 − mean absolute % error.")
        fig = go.Figure()
        fig.add_bar(x=MODEL_NAMES, y=[preds[m] for m in MODEL_NAMES], marker_color=[COLORS[m] for m in MODEL_NAMES],
                    error_y=dict(type="data", array=[err_tbl.loc[m].MAE_HV for m in MODEL_NAMES]),
                    text=[f"{preds[m]:.0f}" for m in MODEL_NAMES], textposition="outside")
        exp = bundle.df[(bundle.df.material == mat_key) & (bundle.df.process == proc) & (bundle.df.ht == ht)]
        if len(exp):
            fig.add_hrect(y0=exp.HV.min(), y1=exp.HV.max(), fillcolor="grey", opacity=0.15, line_width=0,
                          annotation_text="range of published experiments (same alloy/process/HT)", annotation_position="top left")
        if target:
            fig.add_hline(y=target, line_dash="dash", line_color="black", annotation_text=f"target {target} HV")
        fig.update_layout(height=360, yaxis_title="Hardness (HV)", margin=dict(t=30, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Expected microstructure")
        ratio = None
        if comp:
            creq, nieq = ph.creq_nieq(comp)
            ratio = creq / nieq
        elif mat_key == "SS316L":
            creq, nieq = ph.creq_nieq(dict(Cr=17, Ni=12, Mo=2.5, Mn=1.5, Si=0.5, C=0.02, N=0.05, Nb=0, Ti=0, Cu=0))
            ratio = creq / nieq
        for line in ph.microstructure_text(mat_key, proc, ht, phy, T_ht, t_ht, ratio):
            st.markdown(line)
        st.subheader("Other predicted outcomes")
        oth = pd.DataFrame([
            ("Melt-pool width", f"{f('W') * 1e6:.0f} µm"), ("Melt-pool depth", f"{f('D') * 1e6:.0f} µm"),
            ("Melt-pool length", f"{f('L') * 1e6:.0f} µm"), ("Cooling rate at solidus", f"{f('cr'):.2e} K/s"),
            ("Thermal gradient G", f"{f('G'):.2e} K/m"), ("Solidification rate R", f"{f('R'):.3f} m/s"),
            ("G/R (morphology)", f"{f('GR'):.2e} K·s/m²"), ("Cell / dendrite spacing λ", f"{f('lam'):.2f} µm"),
            ("As-built hardness (analytical)", f"{f('HV_ab'):.0f} HV"), ("Normalised enthalpy ΔH/hs", f"{f('dHhs'):.1f}"),
            ("Porosity (analytical)", f"{f('porosity'):.1f} %"),
        ] + ([("Powder-melting energy ratio", f"{f('melt_ratio'):.2f}"),
              ("Track height (mass balance)", f"{f('h_dep') * 1e6:.0f} µm")] if proc == "LDED" else []),
            columns=["Quantity", "Value"])
        st.dataframe(oth, hide_index=True, width="stretch")

    report = pd.DataFrame([dict(name=user, date=dt.date.today().isoformat(), material=mat_key, process=proc, P_W=P,
                                v_mms=v, hatch_um=hatch, layer_um=layer, spot_um=spot, feed_gmin=feed, preheat_C=pre,
                                heat_treatment=ht_opts[ht], T_C=T_ht, t_h=t_ht,
                                **{f"HV_{k}": round(val, 1) for k, val in preds.items()},
                                rel_density_pct=round(f("rel_density"), 2), cooling_rate_Ks=f("cr"),
                                cell_spacing_um=round(f("lam"), 3), regime=badge[0])])
    st.download_button("⬇️ Download my prediction (CSV)", report.to_csv(index=False).encode(),
                       f"RajAM_{(user or 'prediction').replace(' ', '_')}.csv", "text/csv")

# =============================================================================  TAB 2
with tabs[1]:
    st.subheader("Interactive process map — move any slider in the sidebar and the map updates")
    mc1, mc2, mc3 = st.columns(3)
    colour_by = mc1.selectbox("Colour map by", ["Hardness – Hybrid GPR", "Hardness – Analytical", "Hardness – Random Forest",
                                                "Hardness – Gradient Boosting", "Relative density (analytical)"])
    if proc == "LPBF":
        Pr = mc2.slider("Power range (W)", 50, 600, (80, 400), 10)
        vr = mc3.slider("Speed range (mm/s)", 100, 3500, (200, 2500), 50)
    else:
        Pr = mc2.slider("Power range (W)", 100, 4000, (200, 2000), 50)
        vr = mc3.slider("Speed range (mm/s)", 0.5, 50.0, (2.0, 30.0), 0.5)
    nP, nv = 45, 45
    Pg = np.linspace(*Pr, nP); vg = np.linspace(*vr, nv)
    PP, VV = np.meshgrid(Pg, vg, indexing="ij")
    grid = bundle.predict_all(mat_key, proc, PP.ravel(), VV.ravel(), hatch, layer, spot, feed, pre, ht, T_ht, t_ht, kh_thr, A_abs, ball_thr)
    gp = grid["physics"]
    key = {"Hardness – Hybrid GPR": "Hybrid GPR", "Hardness – Analytical": "Analytical",
           "Hardness – Random Forest": "Random Forest", "Hardness – Gradient Boosting": "Gradient Boosting"}.get(colour_by)
    Z = (grid[key] if key else gp["rel_density"]).reshape(nP, nv)
    fig = go.Figure()
    fig.add_heatmap(x=vg, y=Pg, z=Z, colorscale="Viridis" if key else "RdYlGn",
                    colorbar=dict(title="HV" if key else "% dense"), zsmooth="best",
                    hovertemplate="v=%{x:.1f} mm/s<br>P=%{y:.0f} W<br>value=%{z:.1f}<extra></extra>")
    for name, arr, col in [("Lack of fusion", gp["lof"], "#ff4d4d"), ("Keyhole", gp["keyhole"], "#ffffff"),
                           ("Balling", gp["balling"], "#ffb000")]:
        if proc == "LDED" and name == "Balling":
            continue
        fig.add_contour(x=vg, y=Pg, z=np.asarray(arr).reshape(nP, nv), contours=dict(start=1, end=1, size=1, coloring="none", showlabels=False),
                        line=dict(color=col, width=3, dash="dash"), showscale=False, name=f"{name} boundary", showlegend=True,
                        hoverinfo="skip")
    if proc == "LPBF":
        VEDg = PP / (VV * hatch * 1e-3 * layer * 1e-3)
        fig.add_contour(x=vg, y=Pg, z=VEDg, contours=dict(start=20, end=300, size=20, coloring="none", showlabels=True,
                                                          labelfont=dict(color="white", size=10)),
                        line=dict(color="rgba(255,255,255,.35)", width=1), showscale=False, name="VED (J/mm³)",
                        showlegend=True, hoverinfo="skip")
    ex = bundle.df[(bundle.df.material == mat_key) & (bundle.df.process == proc)]
    if len(ex):
        fig.add_scatter(x=ex.v_mms, y=ex.P_W, mode="markers", name="Published experiments",
                        marker=dict(symbol="circle-open", size=10, color="white", line=dict(width=2)),
                        text=[f"{r.id}: {r.HV:.0f} HV ({r.ht})" for r in ex.itertuples()], hovertemplate="%{text}<extra></extra>")
    fig.add_scatter(x=[v], y=[P], mode="markers", name="Your setting",
                    marker=dict(symbol="star", size=20, color="red", line=dict(color="white", width=2)))
    fig.update_layout(height=600, xaxis_title="Scan / traverse speed v (mm/s)", yaxis_title="Laser power P (W)",
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption("Dashed lines: red = lack-of-fusion boundary (Tang–Pistorius–Beuth / DED powder-melting balance), white = keyhole "
               "onset (King normalised enthalpy), amber = balling (Plateau–Rayleigh, L/W limit set under Advanced). Grey lines = iso-VED. "
               "The dense, conduction-mode window lies between the red and white lines.")

    if mat_key in ("18Ni300", "H13", "17-4PH"):
        st.subheader("Heat-treatment map (temperature × time)")
        Tg = np.linspace({"18Ni300": 420, "H13": 450, "17-4PH": 470}[mat_key], {"18Ni300": 620, "H13": 680, "17-4PH": 630}[mat_key], 50)
        tg = np.logspace(np.log10(0.25), np.log10(24), 50)
        TT, tt = np.meshgrid(Tg, tg, indexing="ij")
        ht_use = ht if ht in ("DA", "SA", "DT", "QT") else {"18Ni300": "DA", "H13": "DT", "17-4PH": "DA"}[mat_key]
        H_ab = f("HV_ab")
        Zh = ph.apply_ht(mat_key, proc, np.full(TT.shape, H_ab), ht_use, TT, tt, np.full(TT.shape, f("porosity")))
        fh = go.Figure(go.Contour(x=tg, y=Tg, z=Zh, colorscale="Turbo", contours=dict(showlabels=True),
                                  colorbar=dict(title="HV")))
        fh.add_scatter(x=[t_ht], y=[T_ht], mode="markers", marker=dict(symbol="star", size=18, color="white",
                                                                   line=dict(color="black", width=1.5)), name="Your HT")
        if mat_key == "18Ni300":
            Tp = np.linspace(Tg[0], Tg[-1], 60)
            fh.add_scatter(x=ph.maraging_peak_time(Tp), y=Tp, mode="lines", line=dict(color="white", dash="dot"),
                           name="Peak-aging time (Arrhenius fit)")
        fh.update_xaxes(type="log", title="Time (h)")
        fh.update_layout(height=450, yaxis_title="Temperature (°C)", title=ht_opts[ht_use], margin=dict(t=40))
        st.plotly_chart(fh, width="stretch")

# =============================================================================  TAB 3
with tabs[2]:
    st.subheader("What should I change?")
    for s in ph.suggestions(mat_key, proc, phy, P, v, hatch, layer, target, preds["Hybrid GPR"]):
        st.markdown(f"- {s}")
    st.divider()
    st.subheader("🎯 Inverse design — find settings that hit a target hardness")
    tgt = st.number_input("Target hardness (HV)", 100, 900, int(target or round(preds["Hybrid GPR"])), 5, key="inv_t")
    allow_ht = st.checkbox("Also search heat-treatment temperature/time", value=mat_key != "SS316L")
    if st.button("Search settings", type="primary"):
        if proc == "LPBF":
            Ps = np.linspace(80, 450, 30); vs = np.linspace(200, 2500, 30)
        else:
            Ps = np.linspace(200, 2500, 30); vs = np.linspace(2, 30, 30)
        PP2, VV2 = np.meshgrid(Ps, vs, indexing="ij")
        cands = []
        ht_list = [(ht, T_ht, t_ht)]
        if allow_ht and mat_key != "SS316L":
            code = {"18Ni300": "DA", "H13": "DT", "17-4PH": "DA"}[mat_key]
            ht_list = [("AB", 0, 0)] + [(code, T, t) for T in np.arange(*{"18Ni300": (440, 621, 30), "H13": (450, 681, 50),
                                                                       "17-4PH": (482, 640, 40)}[mat_key]) for t in (1, 3, 6, 10)]
        for (hcode, T, t) in ht_list:
            g = bundle.predict_all(mat_key, proc, PP2.ravel(), VV2.ravel(), hatch, layer, spot, feed, pre, hcode, T, t, kh_thr, A_abs, ball_thr)
            ok = (g["physics"]["lof"] < 0.9) & (g["physics"]["keyhole"] < 0.9)
            if proc == "LPBF":
                ok &= g["physics"]["balling"] < 1.0
            for i in np.where(ok)[0]:
                cands.append(dict(P_W=PP2.ravel()[i], v_mms=VV2.ravel()[i], HT=ph.HT_OPTIONS[mat_key][hcode],
                                  T_C=f"{T:.0f}" if hcode != "AB" else "–", t_h=f"{t:g}" if hcode != "AB" else "–",
                                  HV_hybrid=g["Hybrid GPR"][i], HV_analytical=g["Analytical"][i],
                                  uncertainty_95=1.96 * g["gpr_sd"][i], rel_density=g["physics"]["rel_density"][i],
                                  build_rate_mm3_s=(VV2.ravel()[i] * hatch * layer * 1e-6) if proc == "LPBF" else VV2.ravel()[i] * float(np.ravel(g["physics"]["W"])[i]) * 1e3 * layer * 1e-3))
        if not cands:
            st.error("No defect-free setting found in the search window — change hatch/layer/spot and retry.")
        else:
            cd = pd.DataFrame(cands)
            cd["|error| HV"] = (cd.HV_hybrid - tgt).abs()
            cd = cd.sort_values(["|error| HV", "uncertainty_95"]).head(10)
            if proc == "LPBF":
                cd = cd.sort_values(["|error| HV", "build_rate_mm3_s"], ascending=[True, False])
            st.dataframe(cd.round(1), hide_index=True, width="stretch")
            st.caption("Only conduction-mode, fully-dense, non-balling settings are kept. Ranked by closeness to target "
                       "(then by build rate for LPBF). Uses the Hybrid GPR model. Always confirm with a test coupon.")

# =============================================================================  TAB 4
with tabs[3]:
    st.subheader("Rosenthal temperature field around the moving laser (top surface)")
    th = ph.thermal(mat_key, proc, P, v, spot, pre, A_abs)
    p_ = ph.MATERIALS[mat_key]
    Lx = max(1.6 * f("L"), 2.5 * f("W"))
    xs = np.linspace(-Lx, 0.25 * Lx, 320); ys = np.linspace(-1.5 * f("W"), 1.5 * f("W"), 160)
    X, Y = np.meshgrid(xs, ys)
    Rr = np.sqrt(X ** 2 + Y ** 2 + (0.25 * spot * 1e-6) ** 2)
    al = ph.alpha(mat_key)
    T = th["T0"] + th["Q"] / (2 * np.pi * p_["k"] * Rr) * np.exp(-v * 1e-3 * (X + Rr) / (2 * al))
    T = np.minimum(T, p_["Tb"])
    Tc = np.clip(T - 273.15, pre, 1.5 * (p_["Tl"] - 273.15))
    ft = go.Figure(go.Contour(x=xs * 1e3, y=ys * 1e3, z=Tc, colorscale="Hot", zmin=pre, zmax=1.5 * (p_["Tl"] - 273.15),
                              ncontours=30, contours=dict(coloring="heatmap", showlines=False),
                              colorbar=dict(title="°C", x=1.02), hovertemplate="x=%{x:.2f} mm, y=%{y:.2f} mm<br>T=%{z:.0f} °C<extra></extra>"))
    ft.add_contour(x=xs * 1e3, y=ys * 1e3, z=T, contours=dict(start=p_["Tl"], end=p_["Tl"], size=1, coloring="none"),
                   line=dict(color="cyan", width=3), showscale=False, name="Liquidus (melt pool)")
    ft.add_contour(x=xs * 1e3, y=ys * 1e3, z=T, contours=dict(start=p_["Ts"], end=p_["Ts"], size=1, coloring="none"),
                   line=dict(color="white", width=1.5, dash="dot"), showscale=False, name="Solidus")
    ft.update_layout(height=380, xaxis_title="x behind laser (mm)", yaxis_title="y (mm)",
                     yaxis_scaleanchor="x", margin=dict(t=10),
                     legend=dict(orientation="h", y=1.12))
    st.plotly_chart(ft, width="stretch")
    st.caption("Point-source Rosenthal solution (thick plate, temperature capped at boiling). Cyan line = liquidus isotherm. "
               "It ignores latent heat, Marangoni flow and powder effects, so treat it as a teaching estimate.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Solidification map (G vs R)**")
        Rgr = np.logspace(-4, 0.5, 50)
        fg = go.Figure()
        for GRv, lab in [(1e5, "G/R = 1e5 (CET)"), (1e7, "G/R = 1e7")]:
            fg.add_scatter(x=Rgr, y=GRv * Rgr, mode="lines", line=dict(dash="dash"), name=lab)
        for crv in (1e3, 1e5, 1e7):
            fg.add_scatter(x=Rgr, y=crv / Rgr, mode="lines", line=dict(dash="dot", color="grey"), name=f"G·R = {crv:.0e} K/s")
        fg.add_scatter(x=[f("R")], y=[f("G")], mode="markers", marker=dict(symbol="star", size=18, color="red"), name="Your setting")
        fg.update_xaxes(type="log", title="R (m/s)"); fg.update_yaxes(type="log", title="G (K/m)", range=[4, 8.5])
        fg.update_layout(height=380, margin=dict(t=10), legend=dict(font=dict(size=10)))
        st.plotly_chart(fg, width="stretch")
    with c2:
        st.markdown("**Cell spacing vs cooling rate (λ = 80·Ṫ⁻⁰·³³ µm)**")
        crs = np.logspace(2, 7.5, 60)
        fl = go.Figure(go.Scatter(x=crs, y=ph.cell_spacing_um(crs), mode="lines", name="Katayama–Matsunawa"))
        fl.add_scatter(x=[f("cr")], y=[f("lam")], mode="markers", marker=dict(symbol="star", size=18, color="red"), name="Your setting")
        fl.update_xaxes(type="log", title="Cooling rate (K/s)"); fl.update_yaxes(type="log", title="λ (µm)")
        fl.update_layout(height=380, margin=dict(t=10))
        st.plotly_chart(fl, width="stretch")

# =============================================================================  TAB 5
with tabs[4]:
    st.subheader("How close is each model to the published experiments?")
    st.markdown("Leave-one-out cross-validation: each experimental record is hidden, every model (including the "
                "analytical calibration) is re-fitted without it, and then asked to predict it.")
    c1, c2 = st.columns(2)
    c1.markdown("**All records**")
    c1.dataframe(met_all.round(2), hide_index=True, width="stretch")
    c2.markdown(f"**{mat_key} + {proc} only**")
    if met_sub is not None:
        c2.dataframe(met_sub.round(2), hide_index=True, width="stretch")
    else:
        c2.info("Fewer than 2 records for this combination.")
    L = bundle.loo
    fp = go.Figure()
    lo, hi = L.HV.min() * 0.9, L.HV.max() * 1.05
    fp.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="black", dash="dash"), name="Perfect")
    fp.add_scatter(x=[lo, hi], y=[lo * 1.1, hi * 1.1], mode="lines", line=dict(color="grey", dash="dot"), name="±10 %")
    fp.add_scatter(x=[lo, hi], y=[lo * 0.9, hi * 0.9], mode="lines", line=dict(color="grey", dash="dot"), showlegend=False)
    for m in MODEL_NAMES:
        fp.add_scatter(x=L.HV, y=L[m], mode="markers", name=m, marker=dict(color=COLORS[m], size=8, opacity=.8),
                       text=L.id + " · " + L.material + " " + L.process + " " + L.ht, hovertemplate="%{text}<br>exp %{x:.0f} / pred %{y:.0f}<extra></extra>")
    fp.update_layout(height=520, xaxis_title="Experimental HV", yaxis_title="Predicted HV (LOO)", margin=dict(t=10))
    st.plotly_chart(fp, width="stretch")
    with st.expander("Per-record LOO predictions"):
        st.dataframe(L.round(1), hide_index=True, width="stretch")
    st.info("Reading the numbers honestly: 316L has 33 tabulated points, so its metrics are meaningful. The 18Ni-300, H13 and "
            "17-4PH records are few single-condition anchors, and the analytical constants for those alloys were set from the "
            "same literature, so their analytical errors are optimistic. Pure ML models (RF/GB) cannot extrapolate to an alloy "
            "they have barely seen — the physics-informed Hybrid GPR is the recommended model.")

# =============================================================================  TAB 6
with tabs[5]:
    st.subheader("Experimental database")
    st.dataframe(bundle.df[["id", "material", "process", "P_W", "v_mms", "hatch_um", "layer_um", "spot_um", "feed_gmin",
                            "preheat_C", "ht", "ht_T_C", "ht_t_h", "HV", "rel_density_pct", "quality", "source", "doi", "notes"]],
                 hide_index=True, width="stretch")
    st.download_button("⬇️ Download database (CSV)", bundle.df.to_csv(index=False).encode(), "rajam_dataset.csv", "text/csv")
    with st.expander("➕ Add your own experimental data and retrain"):
        st.markdown("Upload a CSV with the same columns as the database (at least `material, process, P_W, v_mms, HV`; "
                    "material ∈ SS316L / 18Ni300 / 17-4PH / H13, process ∈ LPBF / LDED, ht ∈ AB/SR/SOL/HIP/DA/SA/ST/DT/QT).")
        up = st.file_uploader("CSV file", type="csv")
        if up is not None and st.button("Retrain with my data"):
            st.session_state.extra_csv = up.getvalue().decode()
            st.rerun()
        if st.session_state.get("extra_csv") and st.button("Remove my data"):
            st.session_state.pop("extra_csv"); st.rerun()

    st.subheader("Model equations")
    st.latex(r"E_V=\frac{P}{v\,h\,t}\quad\Big[\tfrac{J}{mm^3}\Big],\qquad W=\sqrt{\frac{8AP}{e\,\pi\,\rho c_p v (T_l-T_0)}},\qquad D\approx W/2")
    st.latex(r"\dot T=\frac{2\pi k (T_s-T_0)^2\,v}{AP},\quad G=\frac{2\pi k (T_l-T_0)^2}{AP},\quad R=v,\quad \lambda=80\,\dot T^{-0.33}\ \mu m")
    st.latex(r"\frac{\Delta H}{h_s}=\frac{AP}{\pi\,\rho c_p T_l\sqrt{\alpha\, v\, \sigma^3}}\ (>30\Rightarrow\text{keyhole}),\qquad \Big(\frac{h}{W}\Big)^2+\Big(\frac{t}{D}\Big)^2\le 1\ (\text{no LoF})")
    st.latex(r"HV = (H_0 + k\,\lambda^{-1/2})\,e^{-b\,\phi},\qquad HV_{aged}=HV_0+\Delta H_{max}\big[1-e^{-(3t/t_p)^{0.9}}\big]-S(T)\,f_{over}(t)")
    st.latex(r"t_p = A\,e^{Q/RT}\ (Q\approx %.0f\ \mathrm{kJ/mol}),\qquad P_{HJ}=T\,(20+\log_{10}t)" % (ph.MAR_Q / 1000))
    st.markdown(f"Calibrated on the current data: H₀ = {bundle.cal['H0_316']:.1f} HV, k = {bundle.cal['kHP_316']:.1f} HV·µm^0.5, "
                f"b = {bundle.cal['b_por']:.2f} (per unit porosity fraction ×100). "
                "ML: Random Forest (300 trees), Gradient Boosting (300 stumps, depth 2), Gaussian Process (RBF + white noise) on the "
                "residual between experiment and analytical model, using cooling rate, LoF index, keyhole ratio, alloy, process and HT as inputs.")

    st.subheader("Sources of experimental data")
    st.markdown("""
1. Bakhtiarian M. et al. (2024) *The effects of SLM process parameters on the relative density and hardness of austenitic stainless steel 316L*, J. Mater. Res. Technol. 29:1616–1629. doi:10.1016/j.jmrt.2024.01.237 — 16 tabulated LPBF conditions.
2. Amar E. et al. (2023) *RSM approach for optimizing the processing parameters of 316L SS in DED*, Materials 16:7253. doi:10.3390/ma16237253 — 15 tabulated LENS conditions.
3. Ali S.N., Taha Z.A., Previtali B. (2026) *Laser DED of SS316L: processing parameters, porosity and microstructure*, Iraqi J. Laser 25(1) — reported hardness range.
4. Casati R. et al. (2016) *Aging behaviour and mechanical performance of 18-Ni 300 steel processed by SLM*, Metals 6:218. doi:10.3390/met6090218 — as-built / solution hardness and peak-aging times (used for the Arrhenius aging kinetics).
5. Hu J. et al. (2025) *Effect of heat treatments on 18Ni300 maraging steel produced by SLM*, Materials 18:2284. doi:10.3390/ma18102284.
6. Felicioni S. et al. (2024) *DED of 18NiM300 steel*, Sci. Technol. Adv. Mater. 25:2346071. doi:10.1080/14686996.2024.2346071 — DED as-built 34–37 HRC, direct-aged (480 °C/6 h) 50–51 HRC (used to scale the DED aging response).
7. Nagahama T. et al. (2019) *Porosity and mechanical properties of H13 tool steel processed by high-speed SLM*, SFF Symposium, pp. 677–683.
8. Kosarava K. et al. (2026) *ML-assisted optimisation of PBF-LB parameters of H13 on a 350 °C preheated platform*, Materials 19:210. doi:10.3390/ma19010210.
9. Sharma V.M. et al. (2024) *Process parameter optimization of DED QT17-4+ steel*, Adv. Mater. Technol. 9:2400024.
10. Greco S. et al. (2020) Int. J. Adv. Manuf. Technol. 108:1551 — 316L LPBF 232–275 HV0.01 (range, not used for training: different indentation load).
""")
    st.subheader("Physics references")
    st.markdown("""
Rosenthal D. (1946) Trans. ASME 68:849 · Eagar T.W., Tsai N.S. (1983) Weld. J. 62:346 · King W.E. et al. (2014) J. Mater. Process. Technol. 214:2915 (keyhole, normalised enthalpy) ·
Tang M., Pistorius P.C., Beuth J.L. (2017) Addit. Manuf. 14:39 (lack-of-fusion criterion) · Yadroitsev I. et al. (2010) J. Mater. Process. Technol. 210:1624 (track stability) ·
Katayama S., Matsunawa A. (1984) ICALEO (cell spacing vs cooling rate) · Hammar Ö., Svensson U. (1979) Creq/Nieq · Hollomon J.H., Jaffe L.D. (1945) Trans. AIME 162:223 ·
Pavlina E.J., Van Tyne C.J. (2008) J. Mater. Eng. Perform. 17:888 (hardness–strength) · ASTM E140 (hardness conversion).
""")
    st.subheader("Open datasets you can add (download, convert to the CSV format above, upload)")
    st.markdown("""
- **NIST AM-Bench** (LPBF IN625/316L/15-5PH benchmark measurements) — https://www.nist.gov/ambench
- **RAISE-LPBF** (LPBF process monitoring dataset) — https://astro.paperswithcode.com/dataset/raise-lpbf
- Petersen H.E. et al. (2023) M300 maraging LPBF parameter study, SFF Symposium (10 parameter sets, hardness in figures)
- Mendeley Data / Zenodo — search “LPBF 316L hardness”, “maraging SLM”, “DED stainless microhardness”
""")
    st.warning("Teaching & research tool. Predictions are estimates from simplified physics and a small literature database; "
               "machine-specific factors (powder, gas flow, scan strategy, part geometry) are not modelled. Verify with experiments.")
    with st.expander("Instructor: usage log"):
        if os.path.exists(LOG_FILE):
            lg = pd.read_csv(LOG_FILE)
            st.write(f"{len(lg)} logged predictions")
            st.download_button("Download usage log", lg.to_csv(index=False).encode(), "rajam_usage_log.csv", "text/csv")
        else:
            st.write("No predictions logged yet.")

c_now = _read_counter()
today_v = c_now.get("today_visits", 0) if c_now.get("today_date") == dt.date.today().isoformat() else 0
st.markdown(f"""
<div class="raj-counter">
  <div class="box"><div class="num">{c_now.get('visits', 0):06d}</div><div class="lab">👥 Total visitors</div></div>
  <div class="box"><div class="num">{today_v:04d}</div><div class="lab">📅 Visitors today</div></div>
  <div class="box"><div class="num">{c_now.get('predictions', 0):06d}</div><div class="lab">🔮 Predictions made</div></div>
</div>
<div class='raj-foot'>Counting since {c_now.get('since', dt.date.today().isoformat())} · Developed by <b>Dr. Gururaja S</b>,
Centre for Innovative Manufacturing Research (CIMR), VIT Vellore · RajAM v1.0 · part of the Raj3DLabs family</div>
""", unsafe_allow_html=True)
