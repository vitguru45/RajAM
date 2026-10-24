"""
RajAM - analytical (physics-based) models for laser AM of steels.

All functions are numpy-vectorised so the same code drives single predictions
and the real-time process maps.

Models used (see README / app "Models & data" tab for references):
  * Energy densities: VED = P/(v h t) [J/mm^3], LED = P/v [J/mm], areal E = P/(v d) [J/mm^2]
  * Rosenthal moving point source (thick plate) -> melt-pool width (Eagar-Tsai high-Peclet form),
    length, cooling rate at solidus, thermal gradient G and solidification rate R
  * Normalised enthalpy dH/hs (King et al. 2014) -> keyhole criterion
  * Lack-of-fusion criterion (Tang, Pistorius & Beuth 2017): (h/W)^2 + (t/D)^2 <= 1
  * Plateau-Rayleigh balling criterion (Yadroitsev et al. 2010): L/W above ~pi..5 -> unstable track (threshold adjustable)
  * DED powder-melting energy balance + mass-balance deposit height
  * Cell spacing  lambda = a * (dT/dt)^-n   (Katayama & Matsunawa 1984, a=80 um, n=0.33)
  * Hardness: Hall-Petch-type  HV = H0 + k*lambda^-0.5, porosity knock-down exp(-b*p)
  * Maraging aging: JMAK precipitation + Arrhenius kinetics fitted to Casati et al. 2016 peak times
  * H13 tempering: Hollomon-Jaffe parameter
  * 17-4PH: condition-based (H900...H1150) aging response
  * HV -> HRC (ASTM E140 approx.), HV -> UTS/YS (Pavlina & Van Tyne 2008)
"""
import numpy as np

R_GAS = 8.314

# ----------------------------------------------------------------------------- materials
MATERIALS = {
    "SS316L": dict(label="SS316L (austenitic stainless)", rho=7900, cp=500, k=20.0, Tl=1723, Ts=1648,
                   Lf=2.7e5, Tb=3090, H0=140.0, kHP=70.0, martensitic=False),
    "18Ni300": dict(label="18Ni-300 maraging steel", rho=8000, cp=450, k=22.0, Tl=1740, Ts=1686,
                    Lf=2.7e5, Tb=3100, H0=325.0, kHP=10.0, martensitic=True),
    "17-4PH": dict(label="17-4PH precipitation-hardening stainless", rho=7780, cp=460, k=18.0, Tl=1713,
                   Ts=1677, Lf=2.7e5, Tb=3100, H0=360.0, kHP=10.0, martensitic=True),
    "H13": dict(label="H13 hot-work tool steel", rho=7760, cp=460, k=25.0, Tl=1727, Ts=1588,
                Lf=2.6e5, Tb=3100, H0=585.0, kHP=10.0, martensitic=True),
}
ABSORPTIVITY = {"LPBF": 0.45, "LDED": 0.35}

# hardness model constants that are calibrated to data (316L) - defaults, overwritten by calibrate()
CAL = {"H0_316": 140.0, "kHP_316": 70.0, "b_por": 6.0}


def mat(m):
    return MATERIALS[m]


def alpha(m):
    p = mat(m)
    return p["k"] / (p["rho"] * p["cp"])


# ----------------------------------------------------------------------------- thermal
def thermal(m, process, P, v_mms, spot_um, preheat_C=25.0, A=None):
    """Rosenthal-based thermal quantities. Returns dict of arrays (SI unless stated)."""
    p = mat(m)
    A = ABSORPTIVITY[process] if A is None else A
    P = np.asarray(P, float)
    v = np.asarray(v_mms, float) * 1e-3                      # m/s
    sig = np.asarray(spot_um, float) * 0.5e-6                # beam radius, m
    T0 = np.asarray(preheat_C, float) + 273.15
    Q = A * P
    dT = np.maximum(p["Ts"] - T0, 50.0)
    dTm = np.maximum(p["Tl"] - T0, 50.0)
    # melt pool width (Eagar-Tsai / Rosenthal high Peclet), conduction-mode depth = W/2
    W = np.sqrt(8.0 * Q / (np.e * np.pi * p["rho"] * p["cp"] * v * dTm))
    # melt pool length behind the source (centre-line Rosenthal, high Peclet)
    L = Q / (2.0 * np.pi * p["k"] * dTm)
    # normalised enthalpy (King et al. 2014)
    hs = p["rho"] * p["cp"] * p["Tl"]
    dHhs = Q / (np.pi * hs * np.sqrt(alpha(m) * v * sig ** 3))
    # depth: conduction W/2, deepened in transition/keyhole regime
    D = 0.5 * W * np.where(dHhs > 20.0, np.sqrt(dHhs / 20.0), 1.0)
    # cooling rate at solidus, gradient, growth rate (centre line)
    cr = 2.0 * np.pi * p["k"] * dT ** 2 * v / Q             # K/s
    G = 2.0 * np.pi * p["k"] * dTm ** 2 / Q                  # K/m
    Rg = v                                                   # m/s
    return dict(Q=Q, W=W, L=L, D=D, dHhs=dHhs, cr=cr, G=G, R=Rg, GR=G / Rg, T0=T0)


def cell_spacing_um(cr):
    """Katayama & Matsunawa (1984) lambda = 80 * cr^-0.33  (um, cr in K/s)."""
    return 80.0 * np.power(np.maximum(cr, 1.0), -0.33)


# ----------------------------------------------------------------------------- defects / density
def defects(m, process, P, v_mms, hatch_um, layer_um, spot_um, feed_gmin=10.0, preheat_C=25.0,
            keyhole_thr=30.0, capture=0.5, A=None, ball_thr=5.0):
    th = thermal(m, process, P, v_mms, spot_um, preheat_C, A)
    p = mat(m)
    W, D, L = th["W"], th["D"], th["L"]
    h = np.asarray(hatch_um, float) * 1e-6
    t = np.asarray(layer_um, float) * 1e-6
    out = dict(th)
    if process == "LPBF":
        lof = (h / W) ** 2 + (t / D) ** 2
        melt_ratio = np.ones_like(lof)
    else:
        # DED: powder melting energy balance + mass-balance deposit height
        mdot = np.asarray(feed_gmin, float) / 60.0 * 1e-3 * capture          # kg/s caught
        need = mdot * (p["cp"] * (p["Tl"] - th["T0"]) + p["Lf"])             # W
        melt_ratio = th["Q"] / np.maximum(need, 1e-9)
        Wd = np.maximum(W, np.asarray(spot_um, float) * 1e-6 * 0.8)
        v = np.asarray(v_mms, float) * 1e-3
        h_dep = mdot / (p["rho"] * v * Wd)                                  # m, single-track height
        lof = 1.0 / np.maximum(melt_ratio, 1e-3)                            # >1: powder not fully melted
        out.update(h_dep=h_dep, W=Wd)
        W = Wd
    kh = th["dHhs"] / keyhole_thr
    ball = L / W / ball_thr      # Plateau-Rayleigh type: track breaks up when L/W exceeds ~pi..5 (Yadroitsev 2010)
    p_lof = 4.0 * np.clip(lof - 0.8, 0, None)            # % porosity
    p_kh = 3.0 * np.clip(kh - 1.0, 0, None)
    p_base = 0.3
    por = np.clip(p_base + p_lof + p_kh, 0, 15)
    out.update(lof=lof, keyhole=kh, balling=ball, porosity=por, rel_density=100 - por,
               melt_ratio=melt_ratio)
    return out


def energy_densities(P, v_mms, hatch_um, layer_um, spot_um):
    P = np.asarray(P, float); v = np.asarray(v_mms, float)
    ved = P / (v * np.asarray(hatch_um, float) * 1e-3 * np.asarray(layer_um, float) * 1e-3)
    led = P / v
    aed = P / (v * np.asarray(spot_um, float) * 1e-3)
    return ved, led, aed


# ----------------------------------------------------------------------------- heat treatment
# Maraging aging kinetics: peak-aging times from Casati et al. (2016): 600C-10min, 540C-1h, 490C-4h, 460C-8h
_TP = np.array([[873.15, 1 / 6], [813.15, 1.0], [763.15, 4.0], [733.15, 8.0]])
_fit = np.polyfit(1.0 / _TP[:, 0], np.log(_TP[:, 1]), 1)
MAR_Q = _fit[0] * R_GAS                         # J/mol
MAR_lnA = _fit[1]


def maraging_peak_time(T_C):
    return np.exp(MAR_lnA + _fit[0] / (np.asarray(T_C, float) + 273.15))


def maraging_aged(H_start, T_C, t_h, process):
    T_C = np.asarray(T_C, float); t_h = np.asarray(t_h, float)
    tp = maraging_peak_time(T_C)
    dHmax = (265.0 - 1.3 * np.clip(T_C - 490, 0, None) - 1.0 * np.clip(470 - T_C, 0, None))
    dHmax = dHmax * (0.7 if process == "LDED" else 1.0)
    prec = 1.0 - np.exp(-(3.0 * t_h / tp) ** 0.9)
    S = 2.2 * np.clip(T_C - 480, 0, None)      # over-aging / reverted austenite softening
    over = 1.0 - np.exp(-np.clip(t_h - tp, 0, None) / (4.0 * tp))
    return H_start + dHmax * prec - S * over


def hollomon_jaffe(T_C, t_h, C=20.0):
    return (np.asarray(T_C, float) + 273.15) * (C + np.log10(np.maximum(np.asarray(t_h, float), 0.01)))


_H13_HJ = np.array([14000, 15692, 16708, 17722, 18737, 19500])
_H13_HV = np.array([610, 622, 605, 500, 390, 300])


def h13_tempered(H_ab, T_C, t_h):
    hj = hollomon_jaffe(T_C, t_h)
    return np.interp(hj, _H13_HJ, _H13_HV) + 0.5 * (H_ab - 600.0)


_PH_T = np.array([482, 496, 552, 579, 593, 621])     # H900, H925, H1025, H1075, H1100, H1150
_PH_HV = np.array([434, 420, 372, 356, 345, 327])


def ph174_aged(H_ab, T_C, t_h, direct):
    base = np.interp(T_C, _PH_T, _PH_HV) + 6.0 * np.log10(np.maximum(t_h, 0.25))
    return base + (0.3 * (H_ab - 360.0) if direct else 0.0)


HT_OPTIONS = {
    "SS316L": {"AB": "As-built", "SR": "Stress relief (650 °C / 2 h)", "SOL": "Solution anneal (1050 °C / 1 h)",
               "HIP": "HIP (1150 °C / 100 MPa / 3 h)"},
    "18Ni300": {"AB": "As-built", "DA": "Direct aging (choose T, t)", "SA": "Solution 820 °C/1 h + aging (choose T, t)",
                "ST": "Solution only (820 °C / 1 h)"},
    "17-4PH": {"AB": "As-built", "DA": "Direct aging (H900…H1150: choose T, t)",
               "SA": "Solution A 1040 °C + aging (choose T, t)"},
    "H13": {"AB": "As-built", "DT": "Direct temper (choose T, t)", "QT": "Quench 1020 °C + temper (choose T, t)"},
}
HT_DEFAULT_T = {"18Ni300": (490, 6.0), "17-4PH": (482, 1.0), "H13": (550, 2.0), "SS316L": (650, 2.0)}


def apply_ht(m, process, H_ab, ht, T_C, t_h, porosity):
    """Return hardness after heat treatment (array-safe)."""
    pf = np.exp(-CAL["b_por"] * porosity / 100.0)
    H_dense = H_ab / pf
    if ht == "AB" or ht is None:
        return H_ab
    if m == "SS316L":
        if ht == "SR":
            return 0.96 * H_ab
        if ht == "SOL":
            return 170.0 * pf
        if ht == "HIP":
            return 165.0 * np.ones_like(H_ab)
    if m == "18Ni300":
        if ht == "ST":
            return 285.0 * pf
        if ht == "DA":
            return maraging_aged(H_dense, T_C, t_h, process) * pf
        if ht == "SA":
            return maraging_aged(285.0, T_C, t_h, process) * pf
    if m == "17-4PH":
        if ht == "DA":
            return ph174_aged(H_dense, T_C, t_h, True) * pf
        if ht == "SA":
            return ph174_aged(H_dense, T_C, t_h, False) * pf
    if m == "H13":
        if ht == "DT":
            return h13_tempered(H_dense, T_C, t_h) * pf
        if ht == "QT":
            return h13_tempered(620.0, T_C, t_h) * pf
    return H_ab


# ----------------------------------------------------------------------------- hardness
def asbuilt_hardness(m, lam_um, porosity, cal=None):
    cal = CAL if cal is None else cal
    p = mat(m)
    if m == "SS316L":
        H = cal["H0_316"] + cal["kHP_316"] * np.power(lam_um, -0.5)
    else:
        H = p["H0"] + p["kHP"] * np.power(lam_um, -0.5)
    return H * np.exp(-cal["b_por"] * porosity / 100.0)


def predict(m, process, P, v_mms, hatch_um, layer_um, spot_um, feed_gmin=10.0, preheat_C=25.0,
            ht="AB", T_C=490.0, t_h=6.0, keyhole_thr=30.0, cal=None, A=None, ball_thr=5.0):
    """Full analytical prediction chain. Returns dict of arrays."""
    d = defects(m, process, P, v_mms, hatch_um, layer_um, spot_um, feed_gmin, preheat_C, keyhole_thr, A=A, ball_thr=ball_thr)
    lam = cell_spacing_um(d["cr"])
    H_ab = asbuilt_hardness(m, lam, d["porosity"], cal)
    H = apply_ht(m, process, H_ab, ht, T_C, t_h, d["porosity"])
    if m == "SS316L" and ht == "HIP":
        d["rel_density"] = np.full_like(d["rel_density"], 99.9)
    d.update(lam=lam, HV_ab=H_ab, HV=H)
    return d


# ----------------------------------------------------------------------------- conversions
_E140_HRC = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
_E140_HV = np.array([238, 266, 302, 345, 392, 446, 513, 595, 697, 832])


def hv_to_hrc(hv):
    hv = np.asarray(hv, float)
    return np.where(hv < 238, np.nan, np.interp(hv, _E140_HV, _E140_HRC))


def hrc_to_hv(hrc):
    return np.interp(hrc, _E140_HRC, _E140_HV)


def hv_to_uts(hv):
    return -99.8 + 3.734 * np.asarray(hv, float)


def hv_to_ys(hv):
    return -90.7 + 2.876 * np.asarray(hv, float)


# ----------------------------------------------------------------------------- microstructure
def creq_nieq(c):
    creq = c["Cr"] + 1.37 * c["Mo"] + 1.5 * c["Si"] + 2 * c["Nb"] + 3 * c["Ti"]
    nieq = c["Ni"] + 0.31 * c["Mn"] + 22 * c["C"] + 14.2 * c["N"] + c["Cu"]
    return creq, nieq


def morphology(GR, cr):
    if GR < 1e5:
        grain = "equiaxed tendency (columnar-to-equiaxed transition)"
    elif GR < 1e7:
        grain = "columnar grains, epitaxial growth along build direction"
    else:
        grain = "strongly columnar / textured grains (<100> along build direction)"
    if cr > 1e5:
        sub = "fine cellular sub-structure"
    elif cr > 1e3:
        sub = "cellular-dendritic sub-structure"
    else:
        sub = "dendritic sub-structure with inter-dendritic segregation"
    return grain, sub


def microstructure_text(m, process, ht, d, T_C=None, t_h=None, ratio_ae=None):
    cr = float(np.ravel(d["cr"])[0]); GR = float(np.ravel(d["GR"])[0]); lam = float(np.ravel(d["lam"])[0])
    grain, sub = morphology(GR, cr)
    lines = [f"**Solidification:** {grain}; {sub} (cell/dendrite spacing ≈ {lam:.2f} µm)."]
    if m == "SS316L":
        phase = "Austenite (FCC γ) with Cr/Mo-enriched cell walls and high dislocation density"
        if ratio_ae is not None:
            if ratio_ae > 1.48 and cr < 1e5:
                phase += f"; Creq/Nieq = {ratio_ae:.2f} → FA solidification, expect a few % inter-dendritic δ-ferrite"
            elif ratio_ae > 1.48:
                phase += (f"; Creq/Nieq = {ratio_ae:.2f} (FA range) but the high cooling rate suppresses δ-ferrite"
                          " → essentially fully austenitic")
            else:
                phase += f"; Creq/Nieq = {ratio_ae:.2f} → A/AF mode, fully austenitic"
        if ht == "SR":
            phase += ". Stress relief keeps the cells and relaxes residual stress."
        elif ht in ("SOL", "HIP"):
            phase += ". After solution/HIP, cell walls dissolve, recrystallised γ grains with annealing twins."
    elif m == "18Ni300":
        phase = "Lath martensite (α′) with a few % retained austenite at cell boundaries"
        if process == "LDED":
            phase += "; lower layers partly in-situ aged by re-heating"
        if ht in ("DA", "SA"):
            tp = float(maraging_peak_time(T_C))
            state = "under-aged" if t_h < 0.7 * tp else ("peak-aged" if t_h < 2.0 * tp else "over-aged")
            phase += (f". After aging at {T_C:.0f} °C / {t_h:.1f} h ({state}; peak time ≈ {tp:.1f} h): "
                      "nano-scale Ni₃(Ti,Mo) / Fe₂Mo precipitates")
            if state == "over-aged" or T_C > 520:
                phase += " + reverted austenite and precipitate coarsening"
        elif ht == "ST":
            phase += ". After solution: homogenised, precipitate-free martensite (soft, machinable)."
    elif m == "17-4PH":
        phase = "Martensite with retained austenite (more with N₂-atomised powder) and some δ-ferrite"
        if ht in ("DA", "SA"):
            phase += f". After aging at {T_C:.0f} °C: Cu-rich ε precipitates; above ~550 °C some reverted austenite."
    elif m == "H13":
        phase = "Fresh martensite + retained austenite (~10–20 %) with inter-dendritic carbides; in-situ tempered lower layers"
        if ht in ("DT", "QT"):
            phase += (f". Tempering at {T_C:.0f} °C / {t_h:.1f} h (Hollomon-Jaffe P = {float(hollomon_jaffe(T_C, t_h)):.0f}): "
                      "tempered martensite + secondary (V,Mo)C carbides; above ~600 °C coarsening and softening")
    lines.append(f"**Phases:** {phase}")
    return lines


# ----------------------------------------------------------------------------- suggestions
def suggestions(m, process, d, P, v, hatch, layer, target=None, HV=None):
    s = []
    lof = float(np.ravel(d["lof"])[0]); kh = float(np.ravel(d["keyhole"])[0]); ball = float(np.ravel(d["balling"])[0])
    if lof > 1.0:
        if process == "LPBF":
            s.append(f"⚠️ **Lack of fusion risk** (LoF index {lof:.2f} > 1). Increase power (≈ {P*1.15:.0f} W), "
                     f"lower speed (≈ {v*0.85:.0f} mm/s), or reduce hatch to ≤ {0.7*float(np.ravel(d['W'])[0])*1e6:.0f} µm / layer to ≤ {0.8*float(np.ravel(d['D'])[0])*1e6:.0f} µm.")
        else:
            s.append(f"⚠️ **Incomplete fusion / poor bonding** (index {lof:.2f}). Raise laser power or lower powder feed / traverse speed so the melt pool re-melts the previous layer.")
    if kh > 1.0:
        s.append(f"⚠️ **Keyhole regime** (ΔH/hs ratio {kh:.2f} > 1). Reduce power (≈ {P*0.85:.0f} W) or increase speed (≈ {v*1.3:.0f} mm/s), or defocus the beam.")
    elif kh > 0.8:
        s.append("ℹ️ Close to the conduction→keyhole transition; small power increases may create keyhole pores.")
    if ball > 1.0 and process == "LPBF":
        s.append(f"⚠️ **Balling / humping risk** (L/W ratio {ball:.2f} × limit). Lower scan speed or increase power to widen the track.")
    if process == "LDED":
        mr = float(np.ravel(d["melt_ratio"])[0])
        if mr < 1.2:
            s.append(f"⚠️ Laser energy is only {mr:.2f}× what the captured powder needs to melt — raise power or cut powder feed.")
    if not s:
        s.append("✅ Parameters sit inside the **conduction-mode, fully dense** window of the analytical process map.")
    if target is not None and HV is not None:
        diff = target - HV
        if abs(diff) < 10:
            s.append(f"🎯 Predicted hardness is within ±10 HV of your target ({target:.0f} HV).")
        elif diff > 0:
            tip = {"SS316L": "faster cooling (lower P/v, smaller spot) refines cells — but 316L tops out near 250–280 HV; consider HIP-free cold work or a harder alloy",
                   "18Ni300": "apply aging at 480–500 °C for 3–6 h (peak hardness ≈ 52–55 HRC)",
                   "17-4PH": "use H900 aging (482 °C / 1 h)",
                   "H13": "temper at ≤ 550 °C (secondary-hardening peak) instead of higher temperatures"}[m]
            s.append(f"⬆️ Need **+{diff:.0f} HV**: {tip}.")
        else:
            tip = {"SS316L": "solution anneal (1050 °C) or HIP brings it to ~165–175 HV",
                   "18Ni300": "use solution-only, or over-age at 540–600 °C",
                   "17-4PH": "age at higher temperature (H1025–H1150)",
                   "H13": "temper at 600–650 °C"}[m]
            s.append(f"⬇️ Need **{diff:.0f} HV**: {tip}.")
    return s
