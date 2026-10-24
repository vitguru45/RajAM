"""
RajAM - data handling, calibration of the analytical model and ML models.

ML models:
  * Random Forest (pure data-driven)
  * Gradient Boosting (pure data-driven)
  * Physics-informed GPR: Gaussian process learns the residual (experiment - analytical),
    so it falls back to physics where there is no data and returns an uncertainty band.
Closeness to experiments is measured with leave-one-out cross-validation (LOO-CV) on the
experimental records only.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

import physics as ph

MATS = list(ph.MATERIALS.keys())
PROCS = ["LPBF", "LDED"]
DEFAULT_SPOT = {"LPBF": 70.0, "LDED": 800.0}


def load_dataset(path="rajam_dataset.csv", extra=None):
    df = pd.read_csv(path)
    if extra is not None and len(extra):
        df = pd.concat([df, extra], ignore_index=True)
    df = df.dropna(subset=["material", "process", "P_W", "v_mms", "HV"]).reset_index(drop=True)
    df["spot_f"] = df.apply(lambda r: r.spot_um if pd.notna(r.spot_um) else DEFAULT_SPOT[r.process], axis=1)
    df["feed_f"] = df["feed_gmin"].fillna(10.0)
    df["pre_f"] = df["preheat_C"].fillna(25.0)
    df["hatch_f"] = df.apply(lambda r: r.hatch_um if pd.notna(r.hatch_um) else 0.6 * r.spot_f, axis=1)
    df["ht"] = df["ht"].fillna("AB")
    df["T_f"] = df["ht_T_C"].fillna(0.0)
    df["t_f"] = df["ht_t_h"].fillna(0.0)
    lay = []
    for _, r in df.iterrows():
        if pd.notna(r.layer_um):
            lay.append(r.layer_um)
        else:  # DED without reported layer: use mass-balance track height
            d = ph.defects(r.material, r.process, r.P_W, r.v_mms, r.hatch_f, np.nan, r.spot_f, r.feed_f, r.pre_f)
            lay.append(float(np.ravel(d.get("h_dep", 0.5e-3))[0]) * 1e6)
    df["layer_f"] = lay
    return df


def analytical_for_rows(df, cal=None, keyhole_thr=30.0):
    out = []
    for _, r in df.iterrows():
        d = ph.predict(r.material, r.process, r.P_W, r.v_mms, r.hatch_f, r.layer_f, r.spot_f, r.feed_f,
                       r.pre_f, r.ht, r.T_f if r.T_f else 490, r.t_f if r.t_f else 1, keyhole_thr, cal)
        out.append({k: float(np.ravel(d[k])[0]) for k in ("HV", "cr", "lof", "keyhole", "lam", "rel_density", "GR")})
    return pd.DataFrame(out, index=df.index)


def calibrate(df):
    """Fit H0, kHP (316L Hall-Petch-type) and porosity knock-down b on as-built 316L records."""
    sub = df[(df.material == "SS316L") & (df.ht == "AB")]
    if len(sub) < 4:
        return dict(ph.CAL)

    def res(x):
        cal = {"H0_316": x[0], "kHP_316": x[1], "b_por": x[2]}
        return analytical_for_rows(sub, cal)["HV"].values - sub.HV.values

    sol = least_squares(res, x0=[140, 70, 6], bounds=([50, 0, 0], [250, 200, 30]))
    return {"H0_316": sol.x[0], "kHP_316": sol.x[1], "b_por": sol.x[2]}


# ----------------------------------------------------------------------------- features
def raw_features(df):
    X = pd.DataFrame(index=df.index)
    for m in MATS:
        X["m_" + m] = (df.material == m).astype(float)
    X["is_DED"] = (df.process == "LDED").astype(float)
    X["P"] = df.P_W
    X["log_v"] = np.log10(df.v_mms)
    X["hatch"] = df.hatch_f
    X["layer"] = df.layer_f
    X["spot"] = df.spot_f
    X["feed"] = df.feed_f
    X["pre"] = df.pre_f
    X["log_LED"] = np.log10(df.P_W / df.v_mms)
    X["aged"] = df.ht.isin(["DA", "SA", "DT", "QT"]).astype(float)
    X["solution"] = df.ht.isin(["SA", "ST", "SOL", "HIP", "QT"]).astype(float)
    X["ht_T"] = df.T_f
    X["ht_logt"] = np.log10(1 + df.t_f)
    return X


def hybrid_features(df, ana):
    X = pd.DataFrame(index=df.index)
    for m in MATS:
        X["m_" + m] = (df.material == m).astype(float)
    X["is_DED"] = (df.process == "LDED").astype(float)
    X["log_cr"] = np.log10(ana.cr)
    X["lof"] = np.clip(ana.lof, 0, 5)
    X["kh"] = ana.keyhole
    X["aged"] = df.ht.isin(["DA", "SA", "DT", "QT"]).astype(float)
    X["solution"] = df.ht.isin(["SA", "ST", "SOL", "HIP", "QT"]).astype(float)
    return X


class HybridGPR:
    def __init__(self):
        k = C(30.0 ** 2, (1.0, 1e5)) * RBF(length_scale=1.5, length_scale_bounds=(0.3, 50)) + WhiteKernel(60.0, (1, 2e3))
        self.gp = GaussianProcessRegressor(kernel=k, normalize_y=False, n_restarts_optimizer=2, random_state=0)
        self.sc = StandardScaler()

    def fit(self, X, resid):
        self.sc.fit(X)
        self.gp.fit(self.sc.transform(X), resid)
        return self

    def predict(self, X):
        mu, sd = self.gp.predict(self.sc.transform(X), return_std=True)
        return mu, sd


def make_rf():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=1, random_state=0)


def make_gbr():
    return GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.9, random_state=0)


class ModelBundle:
    """Everything the app needs: calibration, trained models, LOO metrics."""

    def __init__(self, df):
        self.df = df
        self.cal = calibrate(df)
        ph.CAL.update(self.cal)
        self.ana = analytical_for_rows(df, self.cal)
        self.Xr = raw_features(df)
        self.Xh = hybrid_features(df, self.ana)
        y = df.HV.values
        self.rf = make_rf().fit(self.Xr, y)
        self.gbr = make_gbr().fit(self.Xr, y)
        self.gpr = HybridGPR().fit(self.Xh, y - self.ana.HV.values)
        self.loo = self._loo()

    def _loo(self):
        df, y = self.df, self.df.HV.values
        n = len(df)
        pa, prf, pg, ph_ = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        for i in range(n):
            tr = np.ones(n, bool); tr[i] = False
            cal = calibrate(df[tr])
            ana_all = analytical_for_rows(df, cal)
            pa[i] = ana_all.HV.values[i]
            Xr_tr, Xr_te = self.Xr[tr], self.Xr[~tr]
            prf[i] = make_rf().fit(Xr_tr, y[tr]).predict(Xr_te)[0]
            pg[i] = make_gbr().fit(Xr_tr, y[tr]).predict(Xr_te)[0]
            Xh = hybrid_features(df, ana_all)
            g = HybridGPR().fit(Xh[tr], y[tr] - ana_all.HV.values[tr])
            ph_[i] = ana_all.HV.values[i] + g.predict(Xh[~tr])[0][0]
        ens = (pa + prf + pg + ph_) / 4.0
        out = df[["id", "material", "process", "ht", "HV", "source"]].copy()
        out["Analytical"] = pa; out["Random Forest"] = prf; out["Gradient Boosting"] = pg
        out["Hybrid GPR"] = ph_; out["Ensemble"] = ens
        ph.CAL.update(self.cal)
        return out

    # ------------------------------------------------------------------ inference
    def query_frame(self, m, proc, P, v, hatch, layer, spot, feed, pre, ht, T, t):
        P, v = np.broadcast_arrays(np.atleast_1d(np.asarray(P, float)), np.atleast_1d(np.asarray(v, float)))
        P, v = P.ravel(), v.ravel()
        q = pd.DataFrame(dict(material=m, process=proc, P_W=P, v_mms=v, hatch_f=hatch, layer_f=layer,
                              spot_f=spot, feed_f=feed, pre_f=pre, ht=ht,
                              T_f=T if ht != "AB" else 0.0, t_f=t if ht != "AB" else 0.0))
        return q

    def predict_all(self, m, proc, P, v, hatch, layer, spot, feed, pre, ht, T, t, keyhole_thr=30.0, A=None, ball_thr=5.0):
        q = self.query_frame(m, proc, P, v, hatch, layer, spot, feed, pre, ht, T, t)
        d = ph.predict(m, proc, q.P_W.values, q.v_mms.values, hatch, layer, spot, feed, pre, ht, T, t, keyhole_thr, self.cal, A, ball_thr)
        ana = pd.DataFrame(dict(HV=d["HV"], cr=d["cr"], lof=d["lof"], keyhole=d["keyhole"]))
        Xr = raw_features(q)
        Xh = hybrid_features(q, ana)
        rf = self.rf.predict(Xr)
        gb = self.gbr.predict(Xr)
        mu, sd = self.gpr.predict(Xh)
        hyb = d["HV"] + mu
        return dict(Analytical=d["HV"], **{"Random Forest": rf, "Gradient Boosting": gb, "Hybrid GPR": hyb},
                    Ensemble=(d["HV"] + rf + gb + hyb) / 4.0, gpr_sd=sd, physics=d)

    def metrics(self, subset=None):
        L = self.loo if subset is None else self.loo[subset]
        rows = []
        for c in ["Analytical", "Random Forest", "Gradient Boosting", "Hybrid GPR", "Ensemble"]:
            e = L[c] - L.HV
            ss = ((L.HV - L.HV.mean()) ** 2).sum()
            r2 = 1 - (e ** 2).sum() / ss if ss > 0 and len(L) > 2 else np.nan
            rows.append(dict(Model=c, MAE_HV=e.abs().mean(), RMSE_HV=np.sqrt((e ** 2).mean()),
                             MAPE_pct=(e.abs() / L.HV).mean() * 100, R2=r2, N=len(L)))
        return pd.DataFrame(rows)

    def domain(self, m, proc, ht):
        same = self.df[(self.df.material == m) & (self.df.process == proc)]
        same_ht = same[same.ht == ht]
        return len(same), len(same_ht)
