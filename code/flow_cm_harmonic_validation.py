#!/usr/bin/env python3
"""
FLOW-CM harmonic validation script
=================================

Clean validation code for the publication version of FLOW-CM.

It:
1. Downloads SPARC/Rotmod automatically if files are not present.
2. Reads Rotmod galaxy files.
3. Computes three models:
   - Baryonic Newtonian
   - RAR/MOND benchmark
   - FLOW-CM harmonic
4. Computes per-galaxy metrics.
5. Performs a geometric permutation test on Q_harm.
6. Exports publication-ready CSV tables.
7. Exports point-by-point predictions for plotting.

No galaxy-by-galaxy parameters are fitted.

Output folder:
    ./flow_cm_validation_results/

Author: Eliezer Siqueira
License: MIT
"""

from __future__ import annotations

import math
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("./data/sparc")
OUT_DIR = Path("./flow_cm_validation_results")

NO_DOWNLOAD = False
MAX_GALAXIES = 0          # 0 = all galaxies
SAVE_POINT_PREDICTIONS = True
N_PERMUTATIONS = 16
RANDOM_SEED = 12345

# Baryonic mass-to-light ratios
UPSILON_D = 0.45
UPSILON_B = 0.60

# FLOW-CM global constants
G0 = 1.4e-10
S_PUBLIC = 2.0 / 5.0
BETA_PUBLIC = 1.0 / 2.0
ELL_FRAC_NODE = 0.04
ELL_FRAC_GRAPH = 0.08

# Units
ACCEL_MS2_PER_KMS2_PER_KPC = 1.0e6 / 3.0856775814913673e19
SPARC_ROTMOD_URL = "https://astroweb.case.edu/SPARC/Rotmod_LTG.zip"
EPS = 1e-40


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Galaxy:
    name: str
    r: np.ndarray
    vobs: np.ndarray
    evobs: np.ndarray
    vgas: np.ndarray
    vdisk: np.ndarray
    vbul: np.ndarray
    sbdisk: Optional[np.ndarray] = None
    sbbul: Optional[np.ndarray] = None


# ============================================================
# NUMERIC HELPERS
# ============================================================

def safe(x, floor: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), np.maximum(x, floor), floor)


def signed_square(v: np.ndarray) -> np.ndarray:
    return np.sign(v) * v * v


def accel_from_v2(v2_kms2: np.ndarray, r_kpc: np.ndarray) -> np.ndarray:
    return safe(v2_kms2) / np.maximum(r_kpc, 1e-12) * ACCEL_MS2_PER_KMS2_PER_KPC


def v_from_g(g_ms2: np.ndarray, r_kpc: np.ndarray) -> np.ndarray:
    v2 = safe(g_ms2) * r_kpc / ACCEL_MS2_PER_KMS2_PER_KPC
    return np.sqrt(safe(v2))


def robust_norm(x: np.ndarray, percentile: float = 90.0, floor: float = 1e-30) -> np.ndarray:
    """
    Robust deterministic normalization used in the manuscript:

        N[X] = clip(X / P90(X>0), 0, 5)

    Some final variables are additionally clipped to [0,1].
    """
    x = safe(x)
    vals = x[np.isfinite(x) & (x > 0)]
    if len(vals) == 0:
        return np.zeros_like(x)
    scale = max(float(np.nanpercentile(vals, percentile)), floor)
    return np.clip(x / scale, 0.0, 5.0)


def minmax01(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    if not np.any(np.isfinite(x)):
        return np.zeros_like(x)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if hi - lo < floor:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def grad_log(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = safe(y, EPS)
    x = safe(x, 1e-12)
    if len(x) < 3:
        return np.zeros_like(x)
    return np.gradient(np.log(y), np.log(x), edge_order=1)


def forward_memory(r: np.ndarray, q: np.ndarray) -> np.ndarray:
    if len(r) < 2:
        return safe(q)
    dr = safe(np.gradient(r), np.nanmedian(np.gradient(r)))
    return safe(np.cumsum(safe(q) * dr) / np.maximum(np.cumsum(dr), EPS))


def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = safe(a)
    b = safe(b)
    return 2.0 * a * b / np.maximum(a + b, EPS)


def geom_blend(a: np.ndarray, b: np.ndarray, eta: np.ndarray) -> np.ndarray:
    a = np.clip(safe(a), 1e-12, None)
    b = np.clip(safe(b), 1e-12, None)
    eta = np.clip(safe(eta), 0.0, 1.0)
    return np.exp((1.0 - eta) * np.log(a) + eta * np.log(b))


def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(mask):
        return float("nan")
    return float(np.mean((y[mask] - yhat[mask]) ** 2))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y[mask] - yhat[mask])))


def interp_by_x(source_x: np.ndarray, source_q: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    sx = np.asarray(source_x, dtype=float)
    sq = np.asarray(source_q, dtype=float)
    tx = np.asarray(target_x, dtype=float)

    order = np.argsort(sx)
    sx = sx[order]
    sq = sq[order]

    uniq, idx = np.unique(sx, return_index=True)
    return np.interp(tx, uniq, sq[idx], left=sq[idx][0], right=sq[idx][-1])


# ============================================================
# SPARC / ROTMOD DATA
# ============================================================

def maybe_download_sparc(data_dir: Path, allow_download: bool = True) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    if list(data_dir.rglob("*.dat")):
        return

    if not allow_download:
        raise FileNotFoundError(
            f"No .dat files found in {data_dir}. "
            "Set NO_DOWNLOAD=False or place SPARC Rotmod files manually."
        )

    zip_path = data_dir / "Rotmod_LTG.zip"

    if not zip_path.exists():
        print(f"Downloading SPARC Rotmod data from: {SPARC_ROTMOD_URL}")
        print(f"Destination: {zip_path}")
        urllib.request.urlretrieve(SPARC_ROTMOD_URL, zip_path)

    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)


def parse_rotmod(path: Path) -> Optional[Galaxy]:
    rows: List[List[float]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue

            vals = []
            for part in re.split(r"\s+", line):
                try:
                    vals.append(float(part))
                except ValueError:
                    pass

            if len(vals) >= 6:
                rows.append(vals)

    if not rows:
        return None

    ncols = max(len(row) for row in rows)
    arr = np.full((len(rows), ncols), np.nan)

    for i, row in enumerate(rows):
        arr[i, :len(row)] = row

    r = arr[:, 0]
    vobs = arr[:, 1]
    evobs = arr[:, 2]
    vgas = arr[:, 3]
    vdisk = arr[:, 4]
    vbul = arr[:, 5]
    sbdisk = arr[:, 6] if ncols > 6 else None
    sbbul = arr[:, 7] if ncols > 7 else None

    mask = np.isfinite(r) & np.isfinite(vobs) & (r > 0) & (vobs >= 0)
    if int(np.sum(mask)) < 3:
        return None

    ev = evobs[mask]
    good = np.isfinite(ev) & (ev > 0)
    fallback = np.nanmedian(ev[good]) if np.any(good) else 1.0
    ev = np.where(good, ev, fallback)

    return Galaxy(
        name=path.stem.replace("_rotmod", ""),
        r=r[mask],
        vobs=vobs[mask],
        evobs=ev,
        vgas=vgas[mask],
        vdisk=vdisk[mask],
        vbul=vbul[mask],
        sbdisk=sbdisk[mask] if sbdisk is not None else None,
        sbbul=sbbul[mask] if sbbul is not None else None,
    )


def load_galaxies() -> List[Galaxy]:
    maybe_download_sparc(DATA_DIR, allow_download=not NO_DOWNLOAD)

    files = sorted(DATA_DIR.rglob("*_rotmod.dat")) or sorted(DATA_DIR.rglob("*.dat"))
    galaxies: List[Galaxy] = []

    for path in files:
        gal = parse_rotmod(path)
        if gal is not None:
            galaxies.append(gal)

    if MAX_GALAXIES and MAX_GALAXIES > 0:
        galaxies = galaxies[:MAX_GALAXIES]

    if not galaxies:
        raise RuntimeError(f"No galaxies could be loaded from {DATA_DIR}")

    return galaxies


# ============================================================
# MODEL PHYSICS
# ============================================================

def baryonic_acceleration(gal: Galaxy) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v2bar = safe(
        signed_square(gal.vgas)
        + UPSILON_D * gal.vdisk ** 2
        + UPSILON_B * gal.vbul ** 2
    )
    gbar = accel_from_v2(v2bar, gal.r)
    vbar = v_from_g(gbar, gal.r)
    return v2bar, gbar, vbar


def rar_acceleration(gbar: np.ndarray) -> np.ndarray:
    gb = safe(gbar)
    x = np.sqrt(gb / max(G0, EPS))
    denom = 1.0 - np.exp(-x)
    out = np.zeros_like(gb)
    mask = denom > 1e-12
    out[mask] = gb[mask] / denom[mask]
    return out


def lowacc_extra(gbar: np.ndarray, k_eff: np.ndarray, s: float = S_PUBLIC) -> np.ndarray:
    gb = safe(gbar)
    k = np.clip(safe(k_eff), 0.0, 2.0)
    return np.sqrt(gb * G0) * k * np.power(1.0 / (1.0 + gb / max(G0, EPS)), s)


def node_occupation_fields(r: np.ndarray, gbar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds F_raw and F_occ using the nodal kernel scale ell = 0.04 Rmax.
    """
    if len(r) < 2 or not np.any(gbar > 0):
        z = np.zeros_like(r)
        return z, z

    rmax = max(float(np.nanmax(r)), 1e-12)
    ell = max(ELL_FRAC_NODE * rmax, 1e-6)
    dr = safe(np.gradient(r), np.nanmedian(np.gradient(r)))

    q = robust_norm(gbar * dr, percentile=95)
    dist = np.abs(r[None, :] - r[:, None])
    D = q[:, None] * np.exp(-dist / ell)
    sqrtD = np.sqrt(safe(D))

    raw = 0.5 * ((np.sum(sqrtD, axis=0) ** 2) - np.sum(D, axis=0))
    raw_norm = robust_norm(raw, percentile=95)
    f_occ = raw / (1.0 + raw)

    return raw_norm, safe(f_occ)


def effective_number(q: np.ndarray) -> float:
    q = safe(q)
    s = float(np.sum(q))
    d = float(np.sum(q * q))
    if s <= 0 or d <= 0:
        return 1.0
    return max(s * s / d, 1.0)


def cumulative_collectivity(r: np.ndarray, gbar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    C_ext(r) = 1 - 1 / N_eff(<r)
    with q_i ~ gbar_i r_i dr_i.
    """
    dr = safe(np.gradient(r), np.nanmedian(np.gradient(r))) if len(r) > 1 else np.ones_like(r)
    q = safe(gbar * r * dr)

    n_eff = np.ones_like(r)
    c_ext = np.zeros_like(r)

    for i in range(len(r)):
        ni = effective_number(q[: i + 1])
        n_eff[i] = ni
        c_ext[i] = np.clip(1.0 - 1.0 / ni, 0.0, 1.0)

    return n_eff, c_ext


def density_occupation_proxy(gal: Galaxy, r: np.ndarray, gbar: np.ndarray) -> np.ndarray:
    """
    Optional stabilizing proxy used in the CM construction.
    It is computed deterministically from available surface-brightness columns when present.
    """
    rmax = max(float(np.nanmax(r)), 1e-12)
    rho_dyn = safe(gbar) / np.maximum(r / rmax, 1e-3)

    if gal.sbdisk is not None:
        sb = safe(gal.sbdisk)
        if gal.sbbul is not None:
            sb = sb + safe(gal.sbbul)
        gas = safe(gal.vgas ** 2 / np.maximum(r, 1e-12))
        rho = robust_norm(sb, 90) + robust_norm(gas, 90) + robust_norm(rho_dyn, 90)
    else:
        rho = rho_dyn

    vals = rho[np.isfinite(rho) & (rho > 0)]
    if len(vals) == 0:
        return np.zeros_like(rho)

    ref = max(float(np.nanmedian(vals)), EPS)
    return minmax01(np.log1p(rho / ref))


def build_graph_current(r: np.ndarray, gbar: np.ndarray, k_cm: np.ndarray) -> np.ndarray:
    """
    J_raw(i) = sum_j W_ij |K_i - K_j|
    W_ij = exp(-|r_i-r_j|/ell_g) sqrt(xi_i xi_j)
    ell_g = 0.08 Rmax
    """
    n = len(r)
    if n < 3:
        return np.zeros_like(r)

    rmax = max(float(np.nanmax(r)), 1e-12)
    ell = max(ELL_FRAC_GRAPH * rmax, 1e-6)

    dr = safe(np.gradient(r), np.nanmedian(np.gradient(r)))
    xi = robust_norm(gbar * dr, 90) + 0.5 * robust_norm(k_cm, 90)
    xi = np.clip(xi, 1e-6, None)

    dist = np.abs(r[:, None] - r[None, :])
    W = np.exp(-dist / ell) * np.sqrt(xi[:, None] * xi[None, :])
    np.fill_diagonal(W, 0.0)

    current = np.sum(W * np.abs(k_cm[:, None] - k_cm[None, :]), axis=1)
    return robust_norm(current, 90)


def compute_flow_fields(gal: Galaxy) -> Dict[str, np.ndarray | float]:
    r = gal.r
    _, gbar, vbar = baryonic_acceleration(gal)

    _, f_occ = node_occupation_fields(r, gbar)
    m_in = forward_memory(r, f_occ)
    n_eff, c_ext = cumulative_collectivity(r, gbar)

    # CM2 stabilizing density/memory blend used in the final run.
    m_proxy = np.maximum.accumulate(safe(gbar) * safe(r) ** 2)
    mu = np.clip(m_proxy / max(float(np.nanmax(m_proxy)), EPS), 0.0, 1.0)
    m_log = np.clip(np.log1p(mu) / np.log(2.0), 0.0, 1.0)
    o_rho = density_occupation_proxy(gal, r, gbar)
    d_rho_m = np.sqrt(np.clip(o_rho * m_log, 0.0, None))

    eta_cm = np.clip((1.0 - c_ext) * d_rho_m, 0.0, 1.0)
    f_cm = geom_blend(f_occ, o_rho, eta_cm)
    m_cm = geom_blend(m_in, d_rho_m, eta_cm)

    x_cm = c_ext + f_cm + m_cm
    k_cm = np.clip(1.0 - np.exp(-safe(x_cm)), 0.0, 2.0)

    # Decline gate
    slope_gbar = grad_log(gbar, r)
    decline = np.clip(-slope_gbar, 0.0, None)
    q_decl = decline / (1.0 + decline)

    # Radial transition Q_K
    dK = np.abs(grad_log(k_cm + 1e-6, r))
    q_k = np.clip(robust_norm(dK, 90) * q_decl, 0.0, 1.0)

    # Graph current J_graph
    j_raw = build_graph_current(r, gbar, k_cm)
    j_graph = np.clip(j_raw * q_decl, 0.0, 1.0)

    # Harmonic local conductance and inward memory
    q_loc = harmonic_mean(q_k, j_graph)
    q_harm = forward_memory(r, q_loc)

    # Effective coupling
    k_eff = np.clip(k_cm + (1.0 - k_cm) * np.tanh(2.0 * q_harm), 0.0, 2.0)

    # Compactness gate
    cext_med = float(np.nanmedian(c_ext))
    focc_med = float(np.nanmedian(f_occ))
    min_med = float(np.nanmedian(m_in))
    lambda_cext = float(np.clip(cext_med * math.sqrt(max(focc_med * min_med, 0.0)), 0.0, 1.0))

    y = gbar / max(G0, EPS)
    w_y = y / (1.0 + y)
    g_comp = np.clip(1.0 - BETA_PUBLIC * (1.0 - lambda_cext) * w_y, 0.0, 1.0)

    return {
        "r": r,
        "x": np.clip(r / max(float(np.nanmax(r)), 1e-12), 0.0, 1.0),
        "gbar": gbar,
        "vbar": vbar,
        "f_occ": f_occ,
        "m_in": m_in,
        "n_eff": n_eff,
        "c_ext": c_ext,
        "k_cm": k_cm,
        "q_decl": q_decl,
        "q_k": q_k,
        "j_graph": j_graph,
        "q_loc": q_loc,
        "q_harm": q_harm,
        "k_eff": k_eff,
        "g_comp": g_comp,
        "lambda_cext": lambda_cext,
    }


def predict_models(gal: Galaxy, fields: Dict[str, np.ndarray | float]) -> Dict[str, np.ndarray]:
    r = gal.r
    gbar = np.asarray(fields["gbar"])
    k_eff = np.asarray(fields["k_eff"])
    g_comp = np.asarray(fields["g_comp"])

    v_newton = v_from_g(gbar, r)
    v_rar = v_from_g(rar_acceleration(gbar), r)

    g_flow = lowacc_extra(gbar, k_eff, S_PUBLIC) * g_comp
    v_flow = v_from_g(gbar + g_flow, r)

    return {
        "newton_baryonic": v_newton,
        "rar_mond_benchmark": v_rar,
        "FLOW_tanh_2_harmonic": v_flow,
    }


def predict_flow_with_qharm(gal: Galaxy, fields: Dict[str, np.ndarray | float], q_harm: np.ndarray) -> np.ndarray:
    gbar = np.asarray(fields["gbar"])
    k_cm = np.asarray(fields["k_cm"])
    g_comp = np.asarray(fields["g_comp"])

    k_eff = np.clip(k_cm + (1.0 - k_cm) * np.tanh(2.0 * safe(q_harm)), 0.0, 2.0)
    g_flow = lowacc_extra(gbar, k_eff, S_PUBLIC) * g_comp
    return v_from_g(gbar + g_flow, gal.r)


# ============================================================
# VALIDATION / TABLES
# ============================================================

def evaluate_all(galaxies: List[Galaxy]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
    detail_rows = []
    point_rows = []
    field_store: Dict[str, Dict] = {}

    for idx, gal in enumerate(galaxies, start=1):
        if idx == 1 or idx % 10 == 0 or idx == len(galaxies):
            print(f"Evaluating {idx}/{len(galaxies)}: {gal.name}")

        fields = compute_flow_fields(gal)
        preds = predict_models(gal, fields)
        field_store[gal.name] = fields

        for model, vpred in preds.items():
            m = mse(gal.vobs, vpred)
            detail_rows.append({
                "galaxy": gal.name,
                "model": model,
                "mse": m,
                "rmse": math.sqrt(max(m, 0.0)),
                "mae": mae(gal.vobs, vpred),
                "n_points": len(gal.r),
                "r_max_kpc": float(np.nanmax(gal.r)),
                "vobs_median": float(np.nanmedian(gal.vobs)),
                "qharm_median": float(np.nanmedian(fields["q_harm"])),
                "jgraph_median": float(np.nanmedian(fields["j_graph"])),
                "qk_median": float(np.nanmedian(fields["q_k"])),
                "kcm_median": float(np.nanmedian(fields["k_cm"])),
                "lambda_cext": float(fields["lambda_cext"]),
            })

            if SAVE_POINT_PREDICTIONS:
                for i in range(len(gal.r)):
                    point_rows.append({
                        "galaxy": gal.name,
                        "model": model,
                        "r_kpc": float(gal.r[i]),
                        "vobs": float(gal.vobs[i]),
                        "evobs": float(gal.evobs[i]),
                        "vpred": float(vpred[i]),
                        "residual": float(gal.vobs[i] - vpred[i]),
                    })

    return pd.DataFrame(detail_rows), pd.DataFrame(point_rows), field_store


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    pivot = detail.pivot(index="galaxy", columns="model", values="mse")
    best = pivot.idxmin(axis=1)

    rows = []
    for model in detail["model"].drop_duplicates():
        d = detail[detail["model"] == model]
        row = {
            "model": model,
            "mean_galaxy_mse": float(d["mse"].mean()),
            "median_galaxy_mse": float(d["mse"].median()),
            "mean_rmse": float(d["rmse"].mean()),
            "median_rmse": float(d["rmse"].median()),
            "mean_mae": float(d["mae"].mean()),
            "wins_best_count": int((best == model).sum()),
            "wins_best_frac": float((best == model).mean()),
            "n_galaxies": int(d["galaxy"].nunique()),
        }

        for ref in ["newton_baryonic", "rar_mond_benchmark", "FLOW_tanh_2_harmonic"]:
            if ref in pivot.columns and model != ref:
                ratio = (pivot[model] / pivot[ref]).replace([np.inf, -np.inf], np.nan)
                row[f"wins_vs_{ref}_frac"] = float((pivot[model] < pivot[ref]).mean())
                row[f"ratio_vs_{ref}_mean"] = float(ratio.mean())
                row[f"ratio_vs_{ref}_median"] = float(ratio.median())
                row[f"mean_delta_mse_vs_{ref}"] = float((pivot[ref] - pivot[model]).mean())

        rows.append(row)

    return pd.DataFrame(rows).sort_values("mean_galaxy_mse")


def build_publication_tables(summary: pd.DataFrame, detail: pd.DataFrame, perm_model: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def get_row(model: str) -> pd.Series:
        d = summary[summary["model"] == model]
        if d.empty:
            raise ValueError(f"Missing model in summary: {model}")
        return d.iloc[0]

    flow = get_row("FLOW_tanh_2_harmonic")
    rar = get_row("rar_mond_benchmark")
    newton = get_row("newton_baryonic")

    permuted = np.nan
    if perm_model is not None and len(perm_model):
        d = perm_model[perm_model["model"].astype(str).str.contains("harm", case=False, na=False)]
        if not d.empty and "mean_permuted_mse" in d.columns:
            permuted = float(d.iloc[0]["mean_permuted_mse"])

    final = pd.DataFrame([
        {
            "model_label": "Baryonic Newtonian",
            "model_key": "newton_baryonic",
            "mean_mse": float(newton["mean_galaxy_mse"]),
            "median_mse": float(newton["median_galaxy_mse"]),
            "mean_rmse": float(newton["mean_rmse"]),
            "mean_mae": float(newton["mean_mae"]),
            "permuted_mean_mse": np.nan,
            "comment": "Baryonic mass model only",
        },
        {
            "model_label": "RAR/MOND benchmark",
            "model_key": "rar_mond_benchmark",
            "mean_mse": float(rar["mean_galaxy_mse"]),
            "median_mse": float(rar["median_galaxy_mse"]),
            "mean_rmse": float(rar["mean_rmse"]),
            "mean_mae": float(rar["mean_mae"]),
            "permuted_mean_mse": np.nan,
            "comment": "External acceleration-relation benchmark",
        },
        {
            "model_label": "Harmonic FLOW-CM",
            "model_key": "FLOW_tanh_2_harmonic",
            "mean_mse": float(flow["mean_galaxy_mse"]),
            "median_mse": float(flow["median_galaxy_mse"]),
            "mean_rmse": float(flow["mean_rmse"]),
            "mean_mae": float(flow["mean_mae"]),
            "permuted_mean_mse": permuted,
            "comment": "Final nodal-conductance law",
        },
    ])

    pivot = detail.pivot(index="galaxy", columns="model", values="mse")
    per_gal = pd.DataFrame({
        "galaxy": pivot.index,
        "Newton_baryonic": pivot["newton_baryonic"].values,
        "RAR_MOND": pivot["rar_mond_benchmark"].values,
        "FLOW_harmonic": pivot["FLOW_tanh_2_harmonic"].values,
    })
    per_gal["ratio_FLOW_over_RAR"] = per_gal["FLOW_harmonic"] / per_gal["RAR_MOND"].replace(0, np.nan)
    per_gal["delta_RAR_minus_FLOW"] = per_gal["RAR_MOND"] - per_gal["FLOW_harmonic"]
    per_gal["ratio_FLOW_over_Newton"] = per_gal["FLOW_harmonic"] / per_gal["Newton_baryonic"].replace(0, np.nan)

    return final, per_gal


# ============================================================
# PERMUTATION TEST
# ============================================================

def permutation_test(galaxies: List[Galaxy], field_store: Dict[str, Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    names = [g.name for g in galaxies]
    gal_by_name = {g.name: g for g in galaxies}

    rows = []
    detail_rows = []

    for pidx in range(N_PERMUTATIONS):
        perm = names.copy()
        rng.shuffle(perm)

        per_gal = []
        for target_name, source_name in zip(names, perm):
            gal = gal_by_name[target_name]
            f = field_store[target_name]
            src = field_store[source_name]

            target_q = np.asarray(f["q_harm"])
            source_q = np.asarray(src["q_harm"])
            q_perm = interp_by_x(np.asarray(src["x"]), source_q, np.asarray(f["x"]))

            v_true = predict_flow_with_qharm(gal, f, target_q)
            v_perm = predict_flow_with_qharm(gal, f, q_perm)

            m_true = mse(gal.vobs, v_true)
            m_perm = mse(gal.vobs, v_perm)

            row = {
                "permutation": pidx,
                "model": "perm_FLOW_harmonic_Qharm",
                "galaxy": target_name,
                "source_Qharm_galaxy": source_name,
                "mse_true": m_true,
                "mse_permuted": m_perm,
                "delta_permuted_minus_true": m_perm - m_true,
            }
            detail_rows.append(row)
            per_gal.append(row)

        df = pd.DataFrame(per_gal)
        rows.append({
            "permutation": pidx,
            "model": "perm_FLOW_harmonic_Qharm",
            "mean_true_mse": float(df["mse_true"].mean()),
            "mean_permuted_mse": float(df["mse_permuted"].mean()),
            "delta_permuted_minus_true": float(df["mse_permuted"].mean() - df["mse_true"].mean()),
            "wins_true_vs_permuted_frac": float((df["mse_true"] < df["mse_permuted"]).mean()),
        })

    perm_summary = pd.DataFrame(rows)
    perm_detail = pd.DataFrame(detail_rows)

    by_model = pd.DataFrame([{
        "model": "perm_FLOW_harmonic_Qharm",
        "n_perm": int(len(perm_summary)),
        "mean_true_mse": float(perm_summary["mean_true_mse"].mean()),
        "mean_permuted_mse": float(perm_summary["mean_permuted_mse"].mean()),
        "mean_delta_permuted_minus_true": float(perm_summary["delta_permuted_minus_true"].mean()),
        "mean_wins_true_vs_permuted_frac": float(perm_summary["wins_true_vs_permuted_frac"].mean()),
    }])

    return by_model, perm_detail


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SPARC/Rotmod galaxies...")
    galaxies = load_galaxies()
    print(f"Loaded galaxies: {len(galaxies)}")
    print(f"Output directory: {OUT_DIR.resolve()}")

    print("\nEvaluating models...")
    detail, points, field_store = evaluate_all(galaxies)

    print("\nBuilding summary...")
    summary = build_summary(detail)

    print("\nRunning geometric permutation test...")
    perm_model, perm_detail = permutation_test(galaxies, field_store)

    print("\nBuilding publication tables...")
    final_results, per_galaxy = build_publication_tables(summary, detail, perm_model)

    # Save outputs
    summary.to_csv(OUT_DIR / "summary_models.csv", index=False)
    detail.to_csv(OUT_DIR / "detail_by_galaxy_model.csv", index=False)
    perm_model.to_csv(OUT_DIR / "permutation_by_model.csv", index=False)
    perm_detail.to_csv(OUT_DIR / "permutation_detail_by_galaxy.csv", index=False)
    final_results.to_csv(OUT_DIR / "final_results_publication.csv", index=False)
    per_galaxy.to_csv(OUT_DIR / "per_galaxy_publication_metrics.csv", index=False)

    if SAVE_POINT_PREDICTIONS:
        points.to_csv(OUT_DIR / "point_predictions.csv", index=False)

    print("\n=== SUMMARY MODELS ===")
    print(summary.to_string(index=False))

    print("\n=== PERMUTATION BY MODEL ===")
    print(perm_model.to_string(index=False))

    print("\n=== FINAL RESULTS PUBLICATION ===")
    print(final_results.to_string(index=False))

    print("\nFiles written:")
    for filename in [
        "summary_models.csv",
        "detail_by_galaxy_model.csv",
        "permutation_by_model.csv",
        "permutation_detail_by_galaxy.csv",
        "final_results_publication.csv",
        "per_galaxy_publication_metrics.csv",
        "point_predictions.csv",
    ]:
        path = OUT_DIR / filename
        if path.exists():
            print(" -", path.resolve())


if __name__ == "__main__":
    main()
