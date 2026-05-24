#!/usr/bin/env python3
"""
Generate publication assets for the FLOW-CM paper
=================================================

This script creates the essential files needed for the publication draft:

1) figures/model_pipeline.pdf
2) figures/permutation_test.pdf
3) figures/mse_ratio_histogram.pdf
4) figures/model_performance_table.pdf
5) figures/rotation_curve_examples.pdf
6) tables/final_results_publication.csv
7) tables/per_galaxy_publication_metrics.csv
8) tables/figure_caption_snippets.tex
9) cover_letter_FLOW_CM.tex
10) README_publication_assets.md

It assumes you already ran the final FLOW-CM harmonic validation script and have
an output folder containing:

- summary_models.csv
- detail_by_galaxy_model.csv
- permutation_by_model.csv
- point_predictions.csv, optional but recommended for rotation curves

Default validation output folder expected:

    ./flow_cm_validation_results/

If point_predictions.csv is not available, the script still generates all
non-curve figures.

Author: Eliezer Siqueira
License: MIT
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

# Folder produced by code/flow_cm_harmonic_validation.py
RESULTS_DIR = Path("./flow_cm_validation_results")

# Output folder for publication assets
PUB_DIR = Path("./publication_assets_FLOW_CM")
FIG_DIR = PUB_DIR / "figures"
TABLE_DIR = PUB_DIR / "tables"

# Official model names used by the validation script
MODEL_FLOW = "FLOW_tanh_2_harmonic"
MODEL_FLOW_ALT = "FLOW_harmonic_base"   # fallback if using an older script
MODEL_RAR = "rar_mond_benchmark"
MODEL_NEWTON = "newton_baryonic"

# Optional model to include in example plots if present
MODEL_BASE = "flow_compact_public_base"

# Number of galaxies in the rotation-curve example figure
N_EXAMPLE_GALAXIES = 9

# Random seed only for selecting representative examples if needed
RANDOM_SEED = 12345

# If True, select galaxies where FLOW improves most over RAR.
# If False, select a balanced range of cases.
PREFER_FLOW_WIN_EXAMPLES = False

# Optional frozen values.
# Leave as None to read from summary_models.csv.
FORCED_VALUES = {
    "FLOW_mean_mse": None,
    "FLOW_median_mse": None,
    "FLOW_mean_rmse": None,
    "RAR_mean_mse": None,
    "RAR_median_mse": None,
    "RAR_mean_rmse": None,
    "NEWTON_mean_mse": None,
    "NEWTON_median_mse": None,
    "NEWTON_mean_rmse": None,
    "PERMUTED_FLOW_mean_mse": None,
}


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def read_csv_optional(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def pick_existing_model(summary: pd.DataFrame, primary: str, fallback: Optional[str] = None) -> str:
    models = set(summary["model"].astype(str))
    if primary in models:
        return primary
    if fallback and fallback in models:
        return fallback
    matches = [m for m in models if "harm" in m.lower() and "flow" in m.lower()]
    if matches:
        return sorted(matches)[0]
    raise ValueError(f"Could not find model {primary!r} or fallback {fallback!r} in summary_models.csv")


def row_for_model(summary: pd.DataFrame, model: str) -> pd.Series:
    d = summary[summary["model"] == model]
    if d.empty:
        raise ValueError(f"Model not found in summary: {model}")
    return d.iloc[0]


def get_metric(row: pd.Series, name: str, forced_key: Optional[str] = None) -> float:
    if forced_key and FORCED_VALUES.get(forced_key) is not None:
        return float(FORCED_VALUES[forced_key])
    if name in row and pd.notna(row[name]):
        return float(row[name])
    return float("nan")


def save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


# ============================================================
# FINAL TABLES
# ============================================================

def build_publication_tables(
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    perm_by_model: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    flow_model = pick_existing_model(summary, MODEL_FLOW, MODEL_FLOW_ALT)

    flow = row_for_model(summary, flow_model)
    rar = row_for_model(summary, MODEL_RAR)
    newton = row_for_model(summary, MODEL_NEWTON)

    permuted_mse = np.nan
    if perm_by_model is not None and len(perm_by_model):
        possible = perm_by_model.copy()

        if "q_source" in possible.columns:
            possible = possible[
                possible["model"].astype(str).str.contains("harm", case=False, na=False)
                | possible["q_source"].astype(str).str.contains("harm", case=False, na=False)
            ]
        else:
            possible = possible[
                possible["model"].astype(str).str.contains("harm", case=False, na=False)
            ]

        if not possible.empty and "mean_permuted_mse" in possible.columns:
            permuted_mse = float(possible.iloc[0]["mean_permuted_mse"])

    if FORCED_VALUES.get("PERMUTED_FLOW_mean_mse") is not None:
        permuted_mse = float(FORCED_VALUES["PERMUTED_FLOW_mean_mse"])

    final = pd.DataFrame([
        {
            "model_label": "Baryonic Newtonian",
            "model_key": MODEL_NEWTON,
            "mean_mse": get_metric(newton, "mean_galaxy_mse", "NEWTON_mean_mse"),
            "median_mse": get_metric(newton, "median_galaxy_mse", "NEWTON_median_mse"),
            "mean_rmse": get_metric(newton, "mean_rmse", "NEWTON_mean_rmse"),
            "mean_mae": get_metric(newton, "mean_mae"),
            "comment": "Baryons only",
        },
        {
            "model_label": "RAR/MOND benchmark",
            "model_key": MODEL_RAR,
            "mean_mse": get_metric(rar, "mean_galaxy_mse", "RAR_mean_mse"),
            "median_mse": get_metric(rar, "median_galaxy_mse", "RAR_median_mse"),
            "mean_rmse": get_metric(rar, "mean_rmse", "RAR_mean_rmse"),
            "mean_mae": get_metric(rar, "mean_mae"),
            "comment": "External acceleration-relation benchmark",
        },
        {
            "model_label": "Harmonic FLOW-CM",
            "model_key": flow_model,
            "mean_mse": get_metric(flow, "mean_galaxy_mse", "FLOW_mean_mse"),
            "median_mse": get_metric(flow, "median_galaxy_mse", "FLOW_median_mse"),
            "mean_rmse": get_metric(flow, "mean_rmse", "FLOW_mean_rmse"),
            "mean_mae": get_metric(flow, "mean_mae"),
            "comment": "Final nodal-conductance law",
        },
    ])

    final["permuted_mean_mse"] = np.nan
    final.loc[final["model_key"] == flow_model, "permuted_mean_mse"] = permuted_mse

    pivot = detail.pivot(index="galaxy", columns="model", values="mse")
    needed = [m for m in [flow_model, MODEL_RAR, MODEL_NEWTON] if m in pivot.columns]
    per_gal = pivot[needed].copy()

    per_gal = per_gal.rename(columns={
        flow_model: "FLOW_harmonic",
        MODEL_RAR: "RAR_MOND",
        MODEL_NEWTON: "Newton_baryonic",
    })

    if "FLOW_harmonic" in per_gal.columns and "RAR_MOND" in per_gal.columns:
        per_gal["ratio_FLOW_over_RAR"] = safe_ratio(per_gal["FLOW_harmonic"], per_gal["RAR_MOND"])
        per_gal["delta_RAR_minus_FLOW"] = per_gal["RAR_MOND"] - per_gal["FLOW_harmonic"]

    if "FLOW_harmonic" in per_gal.columns and "Newton_baryonic" in per_gal.columns:
        per_gal["ratio_FLOW_over_Newton"] = safe_ratio(per_gal["FLOW_harmonic"], per_gal["Newton_baryonic"])

    per_gal = per_gal.reset_index()

    final.to_csv(TABLE_DIR / "final_results_publication.csv", index=False)
    per_gal.to_csv(TABLE_DIR / "per_galaxy_publication_metrics.csv", index=False)

    return final, per_gal, flow_model


# ============================================================
# FIGURE 1: MODEL PIPELINE
# ============================================================

def plot_model_pipeline() -> None:
    labels = [
        "Baryonic rotation data\n$V_{gas}, V_{disk}, V_{bulge}$",
        "Baryonic acceleration\n$g_{bar}(r)$",
        "Nodal occupation\n$F_{occ}$",
        "Collectivity + memory\n$C_{ext}, M_{in}$",
        "Basal connectivity\n$K_{CM}$",
        "Radial transition\n$Q_K$",
        "Graph current\n$J_{graph}$",
        "Harmonic conductance\n$Q_{harm}$",
        "Effective coupling\n$K_{eff}$",
        "Rotation curve\n$v_{FLOW}(r)$",
    ]

    positions = {
        0: (0.08, 0.82),
        1: (0.08, 0.62),
        2: (0.35, 0.82),
        3: (0.35, 0.62),
        4: (0.35, 0.42),
        5: (0.62, 0.62),
        6: (0.62, 0.42),
        7: (0.62, 0.22),
        8: (0.35, 0.22),
        9: (0.08, 0.22),
    }

    arrows = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7), (7, 8), (8, 9)]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_axis_off()

    for i, label in enumerate(labels):
        x, y = positions[i]
        ax.text(
            x, y, label,
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black", lw=1.0),
        )

    for a, b in arrows:
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.2, shrinkA=18, shrinkB=18),
        )

    ax.text(
        0.5, 0.04,
        "FLOW-CM builds a deterministic baryon-derived network response; no galaxy-by-galaxy halo parameters are fitted.",
        ha="center", va="center", fontsize=10,
    )

    save_current(FIG_DIR / "model_pipeline.pdf")
    save_current(FIG_DIR / "model_pipeline.png")


# ============================================================
# FIGURE 2: PERFORMANCE TABLE AS PDF
# ============================================================

def plot_model_performance_table(final: pd.DataFrame) -> None:
    display = final[["model_label", "mean_mse", "median_mse", "mean_rmse", "comment"]].copy()

    for col in ["mean_mse", "median_mse", "mean_rmse"]:
        display[col] = display[col].map(lambda x: "--" if pd.isna(x) else f"{x:.2f}")

    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.axis("off")

    table = ax.table(
        cellText=display.values,
        colLabels=["Model", "Mean MSE", "Median MSE", "Mean RMSE", "Comment"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    ax.set_title("Main model comparison on SPARC/Rotmod", fontsize=12, pad=12)
    save_current(FIG_DIR / "model_performance_table.pdf")
    save_current(FIG_DIR / "model_performance_table.png")


# ============================================================
# FIGURE 3: MSE RATIO HISTOGRAM
# ============================================================

def plot_mse_ratio_histogram(per_gal: pd.DataFrame) -> None:
    if "ratio_FLOW_over_RAR" not in per_gal.columns:
        print("Skipping MSE ratio histogram: ratio_FLOW_over_RAR not available.")
        return

    ratio = per_gal["ratio_FLOW_over_RAR"].replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        print("Skipping MSE ratio histogram: no finite ratios.")
        return

    plt.figure(figsize=(7.5, 5))
    plt.hist(ratio, bins=30, edgecolor="black")
    plt.axvline(1.0, linestyle="--", linewidth=1.2)
    plt.xlabel(r"Per-galaxy MSE ratio: FLOW-CM / RAR-MOND")
    plt.ylabel("Number of galaxies")
    plt.title("Distribution of per-galaxy MSE ratios")
    plt.text(
        0.98, 0.95,
        f"Median ratio = {ratio.median():.3f}\nFLOW wins = {(ratio < 1).mean():.2%}",
        transform=plt.gca().transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=0.8),
    )

    save_current(FIG_DIR / "mse_ratio_histogram.pdf")
    save_current(FIG_DIR / "mse_ratio_histogram.png")


# ============================================================
# FIGURE 4: PERMUTATION TEST
# ============================================================

def plot_permutation_test(
    final: pd.DataFrame,
    perm_by_model: Optional[pd.DataFrame],
    flow_model: str,
) -> None:
    flow_row = final[final["model_key"] == flow_model].iloc[0]
    real = float(flow_row["mean_mse"])
    permuted = float(flow_row["permuted_mean_mse"]) if pd.notna(flow_row["permuted_mean_mse"]) else np.nan

    if pd.isna(permuted) and perm_by_model is not None and len(perm_by_model):
        candidates = perm_by_model.copy()
        if "mean_true_mse" in candidates.columns and "mean_permuted_mse" in candidates.columns:
            candidates["distance"] = np.abs(candidates["mean_true_mse"] - real)
            best = candidates.sort_values("distance").iloc[0]
            permuted = float(best["mean_permuted_mse"])

    if pd.isna(permuted):
        print("Skipping permutation plot: no permuted MSE found.")
        return

    plt.figure(figsize=(6, 5))
    labels = ["Real geometry", "Permuted geometry"]
    values = [real, permuted]
    plt.bar(labels, values, edgecolor="black")
    plt.ylabel("Mean galaxy MSE")
    plt.title("Geometric permutation test")

    for x, v in enumerate(values):
        plt.text(x, v, f"{v:.2f}", ha="center", va="bottom")

    plt.text(
        0.5, 0.92,
        f"Degradation: {permuted - real:+.2f}",
        transform=plt.gca().transAxes,
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=0.8),
    )

    save_current(FIG_DIR / "permutation_test.pdf")
    save_current(FIG_DIR / "permutation_test.png")


# ============================================================
# FIGURE 5: ROTATION CURVE EXAMPLES
# ============================================================

def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def plot_rotation_curve_examples(
    point_predictions: Optional[pd.DataFrame],
    per_gal: pd.DataFrame,
    flow_model: str,
) -> None:
    if point_predictions is None or point_predictions.empty:
        print("Skipping rotation curve examples: point_predictions.csv not available.")
        return

    df = point_predictions.copy()

    gal_col = find_column(df, ["galaxy"])
    model_col = find_column(df, ["model"])
    r_col = find_column(df, ["r_kpc", "r", "radius"])
    vobs_col = find_column(df, ["vobs", "v_obs", "Vobs"])
    vpred_col = find_column(df, ["vpred", "v_model", "Vpred"])
    err_col = find_column(df, ["evobs", "e_vobs", "err", "verr"])

    required = [gal_col, model_col, r_col, vobs_col, vpred_col]
    if any(c is None for c in required):
        print("Skipping rotation curve examples: point_predictions.csv lacks required columns.")
        print("Found columns:", list(df.columns))
        return

    available_models = set(df[model_col].astype(str))
    flow_plot_model = flow_model if flow_model in available_models else None

    if flow_plot_model is None:
        candidates = [m for m in available_models if "harm" in m.lower() and "flow" in m.lower()]
        if candidates:
            flow_plot_model = sorted(candidates)[0]

    if flow_plot_model is None:
        print("Skipping rotation curve examples: FLOW model not found in point_predictions.csv.")
        return

    models_to_plot = [m for m in [MODEL_NEWTON, MODEL_RAR, MODEL_BASE, flow_plot_model] if m in available_models]
    model_labels = {
        MODEL_NEWTON: "Newton",
        MODEL_RAR: "RAR/MOND",
        MODEL_BASE: "FLOW base",
        flow_plot_model: "FLOW-CM",
    }

    if {"FLOW_harmonic", "RAR_MOND"}.issubset(per_gal.columns):
        pg = per_gal.dropna(subset=["FLOW_harmonic", "RAR_MOND"]).copy()
    else:
        pg = per_gal.copy()

    if "ratio_FLOW_over_RAR" in pg.columns:
        pg = pg.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio_FLOW_over_RAR"])
        if PREFER_FLOW_WIN_EXAMPLES:
            selected = list(pg.sort_values("ratio_FLOW_over_RAR").head(N_EXAMPLE_GALAXIES)["galaxy"])
        else:
            pg_sorted = pg.sort_values("ratio_FLOW_over_RAR")
            n = len(pg_sorted)
            idxs = np.linspace(0, max(n - 1, 0), N_EXAMPLE_GALAXIES).astype(int)
            selected = list(pg_sorted.iloc[idxs]["galaxy"])
    else:
        rng = np.random.default_rng(RANDOM_SEED)
        galaxies = sorted(df[gal_col].dropna().unique())
        selected = list(rng.choice(galaxies, size=min(N_EXAMPLE_GALAXIES, len(galaxies)), replace=False))

    n = len(selected)
    if n == 0:
        print("Skipping rotation curve examples: no selected galaxies.")
        return

    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.4 * nrows), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, gal in zip(axes.ravel(), selected):
        ax.axis("on")
        gdf = df[df[gal_col] == gal]

        obs = gdf[gdf[model_col] == flow_plot_model].sort_values(r_col)
        if obs.empty:
            obs = gdf.sort_values(r_col)

        obs_unique = obs.drop_duplicates(subset=[r_col])

        if err_col:
            ax.errorbar(
                obs_unique[r_col],
                obs_unique[vobs_col],
                yerr=obs_unique[err_col],
                fmt="o",
                ms=3,
                capsize=1.5,
                label="Observed",
            )
        else:
            ax.plot(obs_unique[r_col], obs_unique[vobs_col], "o", ms=3, label="Observed")

        for m in models_to_plot:
            mdf = gdf[gdf[model_col] == m].sort_values(r_col)
            if not mdf.empty:
                ax.plot(mdf[r_col], mdf[vpred_col], linewidth=1.3, label=model_labels.get(m, m))

        ax.set_title(str(gal), fontsize=9)
        ax.set_xlabel("r [kpc]", fontsize=8)
        ax.set_ylabel("v [km/s]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), fontsize=8)
    fig.suptitle("Representative rotation curves", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "rotation_curve_examples.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "rotation_curve_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CAPTION SNIPPETS AND COVER LETTER
# ============================================================

def write_caption_snippets() -> None:
    captions = r"""
% Publication figure snippets for FLOW-CM

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{figures/model_pipeline.pdf}
\caption{Schematic structure of FLOW-CM. The baryonic distribution generates nodal occupation, structural collectivity, inward memory, basal connectivity, radial transition, graph current, harmonic conductance, and finally the predicted rotation curve.}
\label{fig:pipeline}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.98\textwidth]{figures/rotation_curve_examples.pdf}
\caption{Representative rotation curves comparing observed velocities, baryonic Newtonian predictions, the RAR/MOND benchmark, and harmonic FLOW-CM.}
\label{fig:curves}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.78\textwidth]{figures/mse_ratio_histogram.pdf}
\caption{Distribution of per-galaxy MSE ratios between FLOW-CM and the external RAR/MOND benchmark. Values below unity indicate galaxies for which FLOW-CM gives a lower velocity MSE.}
\label{fig:mse_ratio}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.72\textwidth]{figures/permutation_test.pdf}
\caption{Geometric permutation test. The increase in MSE after permuting $Q_{\rm harm}$ between galaxies indicates that the radial nodal conductance contains galaxy-specific information.}
\label{fig:perm}
\end{figure}
""".strip()

    (TABLE_DIR / "figure_caption_snippets.tex").write_text(captions + "\n", encoding="utf-8")


def write_cover_letter(final: pd.DataFrame) -> None:
    flow = final[final["model_label"] == "Harmonic FLOW-CM"].iloc[0]
    rar = final[final["model_label"] == "RAR/MOND benchmark"].iloc[0]
    newton = final[final["model_label"] == "Baryonic Newtonian"].iloc[0]

    cover = rf"""\documentclass[11pt,a4paper]{{letter}}
\usepackage[margin=2.5cm]{{geometry}}
\usepackage{{hyperref}}

\signature{{Eliezer Siqueira}}
\address{{Independent Researcher}}

\begin{{document}}

\begin{{letter}}{{Editorial Office}}

\opening{{Dear Editor,}}

I am pleased to submit the manuscript entitled \emph{{"FLOW-CM: A Phenomenological Nodal-Conductance Law for Galactic Rotation Curves"}} for consideration.

The manuscript introduces FLOW-CM, a phenomenological effective law for galactic rotation curves motivated by a vacuum-flow framework. Instead of fitting dark-matter halos galaxy by galaxy, the model constructs a deterministic baryon-derived network response based on nodal occupation, structural collectivity, inward memory, graph current, and a harmonic conductance between two geometric channels.

Using the SPARC/Rotmod sample of 175 disk galaxies and fixed mass-to-light ratios, the final harmonic FLOW-CM law obtains a mean galaxy MSE of approximately {flow['mean_mse']:.2f}, compared with {rar['mean_mse']:.2f} for the RAR/MOND benchmark used in the manuscript and {newton['mean_mse']:.2f} for baryonic Newtonian dynamics. A geometric permutation test further suggests that the internal conductance variable carries galaxy-specific structural information rather than acting as a generic saturation factor.

The manuscript is intentionally cautious in scope. FLOW-CM is presented as an effective galactic-scale law, not as a complete covariant theory. The paper discusses its limitations explicitly, including the need for weighted likelihood analyses, independent data validation, lensing tests, and a future variational derivation.

I believe the work may be of interest to readers studying galaxy dynamics, modified gravity phenomenology, and baryon-driven effective descriptions of the radial acceleration relation.

Thank you for your consideration.

\closing{{Sincerely,}}

\end{{letter}}
\end{{document}}
"""

    (PUB_DIR / "cover_letter_FLOW_CM.tex").write_text(cover, encoding="utf-8")


def write_readme(flow_model: str) -> None:
    readme = f"""# FLOW-CM publication assets

Generated files:

## Figures
- figures/model_pipeline.pdf
- figures/permutation_test.pdf
- figures/mse_ratio_histogram.pdf
- figures/model_performance_table.pdf
- figures/rotation_curve_examples.pdf, if point_predictions.csv was available

## Tables
- tables/final_results_publication.csv
- tables/per_galaxy_publication_metrics.csv
- tables/figure_caption_snippets.tex

## Submission draft
- cover_letter_FLOW_CM.tex

Final FLOW model used: `{flow_model}`

Before journal submission:
1. Confirm that `summary_models.csv` and `detail_by_galaxy_model.csv` come from the final official run.
2. Confirm the RAR/MOND benchmark value used in the paper.
3. Re-run this script after any final model run.
4. Place the generated `figures/` folder next to the LaTeX manuscript.
5. Compile the manuscript twice.
"""

    (PUB_DIR / "README_publication_assets.md").write_text(readme, encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_dirs()

    summary = read_csv_required(RESULTS_DIR / "summary_models.csv")
    detail = read_csv_required(RESULTS_DIR / "detail_by_galaxy_model.csv")
    perm_by_model = read_csv_optional(RESULTS_DIR / "permutation_by_model.csv")
    point_predictions = read_csv_optional(RESULTS_DIR / "point_predictions.csv")

    final, per_gal, flow_model = build_publication_tables(summary, detail, perm_by_model)

    print("Using FLOW model:", flow_model)
    print("\nPublication results table:")
    print(final.to_string(index=False))

    plot_model_pipeline()
    plot_model_performance_table(final)
    plot_mse_ratio_histogram(per_gal)
    plot_permutation_test(final, perm_by_model, flow_model)
    plot_rotation_curve_examples(point_predictions, per_gal, flow_model)

    write_caption_snippets()
    write_cover_letter(final)
    write_readme(flow_model)

    print("\nPublication assets generated in:", PUB_DIR.resolve())
    print("\nEssential files:")
    print("-", (FIG_DIR / "model_pipeline.pdf").resolve())
    print("-", (FIG_DIR / "model_performance_table.pdf").resolve())
    print("-", (FIG_DIR / "mse_ratio_histogram.pdf").resolve())
    print("-", (FIG_DIR / "permutation_test.pdf").resolve())

    if (FIG_DIR / "rotation_curve_examples.pdf").exists():
        print("-", (FIG_DIR / "rotation_curve_examples.pdf").resolve())
    else:
        print("- rotation_curve_examples.pdf was skipped because point_predictions.csv was not available.")

    print("-", (TABLE_DIR / "final_results_publication.csv").resolve())
    print("-", (PUB_DIR / "cover_letter_FLOW_CM.tex").resolve())


if __name__ == "__main__":
    main()
