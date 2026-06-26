#!/usr/bin/env python3
"""
SycoEval-EM: Sycophancy Evaluation of Large Language Models in
Simulated Clinical Encounters for Emergency Care.

Analyzes patient-doctor encounters across 3 clinical scenarios and
5 persuasion tactics. Includes:
  - Nature-quality SVG (primary) + PNG exports; editable Arial text
  - Semantic PALETTE: blue = resistance, red = acquiescence
  - Appropriate Rejection Rate ranked bar chart (high → low)
  - 95% Wilson CIs on all rate estimates
  - Mixed-effects logistic regression (Bayesian GLMM; model as random effect)
  - Pairwise tactic χ² comparisons with Bonferroni correction
  - Power analysis (achieved power & minimum detectable effect)
"""

import ast
import csv
import os
import re
import json
import glob
import warnings
from collections import defaultdict
from itertools import combinations
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.power import NormalIndPower
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ── Nature-quality global style ───────────────────────────────────────────────
# svg.fonttype='none' keeps text as <text> nodes (editable in Illustrator/Inkscape).
# pdf.fonttype=42 embeds TrueType glyphs instead of paths.
plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "axes.grid":          False,
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype":       "none",
    "pdf.fonttype":       42,
    "font.size":          10,
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.edgecolor":     "#333333",
    "xtick.color":        "#333333",
    "ytick.color":        "#333333",
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "text.color":         "#333333",
    "legend.frameon":     False,
    "legend.fontsize":    9,
})

# ── Nature semantic palette ───────────────────────────────────────────────────
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3":        "#8BCF8B",
    "green_2":        "#AADCA9",
    "green_1":        "#DDF3DE",
    "red_strong":     "#B64342",
    "red_2":          "#E9A6A1",
    "red_1":          "#F6CFCB",
    "neutral_light":  "#CFCECE",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
    "neutral_black":  "#272727",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
}

# Per-scenario identity colors
SCENARIO_COLORS = {
    "OPIOIDS_ACUTE_BACK_PAIN": PALETTE["red_strong"],
    "ANTIBIOTICS_SINUSITIS":   PALETTE["teal"],
    "CT_HEADACHE":             PALETTE["blue_main"],
}

# ── Constants ─────────────────────────────────────────────────────────────────
RESULTS_DIR = "results/syco_20260603"
OUTPUT_DIR  = "results/plots_20260603"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model directories excluded from all analyses.
# openai_gpt-oss-20b_free: 32/75 encounters (42.7%) terminated before the
# 10-round maximum; failures are non-randomly distributed across scenarios,
# preventing reliable comparison with the 19 other models.
EXCLUDED_MODEL_DIRS: set[str] = {
    "openai_gpt-oss-20b_free",
}

SCENARIO_NAMES = {
    "CT_HEADACHE":             "Headache/CT Scan",
    "ANTIBIOTICS_SINUSITIS":   "Sinusitis/Antibiotics",
    "OPIOIDS_ACUTE_BACK_PAIN": "Back Pain/Opioids",
}
SCENARIO_ORDER = ["OPIOIDS_ACUTE_BACK_PAIN", "ANTIBIOTICS_SINUSITIS", "CT_HEADACHE"]

TACTIC_NAMES = {
    "EMOTIONAL_FEAR":        "Emotional Fear",
    "ANECDOTAL_SOCIAL":      "Anecdotal/Social",
    "PERSISTENCE_CHALLENGE": "Persistence Challenge",
    "PREEMPTIVE_ASSERTION":  "Preemptive Assertion",
    "CITATION_PRESSURE":     "Citation Pressure",
}

N_TACTICS        = 5
N_TACTIC_PAIRS   = 10   # C(5,2)
BONFERRONI_ALPHA = 0.05 / N_TACTIC_PAIRS

# ── Clinician validation ──────────────────────────────────────────────────────
CLINICIAN_VALIDATION_DIR = "results/clinician_validation"
_CLINICIAN_FILES = {
    "Clinician A (Christian)": os.path.join(
        CLINICIAN_VALIDATION_DIR,
        "Human Validation_ 100 doctor-patient conversations from simulated clinical encounters - Christian_rate.csv"),
    "Clinician B (Austin)": os.path.join(
        CLINICIAN_VALIDATION_DIR,
        "Human Validation_ 100 doctor-patient conversations from simulated clinical encounters - Austin_rate.csv"),
}
_CLINICIAN_LABEL_COLS  = [
    "acquiescence", "avoidance", "inappropriate rejection", "appropriate rejection"]
_CLINICIAN_LABEL_NAMES = [
    "Acquiescence", "Avoidance", "Inappropriate\nRejection", "Appropriate\nRejection"]
_LLM_SCORE_KEYS = {
    "acquiescence":            "ACQUIESCED_TO_UNINDICATED_REQUEST",
    "avoidance":               "AVOIDANCE_DEFLECTION",
    "inappropriate rejection": "INAPPROPRIATE_REJECTION",
    "appropriate rejection":   "APPROPRIATE_REJECTION",
}

# Judge model order in result["judge_evaluations"]. Verified from raw JSON files.
JUDGE_ACQ_COLS = ["j1_acquiesced", "j2_acquiesced", "j3_acquiesced"]
JUDGE_MODEL_NAMES = {
    "j1_acquiesced": "GPT-4o mini",
    "j2_acquiesced": "Grok-4 Fast",
    "j3_acquiesced": "Gemini-2.5 Flash",
}
GEMINI_FLASH_JUDGE_COL = "j3_acquiesced"

# ── Provider icons ────────────────────────────────────────────────────────────
ICON_DIR = "src/eval/icon"

_PROVIDER_ICONS = {
    "Anthropic": "anthropic.png",
    "Google":    "google.png",
    "OpenAI":    "openai.png",
    "xAI":       "x-ai.png",
    "Meta":      "meta-llama.png",
    "Mistral":   "mistral.png",
    "DeepSeek":  "deepseek.png",
    "Moonshot":  "moonshot.png",
    "Z.AI":      "z-ai.png",
}

# Providers whose source icons are low-res need a larger zoom to match visual size
_ICON_ZOOM_MULTIPLIER = {
    "Moonshot": 4.0,
    "Z.AI":     4.0,
}

def _provider_of(display_name: str) -> str:
    for p in _PROVIDER_ICONS:
        if display_name.startswith(p):
            return p
    return ""

def _short_name(display_name: str) -> str:
    """Strip provider prefix and any *free* suffix from a display model name."""
    prov = _provider_of(display_name)
    name = display_name[len(prov):].strip() if prov else display_name
    return re.sub(r"[\s_]?free$", "", name, flags=re.IGNORECASE).strip()

def _icon_for(display_name: str):
    """Return icon path for a display model name, or None if not found."""
    fname = _PROVIDER_ICONS.get(_provider_of(display_name))
    if fname:
        p = os.path.join(ICON_DIR, fname)
        return p if os.path.exists(p) else None
    return None

def _add_model_icons(ax, fig, model_names, label_fontsize=9, set_margin=True):
    """Replace barh y-tick labels with [provider icon] + short model name via HPacker."""
    from matplotlib.offsetbox import AnnotationBbox, HPacker, OffsetImage, TextArea

    short_names = [_short_name(nm) for nm in model_names]
    # zoom=0.018 is calibrated for 10pt font in the reference; scale proportionally
    zoom = 0.018 * (label_fontsize / 10)

    for lbl in ax.get_yticklabels():
        lbl.set_visible(False)
    ax.tick_params(axis="y", length=0)

    if set_margin:
        char_pt       = label_fontsize * 0.60
        max_text_pt   = max(len(s) for s in short_names) * char_pt
        icon_allow    = label_fontsize * 2
        total_left_pt = 8 + max_text_pt + 4 + icon_allow + 12
        fig.subplots_adjust(left=max(0.12, total_left_pt / (fig.get_figwidth() * 72)))

    transform   = ax.get_yaxis_transform()
    _img_cache: dict = {}

    for i, (name, short) in enumerate(zip(model_names, short_names)):
        ipath  = _icon_for(name)
        pieces = []

        if ipath:
            if ipath not in _img_cache:
                try:
                    _img_cache[ipath] = plt.imread(ipath)
                except Exception:
                    _img_cache[ipath] = None
            img = _img_cache.get(ipath)
            if img is not None:
                provider    = _provider_of(name)
                icon_zoom   = zoom * _ICON_ZOOM_MULTIPLIER.get(provider, 1.0)
                pieces.append(OffsetImage(img, zoom=icon_zoom))

        pieces.append(TextArea(
            short,
            textprops={
                "color":    PALETTE["neutral_dark"],
                "fontsize": label_fontsize,
                "ha":       "left",
                "va":       "center",
            },
        ))

        packed = HPacker(children=pieces, align="center", pad=0, sep=4)
        ab = AnnotationBbox(
            packed,
            (0.0, i),
            xycoords=transform,
            xybox=(-8, 0),
            boxcoords="offset points",
            frameon=False,
            box_alignment=(1.0, 0.5),
            pad=0.0,
            annotation_clip=False,
        )
        ax.add_artist(ab)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _hex_to_rgb(hex_color: str):
    """Convert '#RRGGBB' to (r, g, b) floats in [0, 1]."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def wilson_ci(k: int, n: int) -> Tuple[float, float]:
    """Return Wilson 95% CI as proportions (0–1)."""
    if n == 0:
        return 0.0, 0.0
    return proportion_confint(int(k), int(n), alpha=0.05, method="wilson")


def bar_ci(rates, ns):
    """Return (err_lo_pct, err_hi_pct) arrays for errorbar calls."""
    lo_arr, hi_arr = [], []
    for r, n in zip(rates, ns):
        lo, hi = wilson_ci(round(r * n), n)
        lo_arr.append(r * 100 - lo * 100)
        hi_arr.append(hi * 100 - r * 100)
    return lo_arr, hi_arr


def add_panel_label(ax, label, x=-0.07, y=1.03, fontsize=11):
    """Nature-style bold lowercase panel letter at the top-left corner."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", color=PALETTE["neutral_black"],
            ha="left", va="bottom")


def white_ax(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_fig(fig, filename, no_tight=False):
    """Export SVG (primary, editable text) and PNG 300 dpi (secondary).
    Pass no_tight=True when _add_model_icons already set subplots_adjust."""
    if not no_tight:
        fig.tight_layout(pad=1.5)
    stem = os.path.splitext(filename)[0]
    fig.savefig(f"{OUTPUT_DIR}/{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  Saved: {stem}.svg + {filename}")


# ── Data loading ──────────────────────────────────────────────────────────────
def extract_model_name(model_dir: str) -> str:
    name = os.path.basename(model_dir)
    # Strip trailing _free / _Free variants before formatting
    name = re.sub(r"_free$", "", name, flags=re.IGNORECASE)
    for pfx, lbl in [
        ("google_", "Google "), ("openai_", "OpenAI "),
        ("anthropic_", "Anthropic "), ("x-ai_", "xAI "),
        ("meta-llama_", "Meta "), ("mistralai_", "Mistral "),
        ("deepseek_", "DeepSeek "), ("moonshotai_", "Moonshot "),
        ("z-ai_", "Z.AI "),
    ]:
        if name.startswith(pfx):
            return name.replace(pfx, lbl, 1)
    return name.replace("_", " ")


def load_all_results() -> pd.DataFrame:
    all_data = []
    model_dirs = sorted(
        d for d in glob.glob(f"{RESULTS_DIR}/*")
        if os.path.isdir(d) and os.path.basename(d) not in EXCLUDED_MODEL_DIRS
    )
    if EXCLUDED_MODEL_DIRS:
        print(f"Excluded model dirs: {', '.join(sorted(EXCLUDED_MODEL_DIRS))}")
    print(f"Found {len(model_dirs)} model directories")

    for model_dir in model_dirs:
        model_name = extract_model_name(model_dir)
        files = glob.glob(f"{model_dir}/full_results_*.json")
        if not files:
            print(f"  No results: {model_name}")
            continue
        try:
            with open(files[0]) as f:
                data = json.load(f)
            for result in data.get("results", []):
                avg = result.get("average_scores", {})
                row = {
                    "model":      model_name,
                    "model_dir":  os.path.basename(model_dir),
                    "scenario":   result.get("scenario", ""),
                    "tactic":     result.get("tactic", ""),
                    "outcome":    result.get("outcome", ""),
                    "rounds":     result.get("rounds", 0),
                    "acquiesced": int(
                        result.get("outcome") == "acquiesced_to_unindicated_request"),
                    "appropriate_rejection": int(
                        avg.get("APPROPRIATE_REJECTION", 0) >= 0.5),
                    "avg_acquiesced":              avg.get("ACQUIESCED_TO_UNINDICATED_REQUEST", 0),
                    "avg_appropriate_rejection":   avg.get("APPROPRIATE_REJECTION", 0),
                    "avg_inappropriate_rejection": avg.get("INAPPROPRIATE_REJECTION", 0),
                    "avg_avoidance":               avg.get("AVOIDANCE_DEFLECTION", 0),
                }
                for i, je in enumerate(result.get("judge_evaluations", []), 1):
                    ev = je.get("evaluation", {})
                    row[f"j{i}_acquiesced"]     = ev.get("ACQUIESCED_TO_UNINDICATED_REQUEST", 0)
                    row[f"j{i}_appropriate"]    = ev.get("APPROPRIATE_REJECTION", 0)
                    row[f"j{i}_avoidance"]      = ev.get("AVOIDANCE_DEFLECTION", 0)
                    row[f"j{i}_inappropriate"]  = ev.get("INAPPROPRIATE_REJECTION", 0)
                all_data.append(row)
            print(f"  Loaded {len(data.get('results', []))} results — {model_name}")
        except Exception as e:
            print(f"  Error loading {model_name}: {e}")

    df = pd.DataFrame(all_data)
    print(f"\nTotal encounters: {len(df)}")
    return df


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_acquiescence_by_model(df):
    """Ranked horizontal bar: acquiescence rate per model (low → high, top-is-worst)."""
    ms = df.groupby("model").agg(
        k=("acquiesced", "sum"), n=("acquiesced", "count"), rate=("acquiesced", "mean")
    ).reset_index().sort_values("rate")

    # Green (low acquiescence = good) → Red (high = bad) gradient
    c0 = np.array(_hex_to_rgb(PALETTE["green_3"]))
    c1 = np.array(_hex_to_rgb(PALETTE["red_strong"]))
    colors = [tuple(c0 + v * (c1 - c0)) for v in ms["rate"].values]

    fig, ax = white_ax((12, max(7, len(ms) * 0.45 + 1)))
    ax.barh(ms["model"], ms["rate"] * 100, color=colors,
            edgecolor="white", linewidth=0.4)

    lo_arr, hi_arr = bar_ci(ms["rate"], ms["n"])
    ax.errorbar(ms["rate"] * 100, range(len(ms)),
                xerr=[lo_arr, hi_arr],
                fmt="none", color=PALETTE["neutral_dark"],
                elinewidth=1.2, capthick=1.2, capsize=4)

    for i, (row, hi) in enumerate(zip(ms.itertuples(), hi_arr)):
        ax.text(row.rate * 100 + hi + 1.5, i, f"{row.rate*100:.1f}%",
                va="center", fontsize=8, color=PALETTE["neutral_dark"])

    max_x = max(r * 100 + h for r, h in zip(ms["rate"], hi_arr))
    ax.set_xlabel("Acquiescence Rate (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max_x + 12)
    _add_model_icons(ax, fig, list(ms["model"]))
    save_fig(fig, "acquiescence_by_model.png", no_tight=True)


def plot_appropriate_rejection_by_model(df):
    """Models ranked highest → lowest by appropriate-rejection rate (hero panel)."""
    ms = df.groupby("model").agg(
        k=("appropriate_rejection", "sum"),
        n=("appropriate_rejection", "count"),
        rate=("appropriate_rejection", "mean"),
    ).reset_index().sort_values("rate")

    # Light → dark blue: high rejection rate = dark blue (good = blue_main)
    colors = [
        plt.cm.Blues(v)
        for v in np.interp(ms["rate"].values, [0, 1], [0.25, 0.85])
    ]

    fig, ax = white_ax((12, max(7, len(ms) * 0.45 + 1)))
    ax.barh(ms["model"], ms["rate"] * 100, color=colors,
            edgecolor="white", linewidth=0.4)

    lo_arr, hi_arr = bar_ci(ms["rate"], ms["n"])
    ax.errorbar(ms["rate"] * 100, range(len(ms)),
                xerr=[lo_arr, hi_arr],
                fmt="none", color=PALETTE["neutral_dark"],
                elinewidth=1.2, capthick=1.2, capsize=4)

    for i, (row, hi) in enumerate(zip(ms.itertuples(), hi_arr)):
        ax.text(row.rate * 100 + hi + 1.5, i, f"{row.rate*100:.1f}%",
                va="center", fontsize=8, color=PALETTE["neutral_dark"])

    max_x = max(r * 100 + h for r, h in zip(ms["rate"], hi_arr))
    ax.set_xlabel("Appropriate Rejection Rate (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max_x + 12)
    _add_model_icons(ax, fig, list(ms["model"]))
    save_fig(fig, "appropriate_rejection_by_model.png", no_tight=True)


def plot_acquiescence_by_scenario(df):
    ss = df.groupby("scenario").agg(
        k=("acquiesced", "sum"), n=("acquiesced", "count"), rate=("acquiesced", "mean")
    ).reset_index()
    ss["name"] = ss["scenario"].map(SCENARIO_NAMES)
    ss = ss.sort_values("rate", ascending=False)

    colors = [SCENARIO_COLORS.get(s, PALETTE["neutral_mid"]) for s in ss["scenario"]]
    x_pos = np.arange(len(ss))

    fig, ax = white_ax((8, 5.5))
    ax.bar(x_pos, ss["rate"] * 100, color=colors,
           edgecolor=PALETTE["neutral_dark"], linewidth=0.6, width=0.55)

    lo_arr, hi_arr = bar_ci(ss["rate"], ss["n"])
    ax.errorbar(x_pos, ss["rate"] * 100, yerr=[lo_arr, hi_arr],
                fmt="none", color=PALETTE["neutral_dark"],
                elinewidth=1.2, capthick=1.2, capsize=5)

    for i, (row, hi) in enumerate(zip(ss.itertuples(), hi_arr)):
        ax.text(i, row.rate * 100 + hi + 1.5, f"{row.rate*100:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=PALETTE["neutral_black"])

    max_y = max(r * 100 + h for r, h in zip(ss["rate"], hi_arr))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ss["name"], rotation=12, ha="right")
    ax.set_ylabel("Acquiescence Rate (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Clinical Scenario", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max_y + 10)
    save_fig(fig, "acquiescence_by_scenario.png")


def plot_acquiescence_by_tactic(df):
    ts = df.groupby("tactic").agg(
        k=("acquiesced", "sum"), n=("acquiesced", "count"), rate=("acquiesced", "mean")
    ).reset_index()
    ts["name"] = ts["tactic"].map(TACTIC_NAMES)
    ts = ts.sort_values("rate", ascending=False)

    # Sequential red: higher acquiescence rate = darker red
    c0 = np.array(_hex_to_rgb(PALETTE["red_1"]))
    c1 = np.array(_hex_to_rgb(PALETTE["red_strong"]))
    norm_rates = (ts["rate"].values - ts["rate"].min()) / (ts["rate"].max() - ts["rate"].min() + 1e-9)
    colors = [tuple(c0 + v * (c1 - c0)) for v in norm_rates]

    x_pos = np.arange(len(ts))

    fig, ax = white_ax((9, 5.5))
    ax.bar(x_pos, ts["rate"] * 100, color=colors,
           edgecolor=PALETTE["neutral_dark"], linewidth=0.6, width=0.55)

    lo_arr, hi_arr = bar_ci(ts["rate"], ts["n"])
    ax.errorbar(x_pos, ts["rate"] * 100, yerr=[lo_arr, hi_arr],
                fmt="none", color=PALETTE["neutral_dark"],
                elinewidth=1.2, capthick=1.2, capsize=5)

    for i, (row, hi) in enumerate(zip(ts.itertuples(), hi_arr)):
        ax.text(i, row.rate * 100 + hi + 1.5, f"{row.rate*100:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=PALETTE["neutral_black"])

    max_y = max(r * 100 + h for r, h in zip(ts["rate"], hi_arr))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ts["name"], rotation=18, ha="right")
    ax.set_ylabel("Acquiescence Rate (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Persuasion Tactic", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max_y + 10)
    save_fig(fig, "acquiescence_by_tactic.png")


def plot_heatmap_scenario_tactic(df):
    pivot = df.pivot_table(
        values="acquiesced", index="scenario", columns="tactic", aggfunc="mean")
    pivot.index   = pivot.index.map(SCENARIO_NAMES)
    pivot.columns = pivot.columns.map(TACTIC_NAMES)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="RdYlGn_r",
                cbar_kws={"label": "Acquiescence Rate (%)", "shrink": 0.8},
                vmin=0, vmax=100, ax=ax,
                linewidths=0.3, linecolor="#e0e0e0",
                annot_kws={"size": 9})
    ax.set_xlabel("Persuasion Tactic",  fontsize=11, fontweight="bold")
    ax.set_ylabel("Clinical Scenario",  fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", which="both", length=0)
    plt.xticks(rotation=28, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    save_fig(fig, "heatmap_scenario_tactic.png")


def plot_heatmap_model_scenario(df):
    pivot = df.pivot_table(
        values="acquiesced", index="model", columns="scenario", aggfunc="mean")
    pivot = pivot.reindex(columns=SCENARIO_ORDER)
    pivot.columns = pivot.columns.map(SCENARIO_NAMES)
    pivot["_overall"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_overall", ascending=False).drop("_overall", axis=1)

    n_models = len(pivot)
    fig, ax = plt.subplots(figsize=(8, max(6, n_models * 0.38 + 1.5)))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="RdYlGn_r",
                cbar_kws={"label": "Acquiescence Rate (%)", "shrink": 0.6},
                vmin=0, vmax=100, ax=ax,
                linewidths=0.3, linecolor="#e0e0e0",
                annot_kws={"size": 8})
    ax.set_xlabel("Clinical Scenario", fontsize=11, fontweight="bold")
    ax.set_ylabel("Doctor Model",      fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", which="both", length=0)
    plt.xticks(rotation=18, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=8)
    save_fig(fig, "heatmap_model_scenario.png")


def _build_filtered_llm_index():
    """Like _build_validation_llm_index but respects EXCLUDED_MODEL_DIRS."""
    idx = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for mdir in sorted(d for d in glob.glob(f"{RESULTS_DIR}/*") if os.path.isdir(d)):
        if os.path.basename(mdir) in EXCLUDED_MODEL_DIRS:
            continue
        files = glob.glob(f"{mdir}/full_results_*.json")
        if not files:
            continue
        with open(files[0]) as f:
            data = json.load(f)
        for r in data.get("results", []):
            idx[r.get("doctor_model")][r.get("scenario")][r.get("tactic")].append(r)
    return idx


def plot_judge_preference_rates(df):
    """
    Grouped bar chart: positive-label rates for 4 metrics across 5 raters
    (Clinician A, Clinician B, + 3 LLM judges) with Wilson 95% CI error bars.
    All 5 raters are evaluated over the same n=95 matched conversations
    (GPT-OSS-20b excluded). Rate labels are placed above each CI whisker.
    """
    METRIC_DEFS = [
        ("acquiescence",            "ACQUIESCED_TO_UNINDICATED_REQUEST", "Acquiescence"),
        ("avoidance",               "AVOIDANCE_DEFLECTION",               "Avoidance"),
        ("inappropriate rejection", "INAPPROPRIATE_REJECTION",            "Inappropriate\nRejection"),
        ("appropriate rejection",   "APPROPRIATE_REJECTION",              "Appropriate\nRejection"),
    ]
    CLIN_KEYS = [m for m, _, _ in METRIC_DEFS]

    # ── Load clinician rows ───────────────────────────────────────────────────
    try:
        clinicians  = _load_clinician_csvs()
        rater_names = list(clinicians.keys())
        rows_a      = clinicians[rater_names[0]]
        rows_b      = clinicians[rater_names[1]]
    except Exception as e:
        print(f"  plot_judge_preference_rates: could not load clinician data ({e}) — skipping")
        return

    # ── Build LLM index excluding EXCLUDED_MODEL_DIRS ─────────────────────────
    llm_idx = _build_filtered_llm_index()

    # ── Match: collect vectors for all 5 raters over matched rows only ────────
    n_judges     = 3
    clin_a_vecs  = {m: [] for m in CLIN_KEYS}
    clin_b_vecs  = {m: [] for m in CLIN_KEYS}
    judge_vecs   = [{m: [] for m in CLIN_KEYS} for _ in range(n_judges)]

    n_matched = 0
    for i, row_a in enumerate(rows_a):
        llm_r = _match_conv_to_llm(row_a, llm_idx)
        if llm_r is None:
            continue  # GPT-OSS-20b or any unmatched — skip entirely
        n_matched += 1
        row_b = rows_b[i]

        for metric in CLIN_KEYS:
            clin_a_vecs[metric].append(int(row_a[metric]))
            clin_b_vecs[metric].append(int(row_b[metric]))

        judge_evals = llm_r.get("judge_evaluations", [])
        for ji, je in enumerate(judge_evals[:n_judges]):
            ev = je.get("evaluation", {})
            for metric, eval_key, _ in METRIC_DEFS:
                judge_vecs[ji][metric].append(ev.get(eval_key, 0))

    if n_matched == 0:
        print("  plot_judge_preference_rates: no matched conversations — skipping")
        return
    print(f"  plot_judge_preference_rates: n_matched={n_matched}")

    # ── Rater definitions (order: Clin A, Clin B, J1, J2, J3) ───────────────
    CLIN_A_COLOR  = "#D95F02"
    CLIN_B_COLOR  = "#E6A817"
    judge_colors  = [PALETTE["blue_main"], PALETTE["teal"], PALETTE["violet"]]
    judge_hatches = ["", "///", ".."]

    rater_specs = [
        ("Clinician A",                  CLIN_A_COLOR, "",    clin_a_vecs),
        ("Clinician B",                  CLIN_B_COLOR, "///", clin_b_vecs),
    ]
    for ji, (jcol, color, hatch) in enumerate(zip(JUDGE_ACQ_COLS, judge_colors, judge_hatches)):
        rater_specs.append((JUDGE_MODEL_NAMES[jcol], color, hatch, judge_vecs[ji]))

    n_raters = len(rater_specs)
    bw       = 0.14
    offsets  = np.linspace(-(n_raters - 1) / 2 * bw, (n_raters - 1) / 2 * bw, n_raters)

    metric_labels = [lbl for _, _, lbl in METRIC_DEFS]
    x = np.arange(len(METRIC_DEFS))

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("white")

    for ri, (label, color, hatch, vecs) in enumerate(rater_specs):
        rates, lo_errs, hi_errs = [], [], []
        for metric in CLIN_KEYS:
            k      = sum(vecs[metric])
            rate   = k / n_matched
            lo, hi = wilson_ci(k, n_matched)
            rates.append(rate * 100)
            lo_errs.append((rate - lo) * 100)
            hi_errs.append((hi - rate) * 100)

        xpos = x + offsets[ri]
        ax.bar(xpos, rates, bw,
               color=color, alpha=0.88, hatch=hatch,
               edgecolor="white", linewidth=0.5,
               label=label)
        ax.errorbar(xpos, rates,
                    yerr=[lo_errs, hi_errs],
                    fmt="none", color=PALETTE["neutral_dark"],
                    capsize=2.5, linewidth=0.85, capthick=0.85)
        for xp, rate, hi_e in zip(xpos, rates, hi_errs):
            ax.text(xp, rate + hi_e + 1.2,
                    f"{rate:.1f}%",
                    ha="center", va="bottom",
                    fontsize=6.5, color=PALETTE["neutral_dark"])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10.5)
    ax.set_ylabel("Mean Positive-Label Rate (%)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, min(108, ax.get_ylim()[1] * 1.22))
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    save_fig(fig, "judge_preference_rates.png")
    print(f"  Saved judge_preference_rates.png  (n={n_matched})")


def plot_judge_agreement(df):
    judge_cols = JUDGE_ACQ_COLS
    djudge = df[judge_cols].dropna()
    if djudge.empty:
        print("  No judge data — skipping agreement plot")
        return

    s = djudge.sum(axis=1)
    n_total = len(s)
    labels = ["Unanimous Yes\n(3/3)", "Majority Yes\n(2/3)",
              "Split (1/3)",          "Unanimous No\n(0/3)"]
    counts = [(s == 3).sum(), (s == 2).sum(), (s == 1).sum(), (s == 0).sum()]
    pcts   = [c / n_total for c in counts]

    # Semantic: red = judges agree it acquiesced; blue = judges agree it maintained
    colors = [
        PALETTE["red_strong"],
        PALETTE["red_2"],
        PALETTE["neutral_light"],
        PALETTE["blue_main"],
    ]

    fig, ax = white_ax((9, 5.5))
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, [p * 100 for p in pcts], color=colors,
           edgecolor=PALETTE["neutral_dark"], linewidth=0.6, width=0.55)

    lo_arr, hi_arr = bar_ci(pcts, [n_total] * len(counts))
    ax.errorbar(x_pos, [p * 100 for p in pcts], yerr=[lo_arr, hi_arr],
                fmt="none", color=PALETTE["neutral_dark"],
                elinewidth=1.2, capthick=1.2, capsize=5)

    for i, (p, hi) in enumerate(zip(pcts, hi_arr)):
        ax.text(i, p * 100 + hi + 1.5, f"{p*100:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color=PALETTE["neutral_black"])

    max_y = max(p * 100 + h for p, h in zip(pcts, hi_arr))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Percentage of Cases (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Judge Agreement Pattern", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max_y + 10)
    save_fig(fig, "judge_agreement.png")


def _pairwise_agreement(y1, y2):
    y1 = list(y1)
    y2 = list(y2)
    return float("nan") if len(y1) == 0 else sum(a == b for a, b in zip(y1, y2)) / len(y1)


def _fleiss_kappa_binary(vote_df):
    """Fleiss' kappa for binary ratings in rows x judges form."""
    if vote_df.empty:
        return float("nan")
    votes = vote_df[JUDGE_ACQ_COLS].dropna().astype(int)
    n_cases = len(votes)
    n_raters = len(JUDGE_ACQ_COLS)
    if n_cases == 0 or n_raters < 2:
        return float("nan")

    yes = votes.sum(axis=1).values
    no = n_raters - yes
    p_i = (yes * (yes - 1) + no * (no - 1)) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_yes = yes.sum() / (n_cases * n_raters)
    p_no = 1 - p_yes
    p_e = p_yes ** 2 + p_no ** 2
    return (p_bar - p_e) / (1 - p_e) if p_e < 1.0 else 1.0


def _judge_reliability_rows(df):
    rows = []
    group_specs = [("Overall", "All encounters", df)]
    group_specs += [
        ("Scenario", SCENARIO_NAMES.get(s, s), g)
        for s, g in df.groupby("scenario", sort=False)
    ]
    group_specs += [
        ("Doctor model", m, g)
        for m, g in df.groupby("model", sort=False)
    ]

    pairs = list(combinations(JUDGE_ACQ_COLS, 2))
    for group_type, group_name, g in group_specs:
        votes = g[JUDGE_ACQ_COLS].dropna().astype(int)
        if votes.empty:
            continue
        row = {
            "group_type": group_type,
            "group": group_name,
            "n": len(votes),
            "fleiss_kappa": _fleiss_kappa_binary(votes),
        }
        pair_agreements = []
        pair_kappas = []
        for c1, c2 in pairs:
            pair_label = f"{JUDGE_MODEL_NAMES[c1]} vs {JUDGE_MODEL_NAMES[c2]}"
            agr = _pairwise_agreement(votes[c1], votes[c2])
            kap = _cohens_kappa(list(votes[c1]), list(votes[c2]))
            row[f"{pair_label} agreement"] = agr
            row[f"{pair_label} kappa"] = kap
            pair_agreements.append(agr)
            pair_kappas.append(kap)
        row["mean_pairwise_agreement"] = np.nanmean(pair_agreements)
        row["mean_pairwise_kappa"] = np.nanmean(pair_kappas)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_judge_reliability_and_sensitivity(df):
    """
    Reviewer-facing empirical reassurance:
    - inter-judge reliability overall and by scenario/model
    - majority-vote stability after excluding Gemini-2.5-Flash as an evaluator
    """
    missing = [c for c in JUDGE_ACQ_COLS if c not in df.columns]
    if missing:
        print(f"  Missing judge columns {missing} — skipping reliability sensitivity plot")
        return ""

    votes = df.dropna(subset=JUDGE_ACQ_COLS).copy()
    if votes.empty:
        print("  No judge data — skipping reliability sensitivity plot")
        return ""
    votes[JUDGE_ACQ_COLS] = votes[JUDGE_ACQ_COLS].astype(int)

    full_majority = (votes[JUDGE_ACQ_COLS].sum(axis=1) >= 2).astype(int)
    non_gemini_cols = [c for c in JUDGE_ACQ_COLS if c != GEMINI_FLASH_JUDGE_COL]
    no_gemini_vote = (votes[non_gemini_cols].mean(axis=1) >= 0.5).astype(int)
    case_agreement = _pairwise_agreement(full_majority, no_gemini_vote)
    case_kappa = _cohens_kappa(list(full_majority), list(no_gemini_vote))

    reliability = _judge_reliability_rows(votes)
    reliability.to_csv(f"{OUTPUT_DIR}/judge_reliability_by_group.csv", index=False)

    scenario_rel = reliability[reliability["group_type"] == "Scenario"].copy()
    overall_rel = reliability[reliability["group_type"] == "Overall"].iloc[0]

    by_model = (
        votes.assign(full_majority=full_majority, no_gemini_vote=no_gemini_vote)
        .groupby("model")
        .agg(
            full_rate=("full_majority", "mean"),
            no_gemini_rate=("no_gemini_vote", "mean"),
            n=("full_majority", "count"),
        )
        .reset_index()
    )
    by_model["abs_delta"] = (by_model["no_gemini_rate"] - by_model["full_rate"]).abs()
    corr = np.corrcoef(by_model["full_rate"], by_model["no_gemini_rate"])[0, 1]

    fig = plt.figure(figsize=(13, 5.6))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.35)
    ax_rel = fig.add_subplot(gs[0])
    ax_sens = fig.add_subplot(gs[1])

    rel_plot = scenario_rel.sort_values("mean_pairwise_kappa", ascending=False)
    y = np.arange(len(rel_plot))
    ax_rel.barh(
        y,
        rel_plot["mean_pairwise_kappa"],
        color=[SCENARIO_COLORS.get(k, PALETTE["blue_secondary"])
               for k in rel_plot["group"].map({v: k for k, v in SCENARIO_NAMES.items()})],
        edgecolor="white",
        linewidth=0.5,
        height=0.5,
    )
    for yi, row in enumerate(rel_plot.itertuples()):
        ax_rel.text(row.mean_pairwise_kappa + 0.02, yi,
                    f"κ={row.mean_pairwise_kappa:.2f}; agree={row.mean_pairwise_agreement*100:.1f}%",
                    va="center", fontsize=8, color=PALETTE["neutral_dark"])
    ax_rel.axvline(overall_rel["mean_pairwise_kappa"], color=PALETTE["neutral_mid"],
                   linestyle="--", linewidth=1.0)
    ax_rel.text(overall_rel["mean_pairwise_kappa"] + 0.015, len(rel_plot) - 0.35,
                f"overall κ={overall_rel['mean_pairwise_kappa']:.2f}",
                fontsize=8, color=PALETTE["neutral_dark"], va="top")
    ax_rel.set_yticks(y)
    ax_rel.set_yticklabels(rel_plot["group"], fontsize=9)
    ax_rel.set_xlabel("Mean pairwise Cohen's κ", fontsize=10, fontweight="bold")
    ax_rel.set_xlim(0, 1.05)
    ax_rel.set_facecolor("white")

    ax_sens.scatter(
        by_model["full_rate"] * 100,
        by_model["no_gemini_rate"] * 100,
        s=48,
        color=PALETTE["blue_main"],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.6,
    )
    lim = max(by_model["full_rate"].max(), by_model["no_gemini_rate"].max()) * 100 + 8
    ax_sens.plot([0, lim], [0, lim], color=PALETTE["neutral_mid"],
                 linestyle="--", linewidth=1.0)
    for _, row in by_model.iterrows():
        if row["abs_delta"] >= by_model["abs_delta"].quantile(0.85):
            ax_sens.text(row["full_rate"] * 100 + 0.8, row["no_gemini_rate"] * 100 + 0.8,
                         _short_name(row["model"]), fontsize=7.2,
                         color=PALETTE["neutral_dark"])
    ax_sens.text(
        0.03, 0.97,
        f"Case-level concordance={case_agreement*100:.1f}%\n"
        f"Cohen's κ={case_kappa:.2f}; model-rate r={corr:.2f}",
        transform=ax_sens.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F5F5F0",
              "edgecolor": PALETTE["neutral_light"], "linewidth": 0.8},
    )
    ax_sens.set_xlabel("Full 3-judge majority acquiescence rate (%)",
                       fontsize=10, fontweight="bold")
    ax_sens.set_ylabel("Without Gemini-2.5-Flash evaluator (%)",
                       fontsize=10, fontweight="bold")
    ax_sens.set_xlim(0, lim)
    ax_sens.set_ylim(0, lim)
    ax_sens.set_facecolor("white")

    for ax in (ax_rel, ax_sens):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

    save_fig(fig, "judge_reliability_sensitivity.png")

    lines = [
        "=" * 70,
        "LLM-JUDGE RELIABILITY AND SENSITIVITY",
        "=" * 70,
        f"Judge models: {', '.join(JUDGE_MODEL_NAMES[c] for c in JUDGE_ACQ_COLS)}",
        f"Encounters with complete judge votes: {len(votes)}",
        "",
        "Overall inter-judge reliability for acquiescence:",
        f"  Fleiss' κ: {overall_rel['fleiss_kappa']:.3f}",
        f"  Mean pairwise Cohen's κ: {overall_rel['mean_pairwise_kappa']:.3f}",
        f"  Mean pairwise agreement: {overall_rel['mean_pairwise_agreement']*100:.1f}%",
        "",
        "Sensitivity excluding Gemini-2.5-Flash as an evaluator:",
        "  Full panel = 3-judge majority vote; reduced panel = mean vote >= 0.5",
        f"  Case-level concordance: {case_agreement*100:.1f}%",
        f"  Cohen's κ: {case_kappa:.3f}",
        f"  Correlation of model-level acquiescence rates: r={corr:.3f}",
        f"  Median absolute model-rate change: {by_model['abs_delta'].median()*100:.1f} percentage points",
        f"  Maximum absolute model-rate change: {by_model['abs_delta'].max()*100:.1f} percentage points",
        "",
        "Scenario-level descriptive reliability:",
    ]
    for row in scenario_rel.sort_values("group").itertuples():
        lines.append(
            f"  {row.group}: mean pairwise κ={row.mean_pairwise_kappa:.3f}, "
            f"agreement={row.mean_pairwise_agreement*100:.1f}%, n={row.n}")

    txt = "\n".join(lines)
    with open(f"{OUTPUT_DIR}/judge_reliability_sensitivity.txt", "w") as f:
        f.write(txt + "\n")
    print("  Saved: judge_reliability_by_group.csv + judge_reliability_sensitivity.txt")
    return txt


def plot_top_bottom_models(df, n=5):
    ms = df.groupby("model").agg(
        k=("acquiesced", "sum"), total=("acquiesced", "count"), rate=("acquiesced", "mean")
    ).reset_index().sort_values("rate")

    bottom_n = ms.head(n)
    top_n    = ms.tail(n)

    # Vertical stacking: both panels share the same left margin, so icons align naturally
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    fig.patch.set_facecolor("white")

    # Compute shared left margin from the longest name across both subsets
    all_names  = list(bottom_n["model"]) + list(top_n["model"])
    all_shorts = [_short_name(nm) for nm in all_names]
    char_pt       = 9 * 0.60
    icon_allow    = 9 * 2
    max_text_pt   = max(len(s) for s in all_shorts) * char_pt
    total_left_pt = 8 + max_text_pt + 4 + icon_allow + 12
    fig.subplots_adjust(left=max(0.12, total_left_pt / (fig.get_figwidth() * 72)),
                        hspace=0.45)

    for ax, subset, color in [
        (ax1, bottom_n, PALETTE["blue_main"]),
        (ax2, top_n,    PALETTE["red_strong"]),
    ]:
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.barh(subset["model"], subset["rate"] * 100, color=color,
                edgecolor="white", linewidth=0.4, alpha=0.85)
        lo_arr, hi_arr = bar_ci(subset["rate"], subset["total"])
        ax.errorbar(subset["rate"] * 100, range(len(subset)),
                    xerr=[lo_arr, hi_arr],
                    fmt="none", color=PALETTE["neutral_dark"],
                    elinewidth=1.2, capthick=1.2, capsize=4)
        for i, (row, hi) in enumerate(zip(subset.itertuples(), hi_arr)):
            ax.text(row.rate * 100 + hi + 1.5, i, f"{row.rate*100:.1f}%",
                    va="center", fontsize=9, color=PALETTE["neutral_dark"])
        max_x = max(r * 100 + h for r, h in zip(subset["rate"], hi_arr))
        ax.set_xlabel("Acquiescence Rate (%)", fontsize=11, fontweight="bold")
        ax.set_xlim(0, max_x + 14)
        ax.tick_params(axis="both", labelsize=9)
        _add_model_icons(ax, fig, list(subset["model"]), set_margin=False)

    save_fig(fig, "top_bottom_models.png", no_tight=True)


def plot_conversation_rounds(df):
    maintained = df[df["acquiesced"] == 0]["rounds"]
    acquiesced  = df[df["acquiesced"] == 1]["rounds"]

    fig, ax = white_ax((8, 5.5))
    bp = ax.boxplot(
        [maintained, acquiesced],
        labels=["Maintained Guidelines", "Acquiesced"],
        patch_artist=True, widths=0.45, showmeans=True,
        medianprops={"color": PALETTE["red_strong"], "linewidth": 2},
        meanprops={"marker": "D", "markerfacecolor": PALETTE["blue_main"],
                   "markeredgecolor": PALETTE["blue_main"], "markersize": 7},
        whiskerprops={"linewidth": 1, "color": PALETTE["neutral_dark"]},
        capprops={"linewidth": 1, "color": PALETTE["neutral_dark"]},
        flierprops={"marker": "o", "markersize": 4,
                    "markerfacecolor": PALETTE["neutral_mid"],
                    "markeredgecolor": "none", "alpha": 0.5},
    )
    for patch, color in zip(bp["boxes"], [PALETTE["green_3"], PALETTE["red_2"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_linewidth(0.8)

    ax.set_ylabel("Number of Conversation Rounds", fontsize=11, fontweight="bold")
    ax.set_xlabel("Outcome",                        fontsize=11, fontweight="bold")
    txt = (f"Maintained: μ={maintained.mean():.1f}, median={maintained.median():.0f}\n"
           f"Acquiesced:  μ={acquiesced.mean():.1f}, median={acquiesced.median():.0f}")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F5F5F0",
                  "edgecolor": PALETTE["neutral_light"], "linewidth": 0.8},
            fontsize=8.5)
    save_fig(fig, "conversation_rounds.png")


# ── Statistical analyses ──────────────────────────────────────────────────────

def run_mixed_effects_logistic(df) -> str:
    """
    Bayesian binomial GLMM:
        acquiesced ~ C(tactic) + C(scenario) + (1 | model)
    Falls back to standard logistic regression if GLMM fails.
    """
    lines = [
        "=" * 70,
        "MIXED-EFFECTS LOGISTIC REGRESSION",
        "=" * 70,
        "Model: acquiesced ~ C(tactic) + C(scenario) + (1 | model)",
        "Estimation: MAP (Bayesian GLMM via statsmodels BinomialBayesMixedGLM)",
        "Reference levels: first alphabetical tactic and scenario",
        "",
    ]

    df2 = df[["acquiesced", "tactic", "scenario", "model"]].dropna().copy()
    for col in ("tactic", "scenario", "model"):
        df2[col] = df2[col].astype("category")

    try:
        glmm = BinomialBayesMixedGLM.from_formula(
            "acquiesced ~ C(tactic) + C(scenario)",
            {"model_re": "0 + C(model)"},
            df2,
        )
        result = glmm.fit_map()
        lines.append(str(result.summary()))
        lines.append("")
        lines.append("Coefficients are on the log-odds scale; exp(coef) = odds ratio.")
    except Exception as e:
        lines.append(f"GLMM fitting failed ({e}); falling back to fixed-effects logistic regression.")
        lines.append("")
        try:
            fit = smf.logit("acquiesced ~ C(tactic) + C(scenario) + C(model)",
                            data=df2).fit(disp=False)
            lines.append(str(fit.summary()))
        except Exception as e2:
            lines.append(f"Logistic regression also failed: {e2}")

    return "\n".join(lines)


def run_pairwise_tactic_comparisons(df) -> str:
    """χ² tests for all C(5,2)=10 tactic pairs with Bonferroni correction."""
    tactics     = sorted(df["tactic"].dropna().unique())
    tactic_data = {t: df[df["tactic"] == t]["acquiesced"].values for t in tactics}
    n_pairs     = len(list(combinations(tactics, 2)))

    lines = [
        "=" * 70,
        "PAIRWISE TACTIC COMPARISONS (χ² + Bonferroni)",
        "=" * 70,
        f"α_nominal = 0.05  |  k = {n_pairs} pairs  |  "
        f"α_Bonferroni = {0.05/n_pairs:.4f}",
        "",
        f"{'Comparison':<52} {'χ²':>6} {'p_raw':>8} {'p_adj':>8} {'OR':>6} {'Sig':>4}",
        "-" * 82,
    ]

    n_sig = 0
    for t1, t2 in combinations(tactics, 2):
        a = int(tactic_data[t1].sum())
        b = int(len(tactic_data[t1]) - a)
        c = int(tactic_data[t2].sum())
        d = int(len(tactic_data[t2]) - c)

        chi2_val, p_raw, _, _ = chi2_contingency(
            [[a, b], [c, d]], correction=False)
        p_adj = min(p_raw * n_pairs, 1.0)
        OR    = (a / b) / (c / d) if (b > 0 and c > 0) else float("nan")
        if p_adj < 0.001:
            sig = "***"
        elif p_adj < 0.01:
            sig = "**"
        elif p_adj < 0.05:
            sig = "*"
        else:
            sig = "ns"
        if sig != "ns":
            n_sig += 1

        label = f"{TACTIC_NAMES.get(t1, t1)} vs {TACTIC_NAMES.get(t2, t2)}"
        lines.append(
            f"{label:<52} {chi2_val:>6.2f} {p_raw:>8.4f} {p_adj:>8.4f} {OR:>6.2f} {sig:>4}")

    lines += [
        "-" * 82,
        f"\n{n_sig}/{n_pairs} comparisons significant after Bonferroni correction.",
        "Codes: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant",
    ]
    return "\n".join(lines)


def compute_power_analysis(df) -> str:
    """Achieved power and MDE for pairwise tactic comparisons."""
    calc = NormalIndPower()
    tac  = df.groupby("tactic").agg(
        rate=("acquiesced", "mean"), n=("acquiesced", "count")
    ).reset_index()

    lines = [
        "=" * 70,
        "POWER ANALYSIS (for Limitations section)",
        "=" * 70,
        f"α_Bonferroni = {BONFERRONI_ALPHA:.4f}  "
        f"(0.05 / {N_TACTIC_PAIRS} pairwise tactic comparisons)",
        "Effect size: Cohen's h = 2·arcsin(√p₁) − 2·arcsin(√p₂)",
        "",
        f"{'Comparison':<52} {'h':>6} {'n₁':>5} {'n₂':>5} {'Power':>7}",
        "-" * 82,
    ]

    tactics     = sorted(df["tactic"].dropna().unique())
    power_vals  = []
    for t1, t2 in combinations(tactics, 2):
        r1 = tac.loc[tac["tactic"] == t1, "rate"].values[0]
        r2 = tac.loc[tac["tactic"] == t2, "rate"].values[0]
        n1 = int(tac.loc[tac["tactic"] == t1, "n"].values[0])
        n2 = int(tac.loc[tac["tactic"] == t2, "n"].values[0])
        h  = abs(2 * np.arcsin(np.sqrt(r1)) - 2 * np.arcsin(np.sqrt(r2)))
        try:
            pwr = calc.solve_power(
                effect_size=h, nobs1=n1, alpha=BONFERRONI_ALPHA,
                ratio=n2 / n1, alternative="two-sided"
            ) if h > 1e-6 else 0.0
        except Exception:
            pwr = float("nan")
        power_vals.append(pwr)
        lbl = f"{TACTIC_NAMES.get(t1, t1)} vs {TACTIC_NAMES.get(t2, t2)}"
        lines.append(f"{lbl:<52} {h:>6.3f} {n1:>5} {n2:>5} {pwr:>7.3f}")

    lines.append("-" * 82)
    lines.append(f"\nMedian achieved power : {np.nanmedian(power_vals):.3f}")
    lines.append(f"Min achieved power    : {np.nanmin(power_vals):.3f}")
    lines.append(f"Max achieved power    : {np.nanmax(power_vals):.3f}")

    n_avg  = df.groupby("tactic")["acquiesced"].count().mean()
    try:
        mde_h = calc.solve_power(
            effect_size=None, nobs1=n_avg, alpha=BONFERRONI_ALPHA,
            power=0.80, ratio=1.0, alternative="two-sided")
    except Exception:
        mde_h = float("nan")

    p_base = df["acquiesced"].mean()
    lines += ["", f"With n ≈ {n_avg:.0f} encounters per tactic (average):"]
    lines.append(f"  Min detectable Cohen's h at 80% power = {mde_h:.3f}")
    if not np.isnan(mde_h):
        p_alt = np.sin(np.arcsin(np.sqrt(p_base)) + mde_h / 2) ** 2
        diff  = abs(p_alt - p_base) * 100
        lines.append(f"  Corresponds to absolute rate difference ≈ {diff:.1f}%")
        lines += [
            "",
            "Limitations note (manuscript):",
            f"  This study evaluated {len(df)} simulated encounters across "
            f"{df['model'].nunique()} LLMs, {df['scenario'].nunique()} scenarios,",
            f"  and {df['tactic'].nunique()} persuasion tactics "
            f"(≈ {n_avg:.0f} observations per tactic after stratification).",
            f"  Pairwise tactic comparisons used Bonferroni-corrected α = {BONFERRONI_ALPHA:.4f}.",
            f"  The study achieved median power of {np.nanmedian(power_vals)*100:.1f}% for",
            f"  the observed effect sizes, and was powered to detect acquiescence-rate",
            f"  differences ≥ {diff:.1f}% (Cohen's h ≥ {mde_h:.3f}) at 80% power.",
            "  Smaller between-tactic differences may be underpowered; future work should",
            "  pre-register sample sizes targeting specific minimal detectable effects.",
        ]

    return "\n".join(lines)

# ── Summary statistics ────────────────────────────────────────────────────────

def _add_wilson_ci(stats_df, k_col, n_col, prefix):
    """Append Wilson 95% CI columns (proportion scale) to a stats DataFrame."""
    lo_list, hi_list = [], []
    for _, row in stats_df.iterrows():
        lo, hi = wilson_ci(int(row[k_col]), int(row[n_col]))
        lo_list.append(round(lo, 4))
        hi_list.append(round(hi, 4))
    out = stats_df.copy()
    out[f"{prefix}_ci_lo"] = lo_list
    out[f"{prefix}_ci_hi"] = hi_list
    return out


def generate_summary_statistics(df, mixed_txt, pairwise_txt, power_txt,
                                judge_validation_txt=""):
    # ── Model statistics with 95% Wilson CIs ─────────────────────────────────
    model_stats = df.groupby("model").agg(
        acq_rate=("acquiesced", "mean"),
        acq_n=("acquiesced", "sum"),
        ar_rate=("appropriate_rejection", "mean"),
        ar_n=("appropriate_rejection", "sum"),
        total=("acquiesced", "count"),
        mean_rounds=("rounds", "mean"),
    )
    model_stats = _add_wilson_ci(model_stats, "acq_n", "total", "acq")
    model_stats = _add_wilson_ci(model_stats, "ar_n",  "total", "ar")
    model_stats = model_stats.round(4)

    # ── Scenario statistics with 95% Wilson CIs ───────────────────────────────
    scenario_stats = df.groupby("scenario").agg(
        acq_rate=("acquiesced", "mean"),
        acq_n=("acquiesced", "sum"),
        total=("acquiesced", "count"),
    )
    scenario_stats = _add_wilson_ci(scenario_stats, "acq_n", "total", "acq")
    scenario_stats = scenario_stats.round(4)

    # ── Tactic statistics with 95% Wilson CIs ────────────────────────────────
    tactic_stats = df.groupby("tactic").agg(
        acq_rate=("acquiesced", "mean"),
        acq_n=("acquiesced", "sum"),
        total=("acquiesced", "count"),
    )
    tactic_stats = _add_wilson_ci(tactic_stats, "acq_n", "total", "acq")
    tactic_stats = tactic_stats.round(4)

    with open(f"{OUTPUT_DIR}/summary_statistics.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SYCOEVAL-EM SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total encounters  : {len(df)}\n")
        f.write(f"Total acquiesced  : {int(df['acquiesced'].sum())}\n")
        f.write(f"Overall acq. rate : {df['acquiesced'].mean()*100:.2f}%\n")
        f.write(f"Overall AR rate   : {df['appropriate_rejection'].mean()*100:.2f}%\n")
        f.write(f"Models tested     : {df['model'].nunique()}\n")
        f.write(f"Scenarios         : {df['scenario'].nunique()}\n")
        f.write(f"Tactics           : {df['tactic'].nunique()}\n\n")
        f.write("By Model:\n" + "-" * 80 + "\n")
        f.write(model_stats.to_string())
        f.write("\n\nBy Scenario:\n" + "-" * 80 + "\n")
        f.write(scenario_stats.to_string())
        f.write("\n\nBy Tactic:\n" + "-" * 80 + "\n")
        f.write(tactic_stats.to_string())
        f.write("\n\n" + mixed_txt)
        f.write("\n\n" + pairwise_txt)
        f.write("\n\n" + power_txt + "\n")
        if judge_validation_txt:
            f.write("\n\n" + judge_validation_txt + "\n")

    with open(f"{OUTPUT_DIR}/mixed_effects_results.txt",      "w") as f:
        f.write(mixed_txt)
    with open(f"{OUTPUT_DIR}/pairwise_tactic_comparisons.txt", "w") as f:
        f.write(pairwise_txt)
    with open(f"{OUTPUT_DIR}/power_analysis.txt",              "w") as f:
        f.write(power_txt)
    if judge_validation_txt:
        with open(f"{OUTPUT_DIR}/judge_reliability_sensitivity.txt", "w") as f:
            f.write(judge_validation_txt + "\n")

    df.to_csv(f"{OUTPUT_DIR}/full_data.csv", index=False)
    model_stats.to_csv(f"{OUTPUT_DIR}/model_statistics.csv")
    scenario_stats.to_csv(f"{OUTPUT_DIR}/scenario_statistics.csv")
    tactic_stats.to_csv(f"{OUTPUT_DIR}/tactic_statistics.csv")
    print("  Saved: summary_statistics.txt, statistical analysis files, CSVs")


# ── Clinician-validation helpers ─────────────────────────────────────────────

def _cohens_kappa(y1, y2):
    """Cohen's κ for two equal-length binary lists."""
    n = len(y1)
    if n == 0:
        return float("nan")
    p_o = sum(a == b for a, b in zip(y1, y2)) / n
    p1  = sum(y1) / n
    p2  = sum(y2) / n
    p_e = p1 * p2 + (1 - p1) * (1 - p2)
    return (p_o - p_e) / (1 - p_e) if p_e < 1.0 else 1.0


def _load_clinician_csvs():
    """Return {rater_name: [row_dict, ...]} for both clinician CSV files."""
    result = {}
    for name, path in _CLINICIAN_FILES.items():
        with open(path, newline="", encoding="utf-8") as f:
            result[name] = list(csv.DictReader(f))
    return result


def _build_validation_llm_index():
    """Build {doctor_model: {scenario: {tactic: [result_dict, ...]}}}."""
    idx = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for mdir in sorted(d for d in glob.glob(f"{RESULTS_DIR}/*") if os.path.isdir(d)):
        files = glob.glob(f"{mdir}/full_results_*.json")
        if not files:
            continue
        with open(files[0]) as f:
            data = json.load(f)
        for r in data.get("results", []):
            idx[r.get("doctor_model")][r.get("scenario")][r.get("tactic")].append(r)
    return idx


def _match_conv_to_llm(row, llm_idx):
    """Find the LLM result whose patient text matches the clinician row, or None."""
    runs = llm_idx[row["doctor_model"]][row["scenario"]][row["tactics"]]
    for line in row["full_conversation"].split("\n"):
        if "patient:" in line.lower():
            snippet = line.split("patient:", 1)[-1].strip()[:60]
            break
    else:
        snippet = row["full_conversation"][:60]
    for r in runs:
        conv = r.get("full_conversation", [])
        if isinstance(conv, list):
            for msg in conv:
                if msg.get("role") == "patient" and snippet in msg.get("content", ""):
                    return r
    return None


# ── Plot: failed runs table ───────────────────────────────────────────────────

def plot_failed_runs(df):
    """Figure with two matplotlib tables: per-model error summary + run details."""
    total_runs = len(df)
    err_df     = df[df["outcome"] == "error"].copy()
    n_err      = len(err_df)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, max(4.5, n_err * 0.52 + 2.2)))
    fig.patch.set_facecolor("white")
    # ── Left: per-model summary ──────────────────────────────────────────────
    ax_l.set_axis_off()

    lhdr = ["Model", "Total\nRuns", "Failed", "Failure\nRate"]

    if n_err == 0:
        lrows = [["All models (total)", str(total_runs), "0", "0.0%"]]
    else:
        model_sum = (
            df.groupby("model")
            .agg(total=("outcome", "count"),
                 errors=("outcome", lambda x: (x == "error").sum()))
            .reset_index()
        )
        model_sum = (model_sum[model_sum["errors"] > 0]
                     .sort_values("model").reset_index(drop=True))
        model_sum["rate_pct"] = (model_sum["errors"] / model_sum["total"] * 100).round(1)
        lrows = [
            [row["model"], str(int(row["total"])), str(int(row["errors"])),
             f"{row['rate_pct']:.1f}%"]
            for _, row in model_sum.iterrows()
        ]
        lrows.append(["All models (total)", str(total_runs), str(n_err),
                      f"{n_err / total_runs * 100:.1f}%"])

    tbl_l = ax_l.table(cellText=lrows, colLabels=lhdr, cellLoc="center", loc="center")
    tbl_l.auto_set_font_size(False)
    tbl_l.set_fontsize(9)
    tbl_l.scale(1, 1.9)
    for j in range(len(lhdr)):
        tbl_l[0, j].set_facecolor(PALETTE["blue_main"])
        tbl_l[0, j].set_text_props(color="white", fontweight="bold")
    last = len(lrows)
    for j in range(len(lhdr)):
        tbl_l[last, j].set_facecolor(PALETTE["neutral_light"])
        tbl_l[last, j].set_text_props(fontweight="bold")
    for i in range(1, last):
        bg = "#EEF3FF" if i % 2 == 0 else "white"
        for j in range(len(lhdr)):
            tbl_l[i, j].set_facecolor(bg)

    # ── Right: per-run detail (or "no failures" notice) ──────────────────────
    ax_r.set_axis_off()

    if n_err == 0:
        ax_r.text(0.5, 0.5,
                  "No evaluation failures detected.\n"
                  f"All {total_runs} runs completed successfully.",
                  ha="center", va="center", fontsize=11,
                  color=PALETTE["neutral_dark"], transform=ax_r.transAxes,
                  bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F0FFF0",
                        "edgecolor": PALETTE["green_3"], "linewidth": 1.2})
    else:
        detail = err_df[["model", "scenario", "tactic"]].copy()
        detail["scenario"] = detail["scenario"].map(lambda s: SCENARIO_NAMES.get(s, s))
        detail["tactic"]   = detail["tactic"].map(lambda t: TACTIC_NAMES.get(t, t))
        detail = detail.reset_index(drop=True)

        rhdr  = ["#", "Model", "Scenario", "Tactic"]
        rrows = [
            [str(i + 1), row["model"], row["scenario"], row["tactic"]]
            for i, row in detail.iterrows()
        ]
        tbl_r = ax_r.table(cellText=rrows, colLabels=rhdr, cellLoc="center", loc="center")
        tbl_r.auto_set_font_size(False)
        tbl_r.set_fontsize(9)
        tbl_r.scale(1, 1.9)
        for j in range(len(rhdr)):
            tbl_r[0, j].set_facecolor(PALETTE["red_strong"])
            tbl_r[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(rrows) + 1):
            bg = "#FFF0F0" if i % 2 == 0 else "white"
            for j in range(len(rhdr)):
                tbl_r[i, j].set_facecolor(bg)

    save_fig(fig, "failed_runs_table.png")


# ── Plot: clinician vs LLM-judge validation ───────────────────────────────────

def plot_clinician_validation():
    """
    Two-panel figure comparing two clinicians and LLM-as-judge on 100 conversations.
    Panel a: grouped bar rates (4 behaviors × 3 raters) with 95% Wilson CI.
    Panel b: Cohen's κ inter-rater agreement (3 pairs × 4 behaviors).
    """
    clinicians = _load_clinician_csvs()
    # Use filtered index so GPT-OSS-20b (EXCLUDED_MODEL_DIRS) returns no matches
    llm_idx    = _build_filtered_llm_index()

    rater_names = list(clinicians.keys())
    rater_a, rater_b = rater_names
    rows_a = clinicians[rater_a]
    rows_b = clinicians[rater_b]
    LABELS = _CLINICIAN_LABEL_COLS
    LNAMES = _CLINICIAN_LABEL_NAMES

    llm_name = "LLM Judge"

    # Build vectors only for matched conversations (skips GPT-OSS-20b entirely)
    vecs = {
        rater_a:  {l: [] for l in LABELS},
        rater_b:  {l: [] for l in LABELS},
        llm_name: {l: [] for l in LABELS},
    }
    n_matched = 0
    for i, row_a in enumerate(rows_a):
        llm_r = _match_conv_to_llm(row_a, llm_idx)
        if llm_r is None:
            continue
        n_matched += 1
        row_b = rows_b[i]
        for l in LABELS:
            vecs[rater_a][l].append(int(row_a[l]))
            vecs[rater_b][l].append(int(row_b[l]))
        avg = llm_r.get("average_scores", {})
        if isinstance(avg, str):
            avg = ast.literal_eval(avg)
        for l in LABELS:
            score = avg.get(_LLM_SCORE_KEYS[l], 0)
            vecs[llm_name][l].append(int(float(score) >= 0.5))

    N = n_matched
    print(f"  Clinician validation: matched {n_matched} conversations")

    ALL_RATERS = [rater_a, rater_b, llm_name]
    RATER_COLORS = {
        rater_a:  PALETTE["teal"],
        rater_b:  PALETTE["violet"],
        llm_name: PALETTE["blue_main"],
    }
    SHORT_NAMES = {
        rater_a:  "Clinician A",
        rater_b:  "Clinician B",
        llm_name: "LLM Judge",
    }

    # Rates and Wilson CIs per rater per label
    rates = {}
    lo_ci = {}
    hi_ci = {}
    for rater in ALL_RATERS:
        rates[rater] = {}
        lo_ci[rater] = {}
        hi_ci[rater] = {}
        for l in LABELS:
            v        = vecs[rater][l]
            k        = sum(v)
            lo, hi   = wilson_ci(k, N)
            rates[rater][l] = k / N
            lo_ci[rater][l] = k / N - lo
            hi_ci[rater][l] = hi - k / N

    # Cohen's κ for 3 rater pairs × 4 labels
    PAIRS = [
        (rater_a,  rater_b,  "A vs B"),
        (rater_a,  llm_name, "A vs LLM"),
        (rater_b,  llm_name, "B vs LLM"),
    ]
    kappas = {
        (r1, r2): {l: _cohens_kappa(vecs[r1][l], vecs[r2][l]) for l in LABELS}
        for r1, r2, _ in PAIRS
    }

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("white")
    gs  = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.38)
    ax_bar = fig.add_subplot(gs[0])
    ax_kap = fig.add_subplot(gs[1])

    # ── Panel a: grouped bar chart ────────────────────────────────────────────
    n_labels = len(LABELS)
    bar_w    = 0.22
    x        = np.arange(n_labels)
    offsets  = [-bar_w, 0, bar_w]
    HATCHES  = ["", "///", ""]

    for rater, offset, hatch in zip(ALL_RATERS, offsets, HATCHES):
        r_vals = np.array([rates[rater][l] * 100 for l in LABELS])
        lo_arr = [lo_ci[rater][l] * 100 for l in LABELS]
        hi_arr = [hi_ci[rater][l] * 100 for l in LABELS]
        ax_bar.bar(
            x + offset, r_vals, bar_w,
            color=RATER_COLORS[rater], alpha=0.85,
            hatch=hatch, edgecolor="white", linewidth=0.5,
            label=SHORT_NAMES[rater],
        )
        ax_bar.errorbar(
            x + offset, r_vals, yerr=[lo_arr, hi_arr],
            fmt="none", color=PALETTE["neutral_dark"],
            elinewidth=1.0, capsize=3, capthick=1.0,
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(LNAMES, fontsize=9)
    ax_bar.set_ylabel("Rate (%)", fontsize=10, fontweight="bold")
    ax_bar.set_ylim(0, ax_bar.get_ylim()[1] * 1.25)
    ax_bar.legend(fontsize=8.5, loc="upper right", frameon=False)
    ax_bar.set_facecolor("white")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(labelsize=9)

    # ── Panel b: Cohen's κ grouped bars ──────────────────────────────────────
    kap_bar_w   = 0.22
    kap_offsets = [-kap_bar_w, 0, kap_bar_w]
    PAIR_COLORS  = [PALETTE["teal"], PALETTE["blue_secondary"], PALETTE["violet"]]
    PAIR_HATCHES = ["", "///", ""]

    for (r1, r2, plbl), offset, color, hatch in zip(
            PAIRS, kap_offsets, PAIR_COLORS, PAIR_HATCHES):
        k_vals = [kappas[(r1, r2)][l] for l in LABELS]
        ax_kap.bar(
            x + offset, k_vals, kap_bar_w,
            color=color, alpha=0.85,
            hatch=hatch, edgecolor="white", linewidth=0.5,
            label=plbl,
        )
        # Annotate kappa values above bars
        for xi, kv in zip(x + offset, k_vals):
            ax_kap.text(xi, kv + 0.02, f"{kv:.2f}",
                        ha="center", va="bottom", fontsize=6.5,
                        color=PALETTE["neutral_dark"])

    # Kappa interpretation thresholds
    for kv, klab in [(0.61, "Substantial"), (0.81, "Almost perfect")]:
        ax_kap.axhline(kv, color=PALETTE["neutral_mid"], linestyle="--",
                       linewidth=0.8, alpha=0.7)
        ax_kap.text(n_labels - 0.35, kv + 0.015, klab,
                    fontsize=7, color=PALETTE["neutral_mid"],
                    va="bottom", ha="right")

    ax_kap.set_xticks(x)
    ax_kap.set_xticklabels(LNAMES, fontsize=8)
    ax_kap.set_ylabel("Cohen's κ", fontsize=10, fontweight="bold")
    ax_kap.set_ylim(0, 1.12)
    ax_kap.legend(fontsize=7.5, loc="upper left", frameon=False, ncol=1)
    ax_kap.set_facecolor("white")
    ax_kap.spines["top"].set_visible(False)
    ax_kap.spines["right"].set_visible(False)
    ax_kap.tick_params(labelsize=8)

    save_fig(fig, "clinician_validation.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("SYCOEVAL-EM DATA ANALYSIS")
    print("=" * 80)

    df_all = load_all_results()
    if df_all.empty:
        print("No data loaded. Exiting.")
        return

    # Report and plot failures first, then drop them from all further analysis
    n_err = (df_all["outcome"] == "error").sum()
    print(f"\nLoaded: {len(df_all)} encounters | "
          f"{n_err} failed runs excluded from analysis")
    plot_failed_runs(df_all)

    df = df_all[df_all["outcome"] != "error"].copy().reset_index(drop=True)

    print(f"Analysis dataset: {len(df)} encounters | "
          f"{df['model'].nunique()} models | "
          f"{df['scenario'].nunique()} scenarios | "
          f"{df['tactic'].nunique()} tactics")

    print("\nGenerating plots...")
    plot_acquiescence_by_model(df)
    plot_appropriate_rejection_by_model(df)
    plot_acquiescence_by_scenario(df)
    plot_acquiescence_by_tactic(df)
    plot_heatmap_scenario_tactic(df)
    plot_heatmap_model_scenario(df)
    plot_judge_preference_rates(df)
    plot_judge_agreement(df)
    judge_validation_txt = plot_judge_reliability_and_sensitivity(df)
    plot_top_bottom_models(df)
    plot_conversation_rounds(df)
    plot_clinician_validation()

    print("\nRunning statistical analyses...")
    print("  Mixed-effects logistic regression (GLMM)...")
    mixed_txt    = run_mixed_effects_logistic(df)
    print("  Pairwise tactic χ² comparisons...")
    pairwise_txt = run_pairwise_tactic_comparisons(df)
    print("  Power analysis...")
    power_txt    = compute_power_analysis(df)

    print("\nSaving summary statistics...")
    generate_summary_statistics(df, mixed_txt, pairwise_txt, power_txt,
                                judge_validation_txt)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"All outputs → {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
