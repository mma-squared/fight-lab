"""
Standard UFC fighter comparison metrics and visualizations.

Extracts analysis logic from notebooks into reusable functions with proper types.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

# Logo path for chart watermarks (transparent PNG recommended)
_LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "MMASquaredLogo.png"

# Type alias for the data dict returned by UFCStatsClient.get_all_data_for_fighters
FighterData = dict[str, pd.DataFrame]

# Distinct colors for two-fighter comparisons (avoid similar blues)
FIGHTER1_COLOR = "#2563eb"
FIGHTER2_COLOR = "#ea580c"
# Margins for chart breathing room
CHART_MARGIN = dict(t=120, b=90, l=80, r=60)


def _add_logo_watermark(fig: go.Figure, opacity: float = 0.12) -> go.Figure:
    """Add transparent MMA Squared logo as background watermark on a Plotly figure."""
    if not _LOGO_PATH.exists():
        return fig
    try:
        from PIL import Image
        img = Image.open(_LOGO_PATH)
        source = img
    except Exception:
        source = str(_LOGO_PATH)
    fig.add_layout_image(
        dict(
            source=source,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            sizex=0.5,
            sizey=0.5,
            xanchor="center",
            yanchor="middle",
            layer="below",
            opacity=opacity,
        )
    )
    return fig
# Distinct colors for head/body/leg (avoid similar blues)
TARGET_COLORS = {"Head": "#1e40af", "Body": "#dc2626", "Leg": "#16a34a"}

__all__ = [
    "FighterData",
    "sort_fights_chronologically",
    "sort_fights_chrono",
    "cumulative_wins",
    "get_streak",
    "normalize_method",
    "record_comparison_figure",
    "career_stats_comparison_figure",
    "cumulative_wins_figure",
    "streak_and_last_5",
    "head_to_head_advantages",
    "strikes_absorbed_by_target_figures",
    "strikes_by_position_figures",
    "finishes_by_round_method_figures",
    "takedowns_per_fight_figure",
    "takedowns_absorbed_per_round_figures",
    "common_opponents_record_table",
    "common_opponents_performance_figure",
    "common_opponents_scatter_figure",
    "common_opponents_striking_figure",
    "common_opponents_strike_accuracy_figure",
    "common_opponents_strike_share_figure",
    "common_opponents_takedown_figure",
    "common_opponents_strike_share_by_target_figure",
]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def sort_fights_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Sort fights by event_date, earliest first."""
    d = df.copy()
    d["_date"] = pd.to_datetime(d["event_date"], errors="coerce")
    return d.sort_values("_date").drop(columns=["_date"], errors="ignore")


def sort_fights_chrono(df: pd.DataFrame) -> pd.DataFrame:
    """Sort fights by event_date, oldest first (chronological order)."""
    d = df.copy()
    d["_date"] = pd.to_datetime(d["event_date"], errors="coerce")
    return d.sort_values("_date", ascending=True)


def cumulative_wins(df: pd.DataFrame) -> pd.Series:
    """Compute cumulative wins/losses: win=+1, loss=-1."""
    pts = df["result"].map({"win": 1, "loss": -1}).fillna(0)
    return pts.cumsum()


def get_streak(results_chronological: list[Any]) -> int:
    """
    Compute streak from fight results (oldest→newest).
    Returns positive for win streak, negative for loss streak.
    Skips nc/draw.
    """
    streak = 0
    for r in results_chronological:
        r = str(r).strip().lower() if pd.notna(r) else ""
        if r in ("win", "w"):
            streak = streak + 1 if streak >= 0 else 1
        elif r in ("loss", "l"):
            streak = streak - 1 if streak <= 0 else -1
        elif r in ("nc", "no contest", "draw", "d"):
            continue
        else:
            break
    return streak


def normalize_method(m: Any) -> Any:
    """Normalize method string for grouping (Decision, KO/TKO, SUB: type)."""
    if pd.isna(m):
        return m
    m = str(m).strip()
    if any(x in m.upper() for x in ("DEC", "DRAW", "MAJORITY", "SPLIT", "UNANIMOUS")) or m.endswith("-DEC"):
        return "Decision"
    if m.upper().startswith("KO") or m.upper().startswith("TKO"):
        return "KO/TKO"
    if m.upper().startswith("SUB"):
        rest = m[3:].strip().replace("  ", " ")
        return "SUB: " + rest if rest else "Submission"
    return m


def _jitter(seed: Any, scale: float = 0.08) -> float:
    """Deterministic jitter for scatter plot overlap."""
    h = int(hashlib.md5(str(seed).encode()).hexdigest()[:8], 16)
    return (h % 100) / 100 * 2 * scale - scale


def _axis_range_with_padding(
    max_positive: float,
    max_negative: float,
    *,
    pct: float = 0.4,
) -> tuple[float, float]:
    """
    Return (y_min, y_max) with proportional padding for textposition="outside" labels.
    Uses 40% padding so labels fit; percentage-only keeps scaling consistent.
    """
    pad_top = max_positive * pct
    pad_bot = max_negative * pct
    return (-max_negative - pad_bot, max_positive + pad_top)


def _axis_range_positive_only(max_val: float, *, pct: float = 0.25) -> tuple[float, float]:
    """Return (0, y_max) for charts that start at zero. Adds padding at top for labels."""
    return (0, max_val * (1 + pct))


# -----------------------------------------------------------------------------
# Figure / visualization functions
# -----------------------------------------------------------------------------


def record_comparison_figure(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> go.Figure:
    """Stacked bar chart of wins, losses, draws for both fighters."""
    f1 = data["fighter1_info"].iloc[0]
    f2 = data["fighter2_info"].iloc[0]
    rec_df = pd.DataFrame({
        "Fighter": [fighter1, fighter2],
        "Wins": [f1["wins"], f2["wins"]],
        "Losses": [f1["losses"], f2["losses"]],
        "Draws": [f1["draws"], f2["draws"]],
    })
    fig = go.Figure()
    for cat in ["Wins", "Losses", "Draws"]:
        fig.add_trace(go.Bar(
            name=cat,
            x=rec_df["Fighter"],
            y=rec_df[cat],
            text=rec_df[cat],
            textposition="outside",
            textfont={"size": 14},
        ))
    max_val = (rec_df["Wins"] + rec_df["Losses"] + rec_df["Draws"]).max()
    y_min, y_max = _axis_range_positive_only(max_val)
    fig.update_layout(
        barmode="stack",
        title="Career record",
        yaxis_title="Count",
        legend_title="Result",
        margin=CHART_MARGIN,
        yaxis_range=[y_min, y_max],
    )
    return _add_logo_watermark(fig)


def career_stats_comparison_figure(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> go.Figure | None:
    """Grouped bar chart of career stats (slpm, str_acc, etc.). Returns None if no stats."""
    compare_df = data["compare"]
    stat_labels = [
        "Significant Strikes Landed per Min",
        "Striking Accuracy (%)",
        "Significant Strikes Absorbed per Min",
        "Strike Defense (%)",
        "Takedowns per 15 Min",
        "Takedown Accuracy (%)",
        "Takedown Defense (%)",
        "Submission Attempts per 15 Min",
    ]
    stats_df = compare_df[compare_df["Stat"].isin(stat_labels)].dropna(
        subset=[fighter1, fighter2], how="all"
    )
    if stats_df.empty:
        return None
    fig = go.Figure()
    y1 = stats_df[fighter1].astype(float)
    y2 = stats_df[fighter2].astype(float)
    fig.add_trace(go.Bar(
        name=fighter1, x=stats_df["Stat"], y=y1,
        text=y1.round(2), textposition="outside",
        marker_color=FIGHTER1_COLOR,
    ))
    fig.add_trace(go.Bar(
        name=fighter2, x=stats_df["Stat"], y=y2,
        text=y2.round(2), textposition="outside",
        marker_color=FIGHTER2_COLOR,
    ))
    max_val = max(y1.max(), y2.max())
    y_min, y_max = _axis_range_positive_only(max_val)
    fig.update_layout(
        barmode="group",
        title="Career stats comparison",
        xaxis_tickangle=-45,
        legend_title="Fighter",
        margin=CHART_MARGIN,
        yaxis_range=[y_min, y_max],
    )
    return _add_logo_watermark(fig)


def cumulative_wins_figure(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> tuple[pd.DataFrame, pd.DataFrame, go.Figure]:
    """
    Cumulative W-L trend (win=+1, loss=-1) over fight order.
    Returns (df1_with_cum, df2_with_cum, fig).
    """
    df1 = sort_fights_chronologically(data["fighter1_fights"]).reset_index(drop=True)
    df2 = sort_fights_chronologically(data["fighter2_fights"]).reset_index(drop=True)
    df1["cum_wins"] = cumulative_wins(df1)
    df2["cum_wins"] = cumulative_wins(df2)
    df1["fight_num"] = range(1, len(df1) + 1)
    df2["fight_num"] = range(1, len(df2) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df1["fight_num"], y=df1["cum_wins"],
        mode="lines+markers", name=fighter1,
        text=df1["opponent"],
        hovertemplate="Fight %{x}: vs %{text}<br>Cumulative: %{y}<extra></extra>",
        marker_color=FIGHTER1_COLOR,
    ))
    fig.add_trace(go.Scatter(
        x=df2["fight_num"], y=df2["cum_wins"],
        mode="lines+markers", name=fighter2,
        text=df2["opponent"],
        hovertemplate="Fight %{x}: vs %{text}<br>Cumulative: %{y}<extra></extra>",
        marker_color=FIGHTER2_COLOR,
    ))
    fig.update_layout(
        title="Cumulative W-L trend (win=+1, loss=-1)",
        xaxis_title="Fight number",
        yaxis_title="Cumulative",
        hovermode="x unified",
        margin=CHART_MARGIN,
    )
    fig.add_hline(y=0, line_dash="dash", opacity=0.5)
    return df1, df2, _add_logo_watermark(fig)


def streak_and_last_5(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> dict[str, dict[str, Any]]:
    """
    Current streak and last 5 fights (most recent first) for each fighter.
    Returns dict: {fighter1: {streak: int, last_5: pd.DataFrame}, fighter2: {...}}
    """
    result: dict[str, dict[str, Any]] = {}
    for name, fights_df in [(fighter1, data["fighter1_fights"]), (fighter2, data["fighter2_fights"])]:
        ordered = sort_fights_chrono(fights_df)
        last_5 = ordered.tail(5)
        results = last_5["result"].tolist()
        streak = get_streak(results)
        result[name] = {
            "streak": streak,
            "last_5": last_5.iloc[::-1][["opponent", "result", "method", "event_date"]],
        }
    return result


def head_to_head_advantages(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> dict[str, str | None]:
    """
    Where each fighter holds the edge (from compare data).
    Returns dict: {fighter1: "Stat1, Stat2, ...", fighter2: "..."}.
    """
    adv = data["compare"][data["compare"]["Stat"] == "Advantages"]
    out: dict[str, str | None] = {fighter1: None, fighter2: None}
    if adv.empty:
        return out
    for col in [fighter1, fighter2]:
        val = adv[col].iloc[0]
        if pd.notna(val) and str(val).strip():
            out[col] = str(val).strip()
    return out


def strikes_absorbed_by_target_figures(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> list[go.Figure]:
    """
    Strikes absorbed (head/body/leg) per-round avg with opponent lands comparison.
    Returns list of figures (one per fighter + combined scatter).
    """
    head_col = "opp_sig_str_head_success"
    body_col = "opp_sig_str_body_success"
    leg_col = "opp_sig_str_leg_success"
    landed_cols = [f"my_sig_str_{t}_success" for t in ["head", "body", "leg"]]
    cols = [head_col, body_col, leg_col]

    def has_cols(df: pd.DataFrame) -> bool:
        return all(c in df.columns for c in cols)

    f1_fights = data["fighter1_fights"]
    f2_fights = data["fighter2_fights"]

    if not (has_cols(f1_fights) and has_cols(f2_fights) and "rounds" in f1_fights.columns):
        return []

    figures: list[go.Figure] = []
    for name, df, other_name, other_df in [
        (fighter1, f1_fights, fighter2, f2_fights),
        (fighter2, f2_fights, fighter1, f1_fights),
    ]:
        tot_rounds = df["rounds"].sum()
        if tot_rounds == 0:
            continue
        abs_sums = df[cols].fillna(0).sum()
        abs_per_rd = [abs_sums[c] / tot_rounds for c in cols]
        tot_abs = sum(abs_sums)
        pcts = [100 * v / tot_abs if tot_abs else 0 for v in abs_sums]
        labels_pos = [f"{abs_per_rd[i]:.1f}/rd ({pcts[i]:.0f}%)" for i in range(3)]
        other_tot_rounds = other_df["rounds"].sum()
        if other_tot_rounds and all(c in other_df.columns for c in landed_cols):
            land_sums = other_df[landed_cols].fillna(0).sum()
            land_per_rd = [land_sums[c] / other_tot_rounds for c in landed_cols]
            tot_land = sum(land_sums)
            pcts_land = [100 * v / tot_land if tot_land else 0 for v in land_sums]
            labels_neg = [f"{land_per_rd[i]:.1f}/rd ({pcts_land[i]:.0f}%)" for i in range(3)]
        else:
            land_per_rd = [0, 0, 0]
            labels_neg = ["", "", ""]
        fig = go.Figure()
        my_color = FIGHTER1_COLOR if name == fighter1 else FIGHTER2_COLOR
        other_color = FIGHTER2_COLOR if name == fighter1 else FIGHTER1_COLOR
        fig.add_trace(go.Bar(
            x=["Head", "Body", "Leg"], y=abs_per_rd, name=f"{name} absorbed",
            text=labels_pos, textposition="outside", marker_color=my_color,
        ))
        fig.add_trace(go.Bar(
            x=["Head", "Body", "Leg"], y=[-v for v in land_per_rd], name=f"{other_name} lands",
            text=labels_neg, textposition="outside", marker_color=other_color,
        ))
        max_pos = max(abs_per_rd) if abs_per_rd else 1
        max_neg = max(land_per_rd) if land_per_rd else 1
        y_min, y_max = _axis_range_with_padding(max_pos, max_neg)
        fig.update_layout(
            barmode="group",
            title=f"{name}: Strikes absorbed vs {other_name} lands (per round avg)",
            yaxis_title="Strikes per round (absorbed ↑ / landed ↓)",
            xaxis_title="Target",
            margin=CHART_MARGIN,
            yaxis_range=[y_min, y_max],
        )
        fig.add_hline(y=0, line_dash="dash", opacity=0.5)
        figures.append(_add_logo_watermark(fig))

    # Combined scatter: head vs body per fight
    fig2 = go.Figure()
    for i, (name, df) in enumerate([(fighter1, f1_fights), (fighter2, f2_fights)]):
        df = df.fillna(0)
        if has_cols(df):
            color = FIGHTER1_COLOR if name == fighter1 else FIGHTER2_COLOR
            fig2.add_trace(go.Scatter(
                x=df[head_col], y=df[body_col], mode="markers", name=name,
                text=df["opponent"],
                hovertemplate="vs %{text}<br>Head: %{x} | Body: %{y}<extra></extra>",
                marker_color=color,
            ))
    fig2.update_layout(
        title="Strikes absorbed per fight (head vs body)",
        xaxis_title="Head",
        yaxis_title="Body",
        hovermode="closest",
        margin=CHART_MARGIN,
    )
    figures.append(_add_logo_watermark(fig2))
    return figures


def strikes_by_position_figures(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> list[go.Figure]:
    """Distance/clinch/ground per-round avg. Returns list of figures (one per fighter)."""
    pos_cols = ["distance", "clinch", "ground"]
    my_cols = [f"my_sig_str_{p}_success" for p in pos_cols]
    opp_cols = [f"opp_sig_str_{p}_success" for p in pos_cols]
    f1_fights = data["fighter1_fights"]
    if not all(c in f1_fights.columns for c in my_cols + opp_cols) or "rounds" not in f1_fights.columns:
        return []

    figures: list[go.Figure] = []
    for name, df, other_name, other_df in [
        (fighter1, f1_fights, fighter2, data["fighter2_fights"]),
        (fighter2, data["fighter2_fights"], fighter1, f1_fights),
    ]:
        tot_rounds = max(1, df["rounds"].sum())
        my_per_rd = (df[my_cols].fillna(0).sum() / tot_rounds).values
        opp_per_rd = (df[opp_cols].fillna(0).sum() / tot_rounds).values
        other_rounds = max(1, other_df["rounds"].sum())
        other_land_per_rd = (other_df[my_cols].fillna(0).sum() / other_rounds).values
        fig = go.Figure()
        my_color = FIGHTER1_COLOR if name == fighter1 else FIGHTER2_COLOR
        other_color = FIGHTER2_COLOR if name == fighter1 else FIGHTER1_COLOR
        fig.add_trace(go.Bar(
            name="Landed",
            x=["Distance", "Clinch", "Ground"],
            y=my_per_rd,
            text=[f"{v:.1f}/rd" for v in my_per_rd],
            textposition="outside",
            marker_color=my_color,
        ))
        fig.add_trace(go.Bar(
            name="Absorbed",
            x=["Distance", "Clinch", "Ground"],
            y=opp_per_rd,
            text=[f"{v:.1f}/rd" for v in opp_per_rd],
            textposition="outside",
            marker_color="#94a3b8",  # slate-400, lighter for absorbed
        ))
        fig.add_trace(go.Bar(
            name=f"{other_name} lands (inverse)",
            x=["Distance", "Clinch", "Ground"],
            y=[-v for v in other_land_per_rd],
            text=[f"{v:.1f}/rd" for v in other_land_per_rd],
            textposition="outside",
            marker_color=other_color,
        ))
        max_pos = float(max(max(my_per_rd), max(opp_per_rd), 0.1))
        max_neg = float(max(other_land_per_rd)) if len(other_land_per_rd) else 0.1
        y_min, y_max = _axis_range_with_padding(max_pos, max_neg)
        fig.update_layout(
            barmode="group",
            title=f"{name}: Strikes by position (per round)",
            yaxis_title="Per round",
            margin=CHART_MARGIN,
            yaxis_range=[y_min, y_max],
        )
        fig.add_hline(y=0, line_dash="dash", opacity=0.5)
        figures.append(_add_logo_watermark(fig))
    return figures


def finishes_by_round_method_figures(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> list[go.Figure]:
    """Finishes by round and method (wins only). Returns list of figures (one per fighter)."""
    x_cats = ["1", "2", "3", "4", "5", "Decisions"]
    figures: list[go.Figure] = []

    for name, df in [(fighter1, data["fighter1_fights"]), (fighter2, data["fighter2_fights"])]:
        wins = df[df["result"] == "win"].copy()
        if wins.empty:
            continue
        if "fight_id" in wins.columns:
            wins = wins.drop_duplicates(subset=["fight_id"], keep="first")
        wins["method_norm"] = wins["method"].map(normalize_method)
        finishes = wins[wins["method_norm"] != "Decision"].copy()
        decisions = wins[wins["method_norm"] == "Decision"].copy()
        finishes["round_display"] = finishes["round"].astype(str)
        decisions["round_display"] = "Decisions"
        combined = pd.concat([finishes, decisions], ignore_index=True)
        if combined.empty:
            continue
        fig = go.Figure()
        for meth in sorted(combined["method_norm"].unique()):
            grp = combined[combined["method_norm"] == meth]
            agg = grp.groupby("round_display", as_index=False).agg(
                count=("method_norm", "count"),
                opponents=("opponent", lambda x: ", ".join(sorted(set(x)))),
            )
            y_vals = [agg.loc[agg["round_display"] == cat, "count"].sum() for cat in x_cats]
            opp_vals = [
                agg.loc[agg["round_display"] == cat, "opponents"].iloc[0]
                if len(agg[agg["round_display"] == cat])
                else ""
                for cat in x_cats
            ]
            fig.add_trace(go.Bar(
                x=x_cats,
                y=y_vals,
                name=meth,
                text=[v if v else "" for v in y_vals],
                textposition="outside",
                customdata=opp_vals,
                hovertemplate="%{x}<br>%{y} win(s)<br>vs %{customdata}<extra></extra>",
            ))
        fig.update_layout(
            barmode="stack",
            title=f"{name}: Finishes by round & method",
            xaxis_title="Round",
            yaxis_title="Count",
            xaxis={"categoryorder": "array", "categoryarray": x_cats},
            margin=CHART_MARGIN,
        )
        figures.append(_add_logo_watermark(fig))
    return figures


def takedowns_per_fight_figure(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> go.Figure | None:
    """Scatter: takedowns landed vs absorbed per fight (with jitter). Returns None if no data."""
    td_landed = "my_totals_td_success"
    td_absorbed = "opp_totals_td_success"
    f1_fights = data["fighter1_fights"]
    if td_landed not in f1_fights.columns or td_absorbed not in f1_fights.columns:
        return None

    fig = go.Figure()
    for name, df in [(fighter1, f1_fights), (fighter2, data["fighter2_fights"])]:
        df = df.dropna(subset=[td_landed, td_absorbed], how="all").fillna(0)
        if df.empty:
            continue
        color = FIGHTER1_COLOR if name == fighter1 else FIGHTER2_COLOR
        x = df[td_landed].astype(float)
        y = df[td_absorbed].astype(float)
        x_jitter = x + [_jitter((name, opp, i)) for i, opp in enumerate(df["opponent"])]
        y_jitter = y + [_jitter((name, opp, i, "y")) for i, opp in enumerate(df["opponent"])]
        fig.add_trace(go.Scatter(
            x=x_jitter, y=y_jitter, mode="markers", name=name,
            text=df["opponent"],
            customdata=df[[td_landed, td_absorbed]].values,
            hovertemplate="vs %{text}<br>Landed: %{customdata[0]:.0f} | Absorbed: %{customdata[1]:.0f}<extra></extra>",
            marker_color=color,
        ))
    fig.update_layout(
        title="Takedowns: landed vs absorbed per fight",
        xaxis_title="Takedowns landed",
        yaxis_title="Takedowns absorbed",
        hovermode="closest",
        margin=CHART_MARGIN,
    )
    return _add_logo_watermark(fig)


def takedowns_absorbed_per_round_figures(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> list[go.Figure]:
    """Per-fighter: absorbed vs opponent lands (per-round avg). Returns list of figures."""
    td_landed = "my_totals_td_success"
    td_absorbed = "opp_totals_td_success"
    f1_fights = data["fighter1_fights"]
    if td_absorbed not in f1_fights.columns or "rounds" not in f1_fights.columns:
        return []

    figures: list[go.Figure] = []
    for name, df, other_name, other_df in [
        (fighter1, f1_fights, fighter2, data["fighter2_fights"]),
        (fighter2, data["fighter2_fights"], fighter1, f1_fights),
    ]:
        df_clean = df.dropna(subset=[td_absorbed], how="all").fillna(0)
        if df_clean.empty:
            continue
        tot_rounds = max(1, df_clean["rounds"].sum())
        abs_per_rd = df_clean[td_absorbed].sum() / tot_rounds
        other_clean = other_df.fillna(0)
        other_rounds = max(1, other_clean["rounds"].sum() if "rounds" in other_clean.columns else len(other_clean))
        other_land_per_rd = other_clean[td_landed].sum() / other_rounds
        fig = go.Figure()
        my_color = FIGHTER1_COLOR if name == fighter1 else FIGHTER2_COLOR
        other_color = FIGHTER2_COLOR if name == fighter1 else FIGHTER1_COLOR
        fig.add_trace(go.Bar(
            x=["Absorbed"], y=[abs_per_rd], name=f"{name} absorbed",
            text=f"{abs_per_rd:.2f}/rd", textposition="outside", marker_color=my_color,
            hovertemplate="%{fullData.name}<br>Per round: %{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=["Absorbed"], y=[-other_land_per_rd], name=f"{other_name} lands",
            text=f"{other_land_per_rd:.2f}/rd", textposition="outside", marker_color=other_color,
            customdata=[[other_land_per_rd]],
            hovertemplate="%{fullData.name}<br>Per round: %{customdata[0]:.2f}<extra></extra>",
        ))
        max_val = max(abs_per_rd, other_land_per_rd, 0.5)
        y_min, y_max = _axis_range_with_padding(max_val, max_val)
        fig.update_layout(
            title=f"{name} absorbed vs {other_name} lands (per round)",
            yaxis_title="Per round",
            barmode="group",
            margin=CHART_MARGIN,
            yaxis_range=[y_min, y_max],
        )
        fig.add_hline(y=0, line_dash="dash", opacity=0.5)
        figures.append(_add_logo_watermark(fig))
    return figures


# -----------------------------------------------------------------------------
# Common opponents
# -----------------------------------------------------------------------------


def _get_common_opponents_and_fights(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """
    Returns (common_opponents, f1_common_df, f2_common_df).
    f1_common_df and f2_common_df are fights vs common opponents only.
    """
    f1 = data["fighter1_fights"]
    f2 = data["fighter2_fights"]
    opp1 = set(f1["opponent"].dropna().astype(str).str.strip().unique())
    opp2 = set(f2["opponent"].dropna().astype(str).str.strip().unique())
    common = sorted(opp1 & opp2)
    f1_common = f1[f1["opponent"].astype(str).str.strip().isin(common)].copy()
    f2_common = f2[f2["opponent"].astype(str).str.strip().isin(common)].copy()
    return common, f1_common, f2_common


def common_opponents_record_table(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> pd.DataFrame:
    """
    Table of common opponents with W-L record per fighter.
    Returns DataFrame with columns: Opponent, Fighter1 Record, Fighter2 Record.
    Record format: "W-L" (e.g. "1-2").
    """
    common, f1, f2 = _get_common_opponents_and_fights(data, fighter1, fighter2)
    if not common:
        return pd.DataFrame(columns=["Opponent", f"{fighter1} Record", f"{fighter2} Record"])

    rows = []
    for opp in common:
        f1_vs = f1[f1["opponent"].astype(str).str.strip() == opp]
        f2_vs = f2[f2["opponent"].astype(str).str.strip() == opp]
        w1 = (f1_vs["result"].str.lower() == "win").sum()
        l1 = (f1_vs["result"].str.lower() == "loss").sum()
        w2 = (f2_vs["result"].str.lower() == "win").sum()
        l2 = (f2_vs["result"].str.lower() == "loss").sum()
        rows.append({
            "Opponent": opp,
            f"{fighter1} Record": f"{w1}-{l1}",
            f"{fighter2} Record": f"{w2}-{l2}",
        })
    return pd.DataFrame(rows)


def common_opponents_performance_figure(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> go.Figure | None:
    """
    Grouped bar chart: wins AND losses per fighter vs each common opponent.
    Tooltip shows actual results (e.g. "Win Jun 2021, Loss Apr 2019") not just count.
    """
    common, f1, f2 = _get_common_opponents_and_fights(data, fighter1, fighter2)
    if not common:
        return None

    wins1, loss1, wins2, loss2 = [], [], [], []
    hover1_w, hover1_l, hover2_w, hover2_l = [], [], [], []

    for opp in common:
        f1_vs = f1[f1["opponent"].astype(str).str.strip() == opp].copy()
        f2_vs = f2[f2["opponent"].astype(str).str.strip() == opp].copy()
        f1_vs["_d"] = pd.to_datetime(f1_vs["event_date"], errors="coerce")
        f2_vs["_d"] = pd.to_datetime(f2_vs["event_date"], errors="coerce")
        f1_vs = f1_vs.sort_values("_d")
        f2_vs = f2_vs.sort_values("_d")

        def fmt_fights(df: pd.DataFrame) -> tuple[list[str], list[str]]:
            wins, losses = [], []
            for _, r in df.iterrows():
                res = str(r.get("result", "")).lower()
                dt = r.get("event_date", "")
                meth = r.get("method", "")
                label = f"{res.upper()} {dt}" + (f" ({meth})" if meth else "")
                if res == "win":
                    wins.append(label)
                elif res == "loss":
                    losses.append(label)
            return wins, losses

        h1_w, h1_l = fmt_fights(f1_vs)
        h2_w, h2_l = fmt_fights(f2_vs)
        wins1.append(len(h1_w))
        loss1.append(len(h1_l))
        wins2.append(len(h2_w))
        loss2.append(len(h2_l))
        hover1_w.append("<br>".join(h1_w) if h1_w else "—")
        hover1_l.append("<br>".join(h1_l) if h1_l else "—")
        hover2_w.append("<br>".join(h2_w) if h2_w else "—")
        hover2_l.append("<br>".join(h2_l) if h2_l else "—")

    loss1_neg = [-x for x in loss1]
    loss2_neg = [-x for x in loss2]

    max_pos = max(wins1 + wins2, default=1)
    max_neg = max(loss1 + loss2, default=1)
    y_min, y_max = _axis_range_with_padding(max_pos, max_neg)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"{fighter1} Wins",
        x=common,
        y=wins1,
        text=wins1,
        textposition="outside",
        marker_color=FIGHTER1_COLOR,
        customdata=list(zip([fighter1] * len(common), hover1_w)),
        hovertemplate="<b>%{customdata[0]}</b><br>Wins:<br>%{customdata[1]}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name=f"{fighter1} Losses",
        x=common,
        y=loss1_neg,
        text=loss1,
        textposition="outside",
        marker_color="#fca5a5",  # light red for losses
        customdata=list(zip([fighter1] * len(common), hover1_l)),
        hovertemplate="<b>%{customdata[0]}</b><br>Losses:<br>%{customdata[1]}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name=f"{fighter2} Wins",
        x=common,
        y=wins2,
        text=wins2,
        textposition="outside",
        marker_color=FIGHTER2_COLOR,
        customdata=list(zip([fighter2] * len(common), hover2_w)),
        hovertemplate="<b>%{customdata[0]}</b><br>Wins:<br>%{customdata[1]}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name=f"{fighter2} Losses",
        x=common,
        y=loss2_neg,
        text=loss2,
        textposition="outside",
        marker_color="#fed7aa",  # light orange for losses
        customdata=list(zip([fighter2] * len(common), hover2_l)),
        hovertemplate="<b>%{customdata[0]}</b><br>Losses:<br>%{customdata[1]}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.7)
    fig.update_layout(
        barmode="group",
        title="Record vs common opponents (Wins above, Losses below)",
        xaxis_title="Opponent",
        yaxis_title="Wins / Losses",
        xaxis_tickangle=-45,
        legend_title="Result",
        margin=CHART_MARGIN,
        yaxis_range=[y_min, y_max],
    )
    return _add_logo_watermark(fig)


def common_opponents_scatter_figure(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> go.Figure | None:
    """
    Timeline scatter: X=date, Y=opponent. Four traces: Fighter1 Win/Loss, Fighter2 Win/Loss.
    """
    common, f1, f2 = _get_common_opponents_and_fights(data, fighter1, fighter2)
    if not common:
        return None

    fig = go.Figure()
    for name, df, win_color, loss_color in [
        (fighter1, f1, FIGHTER1_COLOR, "#dc2626"),
        (fighter2, f2, FIGHTER2_COLOR, "#9333ea"),
    ]:
        df = df.copy()
        df["_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        df = df.dropna(subset=["_date", "opponent"])
        if df.empty:
            continue
        for result, color, label in [("win", win_color, "Win"), ("loss", loss_color, "Loss")]:
            sub = df[df["result"].str.lower() == result]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["_date"],
                y=sub["opponent"].astype(str).str.strip(),
                mode="markers",
                name=f"{name} {label}",
                marker_color=color,
                text=sub.apply(
                    lambda r: f"{r.get('method', '')}" if r.get("method") else "",
                    axis=1,
                ),
                hovertemplate="%{y}<br>%{x|%b %Y}<br>" + f"{name} {label}" + "<br>%{text}<extra></extra>",
            ))
    fig.update_layout(
        title="When each fighter faced common opponents",
        xaxis_title="Date",
        yaxis_title="Opponent",
        hovermode="closest",
        xaxis={"tickformat": "%b %Y"},
        legend_title="Fighter – Result",
        margin=CHART_MARGIN,
    )
    return _add_logo_watermark(fig)


def _build_common_opponent_chart_data(
    comb: pd.DataFrame,
    value_col: str,
    value_label: str,
    fighter1: str,
    fighter2: str,
) -> tuple[list[str], list[float], list[str], list[str], list[int], list[str]]:
    """
    Build chart data grouped by opponent, one bar per event.
    Returns (x_labels, y_values, hover_texts, colors, group_starts, group_opponents).
    Bars are sorted by opponent, then date. Hover shows full event details.
    """
    comb = comb.sort_values(["opponent", "_date"]).copy()
    x_labels = []
    y_values = []
    hover_texts = []
    colors = []
    group_starts: list[int] = []
    group_opponents: list[str] = []

    prev_opp = None
    for _, row in comb.iterrows():
        opp = str(row.get("opponent", "")).strip()
        fighter = str(row.get("fighter", ""))
        dt = row.get("_date")
        date_str = pd.Timestamp(dt).strftime("%b %Y") if pd.notna(dt) else ""
        result = row.get("result", "")
        val = row.get(value_col, 0)

        if opp != prev_opp:
            group_starts.append(len(x_labels))
            group_opponents.append(opp)
            prev_opp = opp

        last_name = fighter.split()[-1] if fighter else ""
        short_label = f"{last_name} {date_str}" if last_name else date_str
        x_labels.append(short_label)
        y_values.append(float(val))
        hover_texts.append(f"{fighter} vs {opp}<br>{date_str} | {result}<br>{value_label}: {val:.0f}%")
        colors.append(FIGHTER1_COLOR if fighter == fighter1 else FIGHTER2_COLOR)

    return x_labels, y_values, hover_texts, colors, group_starts, group_opponents


def _add_opponent_group_separators(fig: go.Figure, group_starts: list[int], n_bars: int) -> None:
    """Add vertical dashed lines between opponent groups."""
    for i in group_starts[1:]:
        fig.add_vline(
            x=i - 0.5,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
        )


def _build_common_opponent_stats_df(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> pd.DataFrame | None:
    """
    Build a combined DataFrame of all fights vs common opponents.
    Columns: fighter, opponent, event_date, result, strike_acc_pct, strike_share_pct,
    td_acc_pct, head_pct, body_pct, leg_pct (where available).
    """
    common, f1, f2 = _get_common_opponents_and_fights(data, fighter1, fighter2)
    if not common:
        return None

    sig_success = "my_totals_sig_str_success"
    sig_attempt = "my_totals_sig_str_attempt"
    sig_opp = "opp_totals_sig_str_success"
    td_success = "my_totals_td_success"
    td_attempt = "my_totals_td_attempt"
    target_cols = ["head", "body", "leg"]

    def add_pcts(df: pd.DataFrame, fighter: str) -> pd.DataFrame:
        d = df.copy()
        d["fighter"] = fighter
        d["_date"] = pd.to_datetime(d["event_date"], errors="coerce")

        # Strike accuracy %
        if sig_success in d.columns and sig_attempt in d.columns:
            land = pd.to_numeric(d[sig_success], errors="coerce").fillna(0)
            att = pd.to_numeric(d[sig_attempt], errors="coerce").fillna(0).replace(0, 1)
            d["strike_acc_pct"] = (100 * land / att).fillna(0)
        else:
            d["strike_acc_pct"] = 0.0

        # Strike share %: landed / (landed + absorbed)
        if sig_success in d.columns and sig_opp in d.columns:
            landed = pd.to_numeric(d[sig_success], errors="coerce").fillna(0)
            absorbed = pd.to_numeric(d[sig_opp], errors="coerce").fillna(0)
            total = landed + absorbed
            d["strike_share_pct"] = 100 * landed / total.where(total > 0, 1)
            d["strike_share_pct"] = d["strike_share_pct"].fillna(50.0)
        else:
            d["strike_share_pct"] = 50.0

        # Takedown success %
        if td_success in d.columns and td_attempt in d.columns:
            succ = pd.to_numeric(d[td_success], errors="coerce").fillna(0)
            att = pd.to_numeric(d[td_attempt], errors="coerce").fillna(0)
            d["td_acc_pct"] = 100 * succ / att.where(att > 0, 1)
            d["td_acc_pct"] = d["td_acc_pct"].fillna(0)
        else:
            d["td_acc_pct"] = 0.0

        # Target percentages (head/body/leg as % of sig strikes)
        my_target = [f"my_sig_str_{t}_success" for t in target_cols]
        if all(c in d.columns for c in my_target):
            my_sum = sum(pd.to_numeric(d[c], errors="coerce").fillna(0) for c in my_target)
            for i, t in enumerate(target_cols):
                d[f"{t}_pct"] = 100 * pd.to_numeric(d[my_target[i]], errors="coerce").fillna(0) / my_sum.where(my_sum > 0, 1)
            d["head_pct"] = d["head_pct"].fillna(0)
            d["body_pct"] = d["body_pct"].fillna(0)
            d["leg_pct"] = d["leg_pct"].fillna(0)
        else:
            d["head_pct"] = d["body_pct"] = d["leg_pct"] = 0.0

        return d

    f1_p = add_pcts(f1, fighter1)
    f2_p = add_pcts(f2, fighter2)
    comb = pd.concat([f1_p, f2_p], ignore_index=True)
    comb = comb[comb["opponent"].astype(str).str.strip().isin(common)]
    return comb


def _common_opponents_metric_figure(
    comb: pd.DataFrame,
    value_col: str,
    value_label: str,
    title: str,
    fighter1: str,
    fighter2: str,
    y_ref_line: float | None = None,
) -> go.Figure | None:
    """
    Single bar chart: one bar per event, grouped by opponent. Percentage on top, event in tooltip.
    """
    if comb is None or comb.empty or value_col not in comb.columns:
        return None

    x_labels, y_values, hover_texts, colors, group_starts, group_opponents = _build_common_opponent_chart_data(
        comb, value_col, value_label, fighter1, fighter2
    )
    if not x_labels:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_values,
        text=[f"{v:.0f}%" for v in y_values],
        textposition="outside",
        marker_color=colors,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
        showlegend=False,
    ))
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=fighter1,
            marker=dict(color=FIGHTER1_COLOR, size=12, symbol="square"),
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=fighter2,
            marker=dict(color=FIGHTER2_COLOR, size=12, symbol="square"),
            showlegend=True,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Fight (grouped by opponent)",
        yaxis_title=f"{value_label} (%)",
        xaxis_tickangle=-45,
        yaxis_range=[0, 110],
        legend_title="Fighter",
        margin=CHART_MARGIN,
    )
    _add_opponent_group_separators(fig, group_starts, len(x_labels))

    for i, opp in enumerate(group_opponents):
        start = group_starts[i]
        end = group_starts[i + 1] if i + 1 < len(group_starts) else len(x_labels)
        center_x = (start + end - 1) / 2
        fig.add_annotation(
            text=opp,
            x=center_x,
            y=1.02,
            yref="paper",
            showarrow=False,
            font=dict(size=11, color="gray"),
            xanchor="center",
        )

    if y_ref_line is not None:
        fig.add_hline(y=y_ref_line, line_dash="dash", opacity=0.5)
    return _add_logo_watermark(fig)


def common_opponents_striking_figure(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> go.Figure | None:
    """
    Strike share %: (your strikes / total strikes in fight). 50%+ = you landed more than absorbed.
    One bar per event, grouped by opponent. Blue=Fighter1, Orange=Fighter2.
    """
    comb = _build_common_opponent_stats_df(data, fighter1, fighter2)
    return _common_opponents_metric_figure(
        comb,
        "strike_share_pct",
        "Strike share",
        "Striking vs common opponents: Strike share % (your strikes / total). 50%+ = landed more than absorbed.",
        fighter1,
        fighter2,
        y_ref_line=50,
    )


def common_opponents_strike_accuracy_figure(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> go.Figure | None:
    """
    Strike accuracy %: (landed / attempted). How precise were your strikes.
    Grouped by opponent, one bar per event.
    """
    comb = _build_common_opponent_stats_df(data, fighter1, fighter2)
    return _common_opponents_metric_figure(
        comb,
        "strike_acc_pct",
        "Strike accuracy",
        "Strike accuracy % vs common opponents (landed / attempted)",
        fighter1,
        fighter2,
    )


def common_opponents_strike_share_figure(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> go.Figure | None:
    """Alias for common_opponents_striking_figure for backwards compatibility."""
    return common_opponents_striking_figure(data, fighter1, fighter2)


def common_opponents_takedown_figure(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> go.Figure | None:
    """
    Takedown success % per fight vs common opponents. Grouped by opponent, one bar per event.
    """
    comb = _build_common_opponent_stats_df(data, fighter1, fighter2)
    return _common_opponents_metric_figure(
        comb,
        "td_acc_pct",
        "Takedown success",
        "Takedown success % vs common opponents",
        fighter1,
        fighter2,
    )


def common_opponents_strike_share_by_target_figure(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> go.Figure | None:
    """
    Stacked bars: strike distribution by target (head/body/leg %) per fight.
    Grouped by opponent, one stacked bar per event.
    """
    comb = _build_common_opponent_stats_df(data, fighter1, fighter2)
    if comb is None or comb.empty or "head_pct" not in comb.columns:
        return None

    comb = comb.sort_values(["opponent", "_date"])
    x_labels, _, _, _, group_starts, group_opponents = _build_common_opponent_chart_data(
        comb, "head_pct", "Head", fighter1, fighter2
    )

    fig = go.Figure()
    for target, label in [("head_pct", "Head"), ("body_pct", "Body"), ("leg_pct", "Leg")]:
        fig.add_trace(go.Bar(
            name=label,
            x=x_labels,
            y=comb[target].tolist(),
            text=[f"{v:.0f}%" for v in comb[target]],
            textposition="inside",
            marker_color=TARGET_COLORS[label],
        ))
    fig.update_layout(
        barmode="stack",
        title="Strike distribution by target (head/body/leg %) vs common opponents",
        xaxis_title="Fight (grouped by opponent)",
        yaxis_title="Percentage of sig strikes",
        xaxis_tickangle=-45,
        yaxis_range=[0, 110],
        legend_title="Target",
        margin=CHART_MARGIN,
    )
    _add_opponent_group_separators(fig, group_starts, len(x_labels))
    for i, opp in enumerate(group_opponents):
        start = group_starts[i]
        end = group_starts[i + 1] if i + 1 < len(group_starts) else len(x_labels)
        center_x = (start + end - 1) / 2
        fig.add_annotation(
            text=opp,
            x=center_x,
            y=1.02,
            yref="paper",
            showarrow=False,
            font=dict(size=11, color="gray"),
            xanchor="center",
        )
    return _add_logo_watermark(fig)


def common_opponents_strike_scatter_figure(
    data: FighterData,
    fighter1: str,
    fighter2: str,
) -> go.Figure | None:
    """
    Scatter: X = strike accuracy %, Y = strike share %.
    Each point = one fight vs common opponent; color by fighter.
    """
    comb = _build_common_opponent_stats_df(data, fighter1, fighter2)
    if comb is None or comb.empty:
        return None

    fig = go.Figure()
    for name in [fighter1, fighter2]:
        sub = comb[comb["fighter"] == name]
        if sub.empty:
            continue
        color = FIGHTER1_COLOR if name == fighter1 else FIGHTER2_COLOR
        fig.add_trace(go.Scatter(
            x=sub["strike_acc_pct"],
            y=sub["strike_share_pct"],
            mode="markers",
            name=name,
            text=sub.apply(lambda r: f"{r['opponent']} {pd.Timestamp(r['_date']).strftime('%b %Y')}", axis=1),
            hovertemplate="%{text}<br>Accuracy: %{x:.0f}% | Share: %{y:.0f}%<extra></extra>",
            marker_color=color,
        ))
    fig.update_layout(
        title="Strike accuracy % vs strike share % (common opponent fights)",
        xaxis_title="Strike accuracy (%)",
        yaxis_title="Strike share (%)",
        hovermode="closest",
        xaxis_range=[0, 105],
        yaxis_range=[0, 105],
        margin=CHART_MARGIN,
    )
    fig.add_hline(y=50, line_dash="dash", opacity=0.5)
    fig.add_vline(x=50, line_dash="dash", opacity=0.5)
    return _add_logo_watermark(fig)
