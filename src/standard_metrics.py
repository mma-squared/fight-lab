"""
Standard UFC fighter comparison metrics and visualizations.

Extracts analysis logic from notebooks into reusable functions with proper types.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import plotly.graph_objects as go

# Type alias for the data dict returned by UFCStatsClient.get_all_data_for_fighters
FighterData = dict[str, pd.DataFrame]

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
    fig.update_layout(
        barmode="stack",
        title="Career record",
        yaxis_title="Count",
        legend_title="Result",
    )
    return fig


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
    ))
    fig.add_trace(go.Bar(
        name=fighter2, x=stats_df["Stat"], y=y2,
        text=y2.round(2), textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        title="Career stats comparison",
        xaxis_tickangle=-45,
        legend_title="Fighter",
    )
    return fig


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
    ))
    fig.add_trace(go.Scatter(
        x=df2["fight_num"], y=df2["cum_wins"],
        mode="lines+markers", name=fighter2,
        text=df2["opponent"],
        hovertemplate="Fight %{x}: vs %{text}<br>Cumulative: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title="Cumulative W-L trend (win=+1, loss=-1)",
        xaxis_title="Fight number",
        yaxis_title="Cumulative",
        hovermode="x unified",
    )
    fig.add_hline(y=0, line_dash="dash", opacity=0.5)
    return df1, df2, fig


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
        fig.add_trace(go.Bar(
            x=["Head", "Body", "Leg"], y=abs_per_rd, name=f"{name} absorbed",
            text=labels_pos, textposition="outside", marker_color="steelblue",
        ))
        fig.add_trace(go.Bar(
            x=["Head", "Body", "Leg"], y=[-v for v in land_per_rd], name=f"{other_name} lands",
            text=labels_neg, textposition="outside", marker_color="coral",
        ))
        fig.update_layout(
            barmode="group",
            title=f"{name}: Strikes absorbed vs {other_name} lands (per round avg)",
            yaxis_title="Strikes per round (absorbed ↑ / landed ↓)",
            xaxis_title="Target",
        )
        fig.add_hline(y=0, line_dash="dash", opacity=0.5)
        figures.append(fig)

    # Combined scatter: head vs body per fight
    fig2 = go.Figure()
    for name, df in [(fighter1, f1_fights), (fighter2, f2_fights)]:
        df = df.fillna(0)
        if has_cols(df):
            fig2.add_trace(go.Scatter(
                x=df[head_col], y=df[body_col], mode="markers", name=name,
                text=df["opponent"],
                hovertemplate="vs %{text}<br>Head: %{x} | Body: %{y}<extra></extra>",
            ))
    fig2.update_layout(
        title="Strikes absorbed per fight (head vs body)",
        xaxis_title="Head",
        yaxis_title="Body",
        hovermode="closest",
    )
    figures.append(fig2)
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
        fig.add_trace(go.Bar(
            name="Landed",
            x=["Distance", "Clinch", "Ground"],
            y=my_per_rd,
            text=[f"{v:.1f}/rd" for v in my_per_rd],
            textposition="outside",
            marker_color="steelblue",
        ))
        fig.add_trace(go.Bar(
            name="Absorbed",
            x=["Distance", "Clinch", "Ground"],
            y=opp_per_rd,
            text=[f"{v:.1f}/rd" for v in opp_per_rd],
            textposition="outside",
            marker_color="lightblue",
        ))
        fig.add_trace(go.Bar(
            name=f"{other_name} lands (inverse)",
            x=["Distance", "Clinch", "Ground"],
            y=[-v for v in other_land_per_rd],
            text=[f"{v:.1f}/rd" for v in other_land_per_rd],
            textposition="outside",
            marker_color="coral",
        ))
        fig.update_layout(
            barmode="group",
            title=f"{name}: Strikes by position (per round)",
            yaxis_title="Per round",
        )
        fig.add_hline(y=0, line_dash="dash", opacity=0.5)
        figures.append(fig)
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
        )
        figures.append(fig)
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
        x = df[td_landed].astype(float)
        y = df[td_absorbed].astype(float)
        x_jitter = x + [_jitter((name, opp, i)) for i, opp in enumerate(df["opponent"])]
        y_jitter = y + [_jitter((name, opp, i, "y")) for i, opp in enumerate(df["opponent"])]
        fig.add_trace(go.Scatter(
            x=x_jitter, y=y_jitter, mode="markers", name=name,
            text=df["opponent"],
            customdata=df[[td_landed, td_absorbed]].values,
            hovertemplate="vs %{text}<br>Landed: %{customdata[0]:.0f} | Absorbed: %{customdata[1]:.0f}<extra></extra>",
        ))
    fig.update_layout(
        title="Takedowns: landed vs absorbed per fight",
        xaxis_title="Takedowns landed",
        yaxis_title="Takedowns absorbed",
        hovermode="closest",
    )
    return fig


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
        fig.add_trace(go.Bar(
            x=["Absorbed"], y=[abs_per_rd], name=f"{name} absorbed",
            text=f"{abs_per_rd:.2f}/rd", textposition="outside", marker_color="steelblue",
            hovertemplate="%{fullData.name}<br>Per round: %{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=["Absorbed"], y=[-other_land_per_rd], name=f"{other_name} lands",
            text=f"{other_land_per_rd:.2f}/rd", textposition="outside", marker_color="coral",
            customdata=[[other_land_per_rd]],
            hovertemplate="%{fullData.name}<br>Per round: %{customdata[0]:.2f}<extra></extra>",
        ))
        fig.update_layout(
            title=f"{name} absorbed vs {other_name} lands (per round)",
            yaxis_title="Per round",
            barmode="group",
        )
        fig.add_hline(y=0, line_dash="dash", opacity=0.5)
        figures.append(fig)
    return figures
