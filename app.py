"""UFC Stats Fighter Comparison - Streamlit App."""

import os
import sys
from pathlib import Path

import streamlit as st

# Add project root so we can import src
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from src.ufc_stats import UFCStatsClient
from src.standard_metrics import (
    record_comparison_figure,
    career_stats_comparison_figure,
    cumulative_wins_figure,
    streak_and_last_5,
    head_to_head_advantages,
    strikes_absorbed_by_target_figures,
    strikes_by_position_figures,
    finishes_by_round_method_figures,
    takedowns_per_fight_figure,
    takedowns_absorbed_per_round_figures,
    common_opponents_record_table,
    common_opponents_performance_figure,
    common_opponents_scatter_figure,
    common_opponents_striking_figure,
    common_opponents_strike_accuracy_figure,
    common_opponents_takedown_figure,
    common_opponents_strike_share_by_target_figure,
)

# Cache directory: use env var for deployment (e.g. Hugging Face Spaces persistent storage)
CACHE_DIR = os.environ.get("UFC_CACHE_DIR", root / "notebooks" / "data")
CACHE_PATH = Path(CACHE_DIR) if isinstance(CACHE_DIR, str) else CACHE_DIR

st.set_page_config(page_title="UFC Stats – Fighter Analysis", layout="wide")

st.title("UFC Stats – Fighter Analysis")
st.caption("Data from [UFC Stats API](https://ufcapi.aristotle.me)")

# Sidebar: inputs
with st.sidebar:
    st.header("Compare Fighters")
    fighter1 = st.text_input("Fighter 1", value="Charles Oliveira", placeholder="e.g. Charles Oliveira")
    fighter2 = st.text_input("Fighter 2", value="Max Holloway", placeholder="e.g. Max Holloway")

    force_refresh = st.checkbox(
        "Force refresh (data looks stale)",
        value=False,
        help="Bypass cache and fetch fresh data from the API. Overwrites cache for both fighters.",
    )

    compare_clicked = st.button("Compare", type="primary", use_container_width=True)

# Main content
if compare_clicked and fighter1.strip() and fighter2.strip():
    with st.spinner("Fetching fighter data…"):
        client = UFCStatsClient(cache_path=CACHE_PATH)
        try:
            data = client.get_all_data_for_fighters(
                fighter1.strip(),
                fighter2.strip(),
                force_refresh=force_refresh,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

    if force_refresh:
        st.success("Data refreshed from API and cache updated.")

    # Overview
    st.header("Overview")

    st.subheader("Record / win-loss comparison")
    fig = record_comparison_figure(data, fighter1, fighter2)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Career stats comparison")
    fig = career_stats_comparison_figure(data, fighter1, fighter2)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Win/loss trend over fights")
    _, _, fig = cumulative_wins_figure(data, fighter1, fighter2)
    st.plotly_chart(fig, use_container_width=True)

    # Streak / momentum
    st.subheader("Streak / momentum summary")
    streak_data = streak_and_last_5(data, fighter1, fighter2)
    for name in [fighter1, fighter2]:
        info = streak_data[name]
        st.markdown(f"**{name}**: Current streak = {info['streak']:+d}")
        st.markdown("Last 5 fights (most recent first):")
        st.dataframe(info["last_5"], use_container_width=True, hide_index=True)

    # Common opponents
    st.header("Common Opponents")
    rec_table = common_opponents_record_table(data, fighter1, fighter2)
    if rec_table.empty:
        st.info("No common opponents.")
    else:
        st.dataframe(rec_table, use_container_width=True, hide_index=True)
        fig = common_opponents_performance_figure(data, fighter1, fighter2)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        fig2 = common_opponents_scatter_figure(data, fighter1, fighter2)
        if fig2 is not None:
            st.plotly_chart(fig2, use_container_width=True)
        for fig_fn in [
            common_opponents_striking_figure,
            common_opponents_strike_accuracy_figure,
            common_opponents_takedown_figure,
            common_opponents_strike_share_by_target_figure,
        ]:
            fig = fig_fn(data, fighter1, fighter2)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    # Head-to-head advantages
    st.header("Head-to-head advantages")
    adv = head_to_head_advantages(data, fighter1, fighter2)
    for name in [fighter1, fighter2]:
        val = adv.get(name)
        if val:
            st.markdown(f"**{name}** leads in: {val}")

    # Secondary breakdowns
    st.header("Secondary Breakdowns")

    st.subheader("Strikes absorbed by target (defense profile)")
    for fig in strikes_absorbed_by_target_figures(data, fighter1, fighter2):
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strikes by position: distance / clinch / ground")
    for fig in strikes_by_position_figures(data, fighter1, fighter2):
        st.plotly_chart(fig, use_container_width=True)

    # Finishes
    st.subheader("Finishes by round & method")
    for fig in finishes_by_round_method_figures(data, fighter1, fighter2):
        st.plotly_chart(fig, use_container_width=True)

    # Takedowns
    st.subheader("Takedowns per fight")
    fig = takedowns_per_fight_figure(data, fighter1, fighter2)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Takedowns absorbed per round")
    for fig in takedowns_absorbed_per_round_figures(data, fighter1, fighter2):
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info(
        "Enter two fighter names and click **Compare** to analyze. "
        "Check **Force refresh** if data looks stale — this fetches fresh data and overwrites the cache."
    )
