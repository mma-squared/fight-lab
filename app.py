"""UFC Stats Fighter Comparison - Streamlit App."""

import json
import os
import sys
from pathlib import Path

# Load .env for local dev (Streamlit Community Cloud uses Secrets -> env vars)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
from streamlit.components.v1 import html as st_html

# Add project root so we can import src
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from src.ufc_stats import UFCStatsClient, SupabaseStorageCache
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

GA4_MEASUREMENT_ID = "G-5R0QD7V4X0"


def _normalize_fighter_pair(f1: str, f2: str) -> str:
    """Normalize pair so A vs B == B vs A (alphabetically sorted)."""
    return " vs ".join(sorted([f1.strip(), f2.strip()], key=str.lower))


def _ga4_tracking_script(fighter_search_event: dict | None = None) -> str:
    """Build GA4 script: base config + optional fighter_search event."""
    script = f"""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id={GA4_MEASUREMENT_ID}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{ dataLayer.push(arguments); }}
  gtag('js', new Date());
  gtag('config', '{GA4_MEASUREMENT_ID}');
"""
    if fighter_search_event:
        # Single search event for the comparison (counts popular fighters + pairs)
        # fighter_pair is normalized so order doesn't matter
        params = ", ".join(
            f"{k}: {json.dumps(v)}" for k, v in fighter_search_event.items()
        )
        script += f"\n  gtag('event', 'search', {{{params}}});\n"
    script += "\n</script>"
    return script

# Cache: Supabase (Streamlit Community Cloud) or local filesystem
def _make_ufc_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if url and key:
        from supabase import create_client
        supabase = create_client(url, key)
        bucket = os.environ.get("SUPABASE_CACHE_BUCKET", "fighter-cache")
        return UFCStatsClient(cache_backend=SupabaseStorageCache(supabase, bucket=bucket))
    cache_dir = os.environ.get("UFC_CACHE_DIR", root / "notebooks" / "data")
    cache_path = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
    return UFCStatsClient(cache_path=cache_path)

st.set_page_config(page_title="UFC Stats – Fighter Analysis", layout="wide")

# MMA Squared branding & CTA
st.markdown("## MMA Squared")
st.markdown(
    "**The home of MMA fight breakdowns and data-driven analytics.** "
    "[Subscribe on YouTube](https://www.youtube.com/@MMA-Squared) · "
    "[Telegram](https://t.me/mmasquared) · "
    "[Contact](mailto:team@mmasquared.com)"
)
st.divider()

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
        client = _make_ufc_client()
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

    # GA4: base script + fighter search event (single search + normalized pair for analytics)
    f1, f2 = fighter1.strip(), fighter2.strip()
    fighter_search_event = {
        "search_term": f"{f1} vs {f2}",
        "fighter_1": f1,
        "fighter_2": f2,
        "fighter_pair": _normalize_fighter_pair(f1, f2),
    }
    st_html(_ga4_tracking_script(fighter_search_event), height=0)

    tab_overview, tab_striking, tab_grappling, tab_common = st.tabs([
        "Overview",
        "Striking",
        "Grappling",
        "Common Opponents",
    ])

    with tab_overview:
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

        st.subheader("Streak / momentum summary")
        streak_data = streak_and_last_5(data, fighter1, fighter2)
        for name in [fighter1, fighter2]:
            info = streak_data[name]
            st.markdown(f"**{name}**: Current streak = {info['streak']:+d}")
            st.markdown("Last 5 fights (most recent first):")
            st.dataframe(info["last_5"], use_container_width=True, hide_index=True)

        st.subheader("Head-to-head advantages")
        adv = head_to_head_advantages(data, fighter1, fighter2)
        for name in [fighter1, fighter2]:
            val = adv.get(name)
            if val:
                st.markdown(f"**{name}** leads in: {val}")

        st.subheader("Finishes by round & method")
        for fig in finishes_by_round_method_figures(data, fighter1, fighter2):
            st.plotly_chart(fig, use_container_width=True)

    with tab_striking:
        st.subheader("Strikes absorbed by target (defense profile)")
        for fig in strikes_absorbed_by_target_figures(data, fighter1, fighter2):
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Strikes by position: distance / clinch / ground")
        for fig in strikes_by_position_figures(data, fighter1, fighter2):
            st.plotly_chart(fig, use_container_width=True)

    with tab_grappling:
        st.subheader("Takedowns per fight")
        fig = takedowns_per_fight_figure(data, fighter1, fighter2)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Takedowns absorbed per round")
        for fig in takedowns_absorbed_per_round_figures(data, fighter1, fighter2):
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    with tab_common:
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

else:
    # GA4 base script when no comparison (or before first comparison)
    st_html(_ga4_tracking_script(), height=0)
    st.info(
        "Enter two fighter names and click **Compare** to analyze. "
        "Check **Force refresh** if data looks stale — this fetches fresh data and overwrites the cache."
    )
