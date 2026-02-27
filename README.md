# Fight Lab

Useful code and fight analytics for UFC fights we cover.

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows

# Install dependencies (includes Streamlit)
pip install -e .

# Run the Streamlit app
streamlit run app.py
```

The app fetches fighter info, fight history, and head-to-head stats from the [UFC Stats API](https://ufcapi.aristotle.me). Data is cached locally so repeat runs are fast. Use the sidebar to enter two fighter names, click **Compare**, and optionally enable **Force refresh** if data looks stale.

## Project Structure

- `app.py` — Streamlit app for fighter comparisons (charts and visualizations)
- `src/ufc_stats/` — UFC Stats API wrapper library
