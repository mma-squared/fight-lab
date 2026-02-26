# Fight Lab

Useful code and fight analytics for UFC fights we cover.

## Getting Started

1. **Copy the starting template** into the `analysis` folder:
   ```
   notebooks/starting-template.ipynb  →  analysis/your-analysis.ipynb
   ```

2. **Update the fighter names** at the top of the notebook:
   ```python
   FIGHTER_1 = "Charles Oliveira"
   FIGHTER_2 = "Max Holloway"
   ```

3. **Run the notebook.** It fetches fighter info, fight history, and head-to-head stats from the [UFC Stats API](https://ufcapi.aristotle.me). Data is cached locally in `notebooks/data/` so repeat runs are fast.

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e .
```

## Project Structure

- `notebooks/starting-template.ipynb` — Template notebook for fighter comparisons (charts and visualizations)
- `notebooks/data-analysis.ipynb` — Raw data tables with key metrics reference
- `analysis/` — Your analysis notebooks (copy from starting-template)
- `src/ufc_stats/` — UFC Stats API wrapper library
