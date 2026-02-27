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

The app fetches fighter info, fight history, and head-to-head stats from the [UFC Stats API](https://ufcapi.aristotle.me). Data is cached locally (or in Supabase when configured). Use the sidebar to enter two fighter names, click **Compare**, and optionally enable **Force refresh** if data looks stale.

## Optional: Supabase cache (persistent storage)

For Streamlit Community Cloud or other environments with ephemeral storage, use [Supabase Storage](https://supabase.com) for persistent cache:

1. Create a [Supabase project](https://supabase.com/dashboard)
2. In **Storage**, create a bucket (default name: `fighter-cache`, or set `SUPABASE_CACHE_BUCKET`)
3. Copy `.env.example` to `.env` and add your credentials (Dashboard → Settings → API):
   - `SUPABASE_URL` — Project URL
   - `SUPABASE_SERVICE_KEY` — Service role key (keep secret)

Local: `.env` is loaded automatically. Secrets are never committed (`.env` is gitignored).

## Deploy to Streamlit Community Cloud

1. Push to GitHub and [deploy on share.streamlit.io](https://share.streamlit.io)
2. In the app’s **Settings** → **Secrets**, add:
   ```
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_SERVICE_KEY = "your-service-role-key"
   ```
3. Ensure the cache bucket exists (default: `fighter-cache`)

## Deploy to Railway (custom domain)

For `stats.mmasquared.com`, deploy on [Railway](https://railway.app). See **[RAILWAY.md](RAILWAY.md)** for step-by-step instructions.

## Project Structure

- `app.py` — Streamlit app for fighter comparisons
- `assets/` — MMA Squared logo (chart watermarks)
- `RAILWAY.md` — Railway deployment + custom domain guide
- `src/ufc_stats/` — UFC Stats API wrapper library
