# Deploy Fight Lab to Railway

Step-by-step guide to deploy the Streamlit app on Railway and add `stats.mmasquared.com`.

---

## Prerequisites

- GitHub account
- Railway account (railway.app — sign up with GitHub)
- Code pushed to a GitHub repo
- Domain (mmasquared.com) on Cloudflare or your DNS provider

---

## Step 1: Create a Railway project

1. Go to [railway.app](https://railway.app) and sign in with GitHub.
2. Click **"New Project"**.
3. Choose **"Deploy from GitHub repo"**.
4. Pick your repo (mma-squared/fight-lab, or wherever fight-lab lives).
5. Railway will detect it as a Python project and start a build.

---

## Step 2: Set the root directory (if needed)

If `fight-lab` is inside a repo like `mma-squared`:

1. Open your service → **Settings**.
2. Under **Build**, set **Root Directory** to `fight-lab` (the folder with `app.py`).
3. Redeploy if the build used the wrong directory.

---

## Step 3: Add environment variables

1. Open your service → **Variables**.
2. Add:

| Variable              | Value                          | Notes                                      |
|-----------------------|--------------------------------|--------------------------------------------|
| `SUPABASE_URL`        | `https://your-project.supabase.co` | Supabase project URL                       |
| `SUPABASE_SERVICE_KEY`| `your-service-role-key`        | Supabase service role key                  |
| `SUPABASE_CACHE_BUCKET` | `fighter-cache` (optional)   | Only if you use a different bucket name    |

You can add them one by one or paste from `.env` (Railway supports bulk import).

---

## Step 4: Configure the start command

1. Service → **Settings**.
2. Under **Deploy**, find **Start Command**.
3. Set it to:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

If Railway picks up the Procfile, this may already be set. Otherwise set it manually.

---

## Step 5: Add a public URL

1. Service → **Settings** → **Networking** (or **Networking** tab).
2. Click **Generate Domain**.
3. Railway will give you a URL like `fight-lab-production-xxxx.up.railway.app`.
4. Open it to test the app.

---

## Step 6: Add custom domain `stats.mmasquared.com`

1. In **Networking**, click **Custom Domain** (or **Add domain**).
2. Enter: `stats.mmasquared.com`
3. Railway will show the DNS target, e.g.:

   ```
   CNAME: stats.mmasquared.com → xxxx.up.railway.app
   ```

   or they may show:

   ```
   CNAME: stats → xxxx.up.railway.app
   ```

   Use the exact target they display.

---

## Step 7: Configure DNS

### Cloudflare

1. Cloudflare Dashboard → **DNS** → **Records**.
2. Add or edit:
   - **Type:** CNAME  
   - **Name:** `stats` (or `stats.mmasquared.com` depending on your setup)
   - **Target:** the Railway domain (e.g. `your-app.up.railway.app`)
   - **Proxy status:** Proxied (orange cloud) or DNS only (grey cloud)
3. Save.

### Other DNS providers

Add the same CNAME record using the Railway target.

---

## Step 8: Wait for SSL

- Railway issues an SSL certificate for your custom domain.
- Propagation usually takes 5–30 minutes, sometimes up to 24 hours.
- After DNS and SSL are ready, `https://stats.mmasquared.com` should load.

---

## Troubleshooting

**Build fails**

- Ensure `requirements.txt` and `Procfile` exist in the deploy root.
- Check build logs in Railway for dependency or Python version errors.

**App loads but “Compare” doesn’t work**

- Verify Supabase variables and cache bucket.
- Check Railway logs for HTTP 5xx or Supabase/connectivity errors.

**Custom domain shows “SSL error”**

- Wait for certificate provisioning.
- In Cloudflare, try switching proxy off temporarily to rule out conflicts.
- Ensure CNAME matches exactly what Railway shows.

**Wrong repo or subfolder**

- Double‑check repo and **Root Directory** in the Railway service settings.
