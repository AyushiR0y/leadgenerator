# Deployment Guide - Render.com

## What I've Set Up

Your app is now configured for **single-service deployment** on Render.com. This means:
- ✅ FastAPI backend serves the API routes
- ✅ FastAPI also serves your `index.html` at the root URL
- ✅ One service = simpler deployment, no CORS issues
- ✅ Auto-detects localhost vs production

## Deployment Steps

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Ready for deployment"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Deploy on Render

1. Go to [render.com](https://render.com) and sign up (free)
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub repository
4. Render will detect `render.yaml` and configure everything automatically
5. Click **"Apply"**

### 3. Add Environment Variables

In the Render dashboard, go to your service and add these environment variables:

**Required:**
- `MAPPLS_CLIENT_ID` - Your MapmyIndia client ID
- `MAPPLS_CLIENT_SECRET` - Your MapmyIndia secret
- `AZURE_OPENAI_API_KEY` - Your Azure OpenAI key
- `AZURE_OPENAI_ENDPOINT` - Your Azure OpenAI endpoint
- `AZURE_OPENAI_DEPLOYMENT_NAME` - Your deployment name
- `GOOGLE_SEARCH_ENGINE_ID` - Your Google search engine ID
- `CONTACTOUT_API_KEY` - Your ContactOut API key

**Optional (has default):**
- `AZURE_OPENAI_API_VERSION` - Already set to `2024-02-15-preview`
- `GOOGLE_PLACES_API_KEY` - Enables Google Places for nearby business leads; if missing or denied, app falls back to OpenStreetMap data

### 4. Deploy

Render will automatically:
- Install dependencies from `requirements.txt`
- Upload your data files (`pincode.csv`, Excel files)
- Start the app with `uvicorn`
- Give you a URL like `https://leadgen-app.onrender.com`

## How It Works

**Local Development:**
- Backend runs on `http://127.0.0.1:8000`
- Frontend auto-detects localhost and uses it

**Production (Render):**
- Everything runs on one URL (e.g., `https://your-app.onrender.com`)
- Frontend auto-detects production and uses same origin
- No API URL configuration needed!

## Free Tier Limitations

⚠️ **Render Free Tier:**
- Sleeps after 15 minutes of inactivity
- First request after sleep takes ~30-60 seconds (cold start)
- 750 hours/month (plenty for testing)
- For always-on service, upgrade to paid ($7/month)

## Testing Locally After Changes

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the app
python -m uvicorn api:app --reload

# Visit http://127.0.0.1:8000
```

## Troubleshooting

**App won't start:**
- Check environment variables are set correctly in Render dashboard
- Check build logs for missing dependencies

**Data files missing:**
- Make sure `.csv` and `.xlsx` files are committed to git
- Check Render logs during build

**API calls failing:**
- Open browser console (F12) to see errors
- Check Render logs for API errors

## Alternative: Split Deployment

If you prefer separate frontend/backend:

**Backend (Render):**
- Deploy just the API

**Frontend (Netlify/Vercel):**
- Deploy just `index.html`
- Update API URL in the code

(Not recommended - adds complexity with CORS)
