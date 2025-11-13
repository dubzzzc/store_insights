# store_insights

Store Insights dashboard for Spirits stores - Web app with native iOS and Android support.

## Features

- FastAPI backend with React frontend
- Progressive Web App (PWA) support
- Native iOS and Android apps via Capacitor
- Camera integration for mobile devices
- Store analytics and insights dashboard

## Mobile App Deployment

This app is configured for deployment as native iOS and Android apps using Capacitor. See [MOBILE_APP_DEPLOYMENT.md](MOBILE_APP_DEPLOYMENT.md) for detailed instructions.

### Quick Start for Mobile Apps

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Sync native projects:**
   ```bash
   npm run cap:sync
   ```

3. **Open in native IDEs:**
   ```bash
   npm run cap:ios      # Opens Xcode (macOS only)
   npm run cap:android  # Opens Android Studio
   ```

## Development

### Backend (FastAPI)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload
```

### Frontend

The frontend consists of static HTML files with React loaded via CDN. No build process required.

## Byte-compiling the uploader script

The `python -m compileall` helper expects paths relative to your current
working directory. Run the command from the repository root so the script is
found correctly:

```
python -m compileall scripts/vfp_dbf_to_rdsv2.py
```

If you are already inside the `scripts/` directory, drop the leading folder
and compile the file directly:

```
python -m compileall vfp_dbf_to_rdsv2.py
```

Both invocations point to the same file; the key difference is the relative
path supplied to `compileall`.

