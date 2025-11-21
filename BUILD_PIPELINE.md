# Build Pipeline Configuration

This project uses GitHub + Render/AWS App Runner as the build pipeline. The build process now includes frontend asset compilation.

## Build Steps

### 1. Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Node.js Dependencies & CSS Build
```bash
npm install
npm run build:css:prod
```

This compiles Tailwind CSS from `app/static/styles.css` → `app/static/tailwind.css`

## Deployment Platforms

### Render (`render.yaml`)
- **Build Command**: `pip install -r requirements.txt && npm install && npm run build:css:prod`
- Builds both Python and Node.js dependencies
- Compiles Tailwind CSS during build
- Serves static files from `app/static/`

### AWS App Runner (`apprunner.yaml`)
- **Pre-build**: Installs Python dependencies
- **Build**: Installs Node.js dependencies and compiles CSS
- Compiles Tailwind CSS during build
- Serves static files from `app/static/`

### AWS Amplify (`amplify.yml`)
- **Pre-build**: Installs Python and Node.js dependencies
- **Build**: Compiles CSS and verifies FastAPI app
- Compiles Tailwind CSS during build
- Serves static files from `app/static/`

## What Gets Built

### During Build:
1. ✅ Python dependencies installed
2. ✅ Node.js dependencies installed (Tailwind CSS, PostCSS, Autoprefixer)
3. ✅ Tailwind CSS compiled (`styles.css` → `tailwind.css`)
4. ✅ CSS minified for production

### What's NOT Built (Runtime Processing):
- React components are processed at runtime with Babel Standalone
- This is intentional - no build step needed for components
- Components load asynchronously via component-loader.js

## Industry Standard Status

### ✅ Now Industry Standard:
- **CSS Pre-compilation** - Tailwind CSS compiled during build
- **Build Pipeline** - Automated builds via GitHub → Render/App Runner
- **Production Optimization** - CSS minified in production

### ⚠️ Still Runtime Processing:
- **JSX Components** - Processed at runtime (pragmatic choice)
- This is fine for your use case - no build step needed for components

## Build Requirements

### Node.js
- Required for Tailwind CSS compilation
- Installed during build process
- Only needed for build, not runtime

### Python
- Required for FastAPI application
- Installed during build process
- Needed for runtime

## Local Development

### Build CSS Locally:
```bash
# Watch mode (development)
npm run build:css

# Production build
npm run build:css:prod
```

### No Build Needed For:
- Component development (runtime processing)
- HTML changes
- JavaScript changes (except CSS)

## CI/CD Flow

```
GitHub Push
    ↓
Render/App Runner/Amplify Detects Changes
    ↓
Build Phase:
  1. Install Python deps
  2. Install Node.js deps
  3. Compile Tailwind CSS
    ↓
Deploy Phase:
  1. Start FastAPI server
  2. Serve static files
    ↓
Runtime:
  1. Components load via component-loader.js
  2. Babel processes JSX at runtime
```

## Making It Fully Industry Standard

To make components pre-compiled (optional):

1. **Add Vite/Webpack** to build pipeline
2. **Bundle components** during build
3. **Remove Babel Standalone** from runtime
4. **Load pre-compiled bundle** instead

**Current approach is fine** for most use cases. The CSS is pre-compiled (industry standard), and component runtime processing is a pragmatic choice.

