# AWS Amplify Deployment Guide

This guide covers deploying the Store Insights FastAPI application to AWS Amplify.

## Important Note

**AWS Amplify Hosting is designed for static sites and cannot run long-running server processes like FastAPI.**

For a FastAPI Python backend, you have these options:

### Option 1: AWS App Runner (Recommended - Easiest)
Deploy the FastAPI backend to AWS App Runner. This is the simplest option for containerized Python applications and works well with Amplify for frontend hosting.

### Option 2: AWS Elastic Beanstalk
Traditional PaaS option, good for full control over the environment.

### Option 3: ECS/Fargate
More complex but gives you full container orchestration control.

### Option 4: Amplify Hosting (Static Files Only)
Use Amplify Hosting only for the static frontend files (`app/static`), and deploy the FastAPI backend separately to App Runner or another service.

**Recommendation**: Use AWS App Runner for the FastAPI backend and Amplify Hosting for static files (if needed separately).

## Environment Variables

Configure these environment variables in the Amplify Console:

### Required Variables

```
CORE_DB_HOST=spirits-db.cbuumpmfxesr.us-east-1.rds.amazonaws.com
CORE_DB_USER=admin
CORE_DB_PASSWORD=your_admin_password
CORE_DB_NAME=platform_core
JWT_SECRET=your_jwt_secret_key_here
```

### Optional Variables

```
ADMIN_API_KEY=your_admin_api_key_here
STORE_RDS_HOST=your_store_rds_host
STORE_RDS_ADMIN_USER=your_store_rds_admin_user
STORE_RDS_ADMIN_PASS=your_store_rds_admin_pass
STORE_RDS_PORT=3306
DEFAULT_ADMIN_EMAIL=admin@storeinsights.com
DEFAULT_ADMIN_PASSWORD=your_default_admin_password
DEFAULT_ADMIN_NAME=Store Insights Admin
DEFAULT_ADMIN_RESET_PASSWORD=false
```

### Performance Tuning (Optional)

```
STORE_INSIGHTS_MAX_CACHED_ENGINES=10
STORE_INSIGHTS_POOL_SIZE=10
STORE_INSIGHTS_POOL_MAX_OVERFLOW=10
STORE_INSIGHTS_POOL_TIMEOUT=30
STORE_INSIGHTS_POOL_RECYCLE=1200
STORE_INSIGHTS_CONNECT_TIMEOUT=10
STORE_INSIGHTS_QUERY_TIMEOUT=30
STORE_INSIGHTS_SLOW_QUERY_THRESHOLD=2.0
```

## Deployment Strategy

### Recommended: AWS App Runner for Backend

1. **Create App Runner Service**
   - Go to AWS App Runner Console
   - Click "Create service"
   - Choose "Source code repository" or "Container image"
   - If using source code:
     - Connect your repository
     - Select branch
     - App Runner will auto-detect `apprunner.yaml` or you can configure manually

2. **Configure App Runner**
   - Runtime: Python 3
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
   - Port: 8000

3. **Set Environment Variables in App Runner**
   - Go to Configuration → Environment variables
   - Add all required variables from the list above
   - **Important**: Use AWS Secrets Manager for sensitive values

4. **Get App Runner URL**
   - After deployment, note the service URL (e.g., `https://xxxxx.us-east-1.awsapprunner.com`)

### Optional: Amplify Hosting for Static Files

If you want to serve static files separately:

1. **Connect Repository to Amplify**
   - Go to AWS Amplify Console
   - Click "New app" → "Host web app"
   - Connect your repository
   - Select the branch

2. **Configure Build Settings**
   - Use the `amplify.yml` file (for static files only)
   - Or configure to copy `app/static` files to the build output

3. **Add Environment Variables**
   - Go to "Environment variables" in the app settings
   - Add `ALLOWED_ORIGINS` with your App Runner URL

4. **Configure Rewrites**
   - Point API calls to your App Runner service URL
   - Or use Amplify's proxy feature

## Alternative: AWS App Runner Deployment

For better performance and scalability, consider deploying the FastAPI backend to AWS App Runner:

### Create `apprunner.yaml` (if using App Runner)

```yaml
version: 1.0
runtime: python3
build:
  commands:
    build:
      - pip install -r requirements.txt
run:
  runtime-version: 3.11
  command: uvicorn app.main:app --host 0.0.0.0 --port 8000
  network:
    port: 8000
    env: PORT
  env:
    - name: CORE_DB_HOST
      value: spirits-db.cbuumpmfxesr.us-east-1.rds.amazonaws.com
    - name: CORE_DB_USER
      value: admin
    - name: CORE_DB_NAME
      value: platform_core
```

### App Runner Setup Steps

1. Create an App Runner service in AWS Console
2. Connect to your source repository
3. Configure build and runtime settings
4. Set environment variables
5. Deploy

## CORS Configuration

CORS is already configured in `app/main.py` to accept origins from the `ALLOWED_ORIGINS` environment variable.

**In App Runner**, set the environment variable:
```
ALLOWED_ORIGINS=https://your-amplify-domain.amplifyapp.com,https://your-custom-domain.com
```

This allows your Amplify-hosted frontend (if using Amplify) or any other frontend to make API calls to your App Runner backend.

## Database Security

Ensure your RDS instance security group allows connections from:
- Amplify Hosting IP ranges (if using Amplify)
- App Runner service security group (if using App Runner)
- Your VPC (if using VPC connectivity)

## Monitoring and Logs

- **Amplify**: Check build logs and runtime logs in Amplify Console
- **CloudWatch**: If using App Runner, logs automatically go to CloudWatch
- **Application Logs**: FastAPI logs will appear in the respective service logs

## Troubleshooting

### Build Failures
- Check Python version (Amplify uses Python 3.11 by default)
- Verify all dependencies in `requirements.txt` are compatible
- Check build logs for specific error messages

### Runtime Issues
- Verify all environment variables are set correctly
- Check database connectivity from Amplify/App Runner
- Review application logs for errors

### Static Files Not Serving
- Verify `app/static` directory structure
- Check Amplify rewrites/redirects configuration
- Ensure FastAPI static file mounting is correct

## Next Steps

1. Set up environment variables in Amplify Console
2. Update CORS origins in `app/main.py` with your Amplify domain
3. Configure database security groups
4. Deploy and test
5. Monitor logs and performance

