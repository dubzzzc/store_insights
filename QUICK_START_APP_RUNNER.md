# Quick Start: Deploy to AWS App Runner

This is the fastest way to deploy your FastAPI application to AWS.

## Prerequisites

- AWS Account
- Repository connected to GitHub/GitLab/Bitbucket (or use AWS CodeCommit)
- RDS database accessible from App Runner

## Step-by-Step Deployment

### 1. Prepare Your Repository

Make sure these files are committed:
- ✅ `requirements.txt`
- ✅ `apprunner.yaml` (or we'll configure manually)
- ✅ All application code

### 2. Create App Runner Service

1. Go to [AWS App Runner Console](https://console.aws.amazon.com/apprunner/)
2. Click **"Create service"**
3. Choose **"Source code repository"**
4. Connect your repository:
   - Select your Git provider (GitHub, GitLab, Bitbucket)
   - Authorize AWS to access your repository
   - Select the repository: `store_insights`
   - Select branch: `main` (or your default branch)

### 3. Configure Build Settings

**Option A: Use apprunner.yaml (Recommended)**
- App Runner will auto-detect `apprunner.yaml`
- Edit the file to replace `REPLACE_WITH_YOUR_VALUE` placeholders with actual values
- Or override values in the console

**Option B: Manual Configuration**
- **Runtime**: Python 3
- **Build command**: `pip install --upgrade pip && pip install -r requirements.txt`
- **Start command**: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- **Port**: `8000`

### 4. Configure Environment Variables

In the App Runner configuration, add these environment variables:

**Required:**
```
CORE_DB_HOST=spirits-db.cbuumpmfxesr.us-east-1.rds.amazonaws.com
CORE_DB_USER=admin
CORE_DB_PASSWORD=your_actual_password
CORE_DB_NAME=platform_core
JWT_SECRET=your_secure_jwt_secret
```

**Optional (but recommended):**
```
ADMIN_API_KEY=your_admin_api_key
STORE_RDS_HOST=your_store_rds_host
STORE_RDS_ADMIN_USER=your_store_rds_admin_user
STORE_RDS_ADMIN_PASS=your_store_rds_admin_pass
STORE_RDS_PORT=3306
ALLOWED_ORIGINS=https://your-frontend-domain.com
```

**For sensitive values**, use AWS Secrets Manager:
1. Create secrets in AWS Secrets Manager
2. In App Runner, reference them as: `arn:aws:secretsmanager:region:account:secret:name`

### 5. Configure Service Settings

- **Service name**: `store-insights-api` (or your preferred name)
- **Virtual CPU**: 1 vCPU (start with this, scale up if needed)
- **Memory**: 2 GB (start with this, adjust based on usage)
- **Auto scaling**: Enable (recommended)
  - Min instances: 1
  - Max instances: 10 (adjust based on needs)

### 6. Configure Network (Important!)

**VPC Configuration:**
- If your RDS is in a VPC, configure App Runner to use the same VPC
- Go to "Networking" section
- Select your VPC, subnets, and security group
- Ensure the security group allows outbound connections to RDS

**If RDS is publicly accessible:**
- You can use default networking (no VPC configuration needed)
- Ensure RDS security group allows connections from App Runner

### 7. Deploy

1. Review all settings
2. Click **"Create & deploy"**
3. Wait for deployment (usually 5-10 minutes)
4. Note your service URL (e.g., `https://xxxxx.us-east-1.awsapprunner.com`)

### 8. Test Your Deployment

1. Visit your App Runner URL: `https://your-service-url.awsapprunner.com`
2. You should see: `{"message": "Spirit Store Insights API is running"}`
3. Test login endpoint: `https://your-service-url.awsapprunner.com/auth/login`

### 9. Update CORS (If Using Separate Frontend)

If you're hosting the frontend separately (e.g., on Amplify), update the `ALLOWED_ORIGINS` environment variable in App Runner:

```
ALLOWED_ORIGINS=https://your-amplify-app.amplifyapp.com,https://your-custom-domain.com
```

Then trigger a new deployment in App Runner.

## Cost Estimation

App Runner pricing (US East region):
- **Compute**: ~$0.007 per vCPU-hour, ~$0.0008 per GB-hour
- **Example**: 1 vCPU, 2 GB, running 24/7 = ~$15-20/month
- Much cheaper than always-on EC2 instances
- Scales to zero when not in use (if configured)

## Monitoring

- **CloudWatch Logs**: Automatically enabled
- **Metrics**: CPU, memory, requests in App Runner console
- **Alarms**: Set up CloudWatch alarms for errors or high latency

## Troubleshooting

### Service Won't Start
- Check CloudWatch logs for errors
- Verify all environment variables are set
- Check database connectivity

### Database Connection Issues
- Verify RDS security group allows App Runner IPs
- If using VPC, ensure VPC configuration is correct
- Check RDS endpoint and credentials

### High Latency
- Increase CPU/memory allocation
- Check database query performance
- Review CloudWatch metrics

## Next Steps

1. Set up a custom domain (optional)
2. Configure auto-scaling based on traffic
3. Set up CloudWatch alarms
4. Enable AWS WAF for additional security (optional)
5. Set up CI/CD for automatic deployments

## Support

- [App Runner Documentation](https://docs.aws.amazon.com/apprunner/)
- [App Runner Pricing](https://aws.amazon.com/apprunner/pricing/)

