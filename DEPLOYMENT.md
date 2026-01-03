# AWS Deployment Guide

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **Docker** installed locally
3. **AWS CLI** configured
4. **ECR Repository** created

## Step 1: Build Docker Image

```bash
# Navigate to project root
cd /path/to/Retail-Demand-Promotion-Intelligence

# Build Docker image
docker build -t retail-forecasting:latest .

# Test locally (optional)
docker run --rm retail-forecasting:latest
```

## Step 2: Push to AWS ECR

```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Create ECR repository (if not exists)
aws ecr create-repository --repository-name retail-forecasting --region us-east-1

# Tag image
docker tag retail-forecasting:latest \
  <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/retail-forecasting:latest

# Push image
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/retail-forecasting:latest
```

## Step 3: Setup AWS Batch

### 3.1 Create Compute Environment

```bash
aws batch create-compute-environment \
  --compute-environment-name retail-forecasting-env \
  --type MANAGED \
  --state ENABLED \
  --compute-resources type=EC2,minvCpus=0,maxvCpus=16,desiredvCpus=0,instanceTypes=c5.xlarge,subnets=<SUBNET_IDS>,securityGroupIds=<SG_IDS>,instanceRole=<ECS_INSTANCE_ROLE_ARN>
```

### 3.2 Create Job Queue

```bash
aws batch create-job-queue \
  --job-queue-name retail-forecasting-queue \
  --state ENABLED \
  --priority 1 \
  --compute-environment-order order=1,computeEnvironment=retail-forecasting-env
```

### 3.3 Register Job Definition

```bash
aws batch register-job-definition \
  --job-definition-name retail-forecasting-job \
  --type container \
  --container-properties '{
    "image": "<AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/retail-forecasting:latest",
    "vcpus": 4,
    "memory": 8192,
    "jobRoleArn": "<JOB_ROLE_ARN>",
    "executionRoleArn": "<EXECUTION_ROLE_ARN>",
    "environment": [
      {"name": "AWS_DEFAULT_REGION", "value": "us-east-1"}
    ],
    "mountPoints": [
      {
        "sourceVolume": "data",
        "containerPath": "/app/data",
        "readOnly": false
      }
    ],
    "volumes": [
      {
        "name": "data",
        "host": {"sourcePath": "/mnt/efs/retail-forecasting"}
      }
    ]
  }'
```

## Step 4: Submit Job

```bash
aws batch submit-job \
  --job-name retail-forecast-$(date +%Y%m%d) \
  --job-queue retail-forecasting-queue \
  --job-definition retail-forecasting-job
```

## Step 5: Monitor Job

```bash
# List jobs
aws batch list-jobs --job-queue retail-forecasting-queue --job-status RUNNING

# Describe job
aws batch describe-jobs --jobs <JOB_ID>

# View logs (CloudWatch)
aws logs tail /aws/batch/job --follow
```

## Step 6: Retrieve Results

Results are saved to `/app/data/model_output/final_forecasts/final_forecast.parquet` inside the container.

To persist results:
1. **Option A:** Mount EFS volume and read from there
2. **Option B:** Copy to S3 at end of pipeline:

```python
# Add to run_pipeline.py
import boto3
s3 = boto3.client('s3')
s3.upload_file(
    'data/model_output/final_forecasts/final_forecast.parquet',
    'your-bucket-name',
    f'forecasts/{date.today()}/final_forecast.parquet'
)
```

## Scheduled Execution (Optional)

Use **AWS EventBridge** to trigger jobs on a schedule:

```bash
aws events put-rule \
  --name retail-forecasting-daily \
  --schedule-expression "cron(0 2 * * ? *)"  # 2 AM daily

aws events put-targets \
  --rule retail-forecasting-daily \
  --targets "Id"="1","Arn"="<BATCH_JOB_QUEUE_ARN>","RoleArn"="<EVENTS_ROLE_ARN>","BatchParameters"={"JobDefinition"="retail-forecasting-job","JobName"="scheduled-forecast"}
```

## Cost Optimization

1. Use **Spot Instances** in Batch Compute Environment
2. Set `minvCpus=0` to scale down when idle
3. Use lifecycle policies to delete old CloudWatch logs
4. Store results in S3 with Intelligent-Tiering

## Troubleshooting

### Job fails with "ResourceInitializationError"
- Check ECS instance role has ECR pull permissions
- Verify security groups allow outbound traffic

### Out of Memory
- Increase `memory` in job definition
- Use larger instance types (e.g., `r5.xlarge`)

### Long training time
- Enable XGBoost GPU support: change `tree_method` to `gpu_hist`
- Use GPU-optimized instances (p3.2xlarge)

### Data not persisting
- Mount EFS volume to persist data across runs
- Or copy results to S3 at end of pipeline