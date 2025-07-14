# MLOps Troubleshooting Scenarios

## **Scenario 1: Production Model Performance Degradation**

### **Situation:**
Your fraud detection model, which has been running in production for 3 months with 95% accuracy, suddenly shows a drop to 78% accuracy over the past week. Customer complaints are increasing.

### **Investigation Steps:**
1. **Data Analysis:**
   - Check recent data for distribution shifts
   - Analyze feature importance changes
   - Look for missing or corrupted features
   - Compare current vs training data statistics

2. **Infrastructure Review:**
   - Check model serving logs for errors
   - Verify resource utilization (CPU, memory)
   - Review recent deployments or configuration changes
   - Check data pipeline health

3. **Model Diagnostics:**
   - Run model validation on recent data
   - Check prediction confidence distributions
   - Analyze error patterns by customer segment
   - Review A/B test configurations

### **Potential Root Causes:**
- **Data drift:** New fraud patterns not seen in training
- **Feature pipeline issues:** Upstream data source changes
- **Model staleness:** Concept drift in fraud behavior
- **Infrastructure problems:** Resource constraints affecting inference

### **Interview Questions:**
1. "How would you quickly identify if this is a data issue vs model issue?"
2. "What metrics would you implement to catch this earlier?"
3. "How do you decide between retraining vs rolling back?"
4. "What's your communication strategy during this incident?"

### **Expected Solutions:**
- Implement real-time drift detection
- Set up automated model validation pipelines
- Create rollback procedures
- Establish performance monitoring dashboards

---

## **Scenario 2: Kubernetes Pod Crashes in Production**

### **Situation:**
Your ML model serving pods in Kubernetes keep crashing with OOMKilled errors. The application was working fine last week, but now shows intermittent failures affecting 30% of requests.

### **Debugging Process:**

#### **Step 1: Immediate Assessment**
```bash
# Check pod status
kubectl get pods -n ml-production
kubectl describe pod <failing-pod> -n ml-production

# Check resource usage
kubectl top pods -n ml-production
kubectl top nodes

# Review recent events
kubectl get events -n ml-production --sort-by='.lastTimestamp'
```

#### **Step 2: Log Analysis**
```bash
# Check pod logs
kubectl logs <pod-name> -n ml-production --previous
kubectl logs <pod-name> -n ml-production --follow

# Check node logs
journalctl -u kubelet -n 50
```

#### **Step 3: Resource Investigation**
```bash
# Check resource limits and requests
kubectl get deployment ml-model -n ml-production -o yaml | grep -A 10 resources

# Monitor memory usage over time
kubectl exec -it <pod-name> -n ml-production -- cat /proc/meminfo
```

### **Root Cause Analysis:**
- **Memory leaks** in model inference code
- **Increased traffic** without proper resource scaling
- **Large model artifacts** loaded into memory
- **Inefficient data preprocessing** causing memory spikes
- **Resource limits** set too low for current workload

### **Interview Questions:**
1. "How do you differentiate between application issues and infrastructure issues?"
2. "What monitoring would you add to prevent this?"
3. "How do you handle this without affecting other services?"
4. "What's your strategy for capacity planning?"

---

## **Scenario 3: Data Pipeline Failure at 3 AM**

### **Situation:**
You receive a PagerDuty alert at 3 AM: "Daily ETL pipeline failed - no new training data available for model retraining scheduled at 6 AM."

### **Immediate Response Protocol:**

#### **Step 1: Assess Impact (5 minutes)**
```bash
# Check pipeline status
airflow dags list-runs -d daily_ml_pipeline --state failed

# Check data freshness
ls -la /data/ml_input/ | tail -10
aws s3 ls s3://ml-data-bucket/daily/ --recursive | tail -10

# Verify downstream systems
curl -f http://model-training-service:8080/health
```

#### **Step 2: Quick Diagnosis (10 minutes)**
```bash
# Check pipeline logs
airflow tasks failed-deps daily_ml_pipeline data_extraction

# Look for infrastructure issues
kubectl get pods -n data-pipeline
docker ps | grep pipeline

# Check external dependencies
nslookup database.internal.com
telnet api.external-source.com 443
```

#### **Step 3: Root Cause Categories**
1. **Data Source Issues:**
   - External API downtime
   - Database connectivity problems
   - Authentication/authorization failures
   - Schema changes in source systems

2. **Infrastructure Problems:**
   - Kubernetes node failures
   - Storage volume issues
   - Network connectivity problems
   - Resource exhaustion

3. **Code/Logic Errors:**
   - Recent pipeline code changes
   - Data validation failures
   - Dependency conflicts
   - Configuration drift

### **Mitigation Strategies:**
```bash
# Option 1: Use yesterday's data for training
cp /data/ml_input/2023-12-01/* /data/ml_input/2023-12-02/

# Option 2: Skip today's training, use existing model
kubectl patch cronjob ml-training -p '{"spec":{"suspend":true}}'

# Option 3: Run manual partial pipeline
airflow tasks run daily_ml_pipeline data_extraction 2023-12-02 --local
```

### **Interview Questions:**
1. "How do you prioritize investigation when multiple things could be wrong?"
2. "What's your decision framework for manual intervention vs automated recovery?"
3. "How do you communicate with stakeholders at 3 AM?"
4. "What post-incident processes do you follow?"

---

## **Scenario 4: Model Serving Latency Spike**

### **Situation:**
Model inference latency suddenly increased from 50ms to 2000ms. Load balancer health checks are failing, and the auto-scaler is creating more pods but they're not helping.

### **Investigation Framework:**

#### **Application Layer Analysis:**
```python
# Check model loading time
import time
start = time.time()
model = load_model('/app/models/model.pkl')
print(f"Model load time: {time.time() - start:.2f}s")

# Profile prediction function
import cProfile
cProfile.run('model.predict(sample_data)')

# Check memory usage during inference
import psutil
import gc
before = psutil.Process().memory_info().rss
prediction = model.predict(data)
after = psutil.Process().memory_info().rss
print(f"Memory used: {(after - before) / 1024 / 1024:.2f} MB")
```

#### **Infrastructure Layer Analysis:**
```bash
# Check container resource usage
docker stats <container-id>

# Monitor system resources
iostat -x 1 10
sar -u 1 10
sar -r 1 10

# Check network connectivity
traceroute model-service.internal.com
ss -tuln | grep :8000
```

#### **Database/Storage Investigation:**
```bash
# Check database connection pool
psql -h db-host -c "SELECT count(*) FROM pg_stat_activity;"

# Monitor I/O wait times
iotop -ao

# Check file system performance
time dd if=/dev/zero of=/tmp/test bs=1M count=100
```

### **Common Root Causes:**
1. **Model-Related:**
   - Large model size causing memory swapping
   - Inefficient inference code
   - Cold start issues with model loading
   - Feature engineering bottlenecks

2. **Infrastructure-Related:**
   - CPU throttling due to resource limits
   - Network latency to external services
   - Storage I/O bottlenecks
   - Memory pressure causing garbage collection

3. **Code-Related:**
   - Inefficient algorithms
   - Memory leaks
   - Blocking I/O operations
   - Poor connection pool management

### **Interview Questions:**
1. "How do you isolate the bottleneck in a complex microservices architecture?"
2. "What tools would you use to profile a production system safely?"
3. "How do you balance investigation speed vs system stability?"
4. "What preventive measures would you implement?"

---

## **Scenario 5: MLflow Experiment Tracking Down**

### **Situation:**
Data scientists can't access MLflow UI, experiments aren't being logged, and the model registry is unavailable. This is blocking the entire ML team.

### **Diagnostic Steps:**

#### **Service Health Check:**
```bash
# Check MLflow server status
curl -f http://mlflow-server:5000/health
systemctl status mlflow

# Check dependencies
pg_isready -h postgres-host -p 5432
aws s3 ls s3://mlflow-artifacts-bucket/

# Check container/pod status
kubectl get pods -l app=mlflow -n mlflow
docker ps | grep mlflow
```

#### **Database Investigation:**
```bash
# Check database connectivity
psql -h postgres-host -U mlflow -d mlflow -c "SELECT 1;"

# Check database space
psql -h postgres-host -U mlflow -d mlflow -c "
  SELECT pg_size_pretty(pg_database_size('mlflow'));
"

# Check connection limits
psql -h postgres-host -c "
  SELECT setting FROM pg_settings WHERE name = 'max_connections';
  SELECT count(*) FROM pg_stat_activity;
"
```

#### **Storage Backend Issues:**
```bash
# Check S3/MinIO connectivity
aws s3api head-bucket --bucket mlflow-artifacts-bucket
mc admin info minio-server

# Check storage space
df -h /mlflow-artifacts
```

### **Recovery Procedures:**

#### **Option 1: Database Recovery**
```bash
# Restart MLflow with database repair
mlflow server --backend-store-uri postgresql://user:pass@host/mlflow \
              --default-artifact-root s3://bucket \
              --host 0.0.0.0 \
              --port 5000 \
              --gunicorn-opts "--timeout 60"
```

#### **Option 2: Artifact Store Recovery**
```bash
# Fix S3 permissions
aws s3api put-bucket-policy --bucket mlflow-artifacts-bucket \
                          --policy file://bucket-policy.json

# Migrate to different storage
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root /tmp/mlflow-artifacts
```

### **Interview Questions:**
1. "How do you prioritize recovery vs root cause analysis?"
2. "What's your backup and disaster recovery strategy for ML metadata?"
3. "How do you communicate with frustrated data scientists during outage?"
4. "What monitoring would prevent this in the future?"

---

## **Scenario 6: Feature Store Inconsistency**

### **Situation:**
Model predictions are inconsistent between training and serving. Investigation reveals that the feature store is serving different values for the same features during training vs inference.

### **Investigation Process:**

#### **Data Lineage Tracking:**
```python
# Compare training vs serving features
training_features = get_features_from_training_data(user_id=12345, timestamp='2023-12-01')
serving_features = get_features_from_feature_store(user_id=12345, timestamp='2023-12-01')

# Check for differences
feature_diff = {}
for feature_name in training_features.keys():
    if training_features[feature_name] != serving_features[feature_name]:
        feature_diff[feature_name] = {
            'training': training_features[feature_name],
            'serving': serving_features[feature_name]
        }

print("Feature differences:", feature_diff)
```

#### **Pipeline Validation:**
```python
# Validate feature computation pipeline
def validate_feature_pipeline(feature_name, test_cases):
    for test_case in test_cases:
        expected = compute_feature_training_logic(test_case['input'])
        actual = compute_feature_serving_logic(test_case['input'])
        
        if abs(expected - actual) > 0.001:  # tolerance for floating point
            print(f"MISMATCH: {feature_name}")
            print(f"Expected: {expected}, Actual: {actual}")
            return False
    return True
```

#### **Schema Evolution Check:**
```sql
-- Check for schema changes
SELECT column_name, data_type, is_nullable
FROM information_schema.columns 
WHERE table_name = 'user_features'
ORDER BY ordinal_position;

-- Check for data type inconsistencies
SELECT feature_name, 
       COUNT(DISTINCT typeof(feature_value)) as type_count,
       array_agg(DISTINCT typeof(feature_value)) as types
FROM feature_store 
WHERE created_at > '2023-11-01'
GROUP BY feature_name
HAVING COUNT(DISTINCT typeof(feature_value)) > 1;
```

### **Common Causes:**
1. **Code Inconsistencies:**
   - Different feature computation logic between training and serving
   - Library version differences
   - Timezone handling issues
   - Rounding/precision differences

2. **Data Issues:**
   - Late-arriving data in batch vs real-time processing
   - Different data sources for training vs serving
   - Schema evolution without proper migration
   - Point-in-time correctness violations

3. **Infrastructure Issues:**
   - Clock synchronization problems
   - Caching inconsistencies
   - Database replication lag
   - Network partition effects

### **Interview Questions:**
1. "How do you ensure training/serving skew doesn't happen?"
2. "What testing strategy would catch this before production?"
3. "How do you handle schema evolution in feature stores?"
4. "What governance processes would prevent this?"

---

## **General Troubleshooting Best Practices**

### **Incident Response Framework:**
1. **Acknowledge (2 minutes):** Confirm the issue and its impact
2. **Assess (5 minutes):** Determine severity and affected systems
3. **Investigate (15 minutes):** Gather data and identify root cause
4. **Mitigate (30 minutes):** Implement temporary fixes
5. **Resolve (variable):** Apply permanent solution
6. **Review (post-incident):** Document learnings and improvements

### **Investigation Toolkit:**
```bash
# System monitoring
htop, iostat, sar, netstat, ss

# Container debugging  
kubectl, docker, crictl

# Application profiling
py-spy, pprof, jaeger, datadog

# Network troubleshooting
tcpdump, wireshark, nslookup, traceroute

# Database debugging
psql, mysql, redis-cli, mongo

# Log analysis
grep, awk, sed, jq, elk stack
```

### **Communication Templates:**

#### **Initial Incident Report:**
```
🚨 INCIDENT: Model Performance Degradation
SEVERITY: High
START TIME: 2023-12-02 14:30 UTC
IMPACT: 30% increase in prediction errors
INVESTIGATING: @alice @bob
STATUS: Root cause analysis in progress
NEXT UPDATE: 15 minutes
```

#### **Resolution Update:**
```
✅ RESOLVED: Model Performance Degradation
ROOT CAUSE: Data pipeline schema change
SOLUTION: Updated feature transformation logic
PREVENTION: Added schema validation tests
POST-MORTEM: Schedule for Monday team meeting
```

### **Interview Meta-Questions:**
1. "Walk me through your general troubleshooting methodology"
2. "How do you balance speed vs thoroughness during incidents?"
3. "What tools and commands do you always reach for first?"
4. "How do you learn from incidents to prevent future occurrences?"
5. "How do you handle pressure during critical production issues?"