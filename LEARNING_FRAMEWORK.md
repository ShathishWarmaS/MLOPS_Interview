# 📚 MLOps Interview Learning Framework
## Step-by-Step Task + Learning Plan

---

## 🎯 **WEEK 1: MLOps Foundations**

### **DAY 1: MLOps Pipeline Concepts**
#### **📖 Learning Phase (2 hours)**
**Topics to Study:**
- What is MLOps and why it matters
- ML lifecycle: Data → Model → Deployment → Monitoring
- Difference between DevOps and MLOps
- Key MLOps tools overview

**Resources:**
```bash
# Read these sections in 01_mlops_pipeline_exercises.md:
- Exercise 1: E-commerce Recommendation System
- Focus on architecture components
- Understand data flow patterns
```

#### **✅ Tasks to Complete (1 hour)**
1. **Task 1.1:** Draw a simple ML pipeline diagram
   - Include: Data Ingestion → Processing → Training → Deployment → Monitoring
   - Write 2-3 sentences explaining each component

2. **Task 1.2:** Answer these questions:
   - What happens when a model's performance drops in production?
   - How is ML deployment different from web application deployment?
   - Name 3 challenges unique to ML systems

#### **🔍 Interview Prep (30 minutes)**
**Practice Question:** "Explain the MLOps lifecycle to a non-technical stakeholder"
- Practice 5-minute explanation
- Use simple analogies
- Focus on business value

---

### **DAY 2: MLflow Basics**
#### **📖 Learning Phase (2 hours)**
**Topics to Study:**
- MLflow components: Tracking, Projects, Models, Registry
- Experiment tracking best practices
- Model versioning concepts

**Hands-on Reading:**
```python
# Study these sections in 02_mlflow_coding_challenges.py:
- Challenge 1: Multi-Algorithm Experiment Tracking
- Challenge 2: Model Registry Management
- Read all TODO comments and understand what needs implementation
```

#### **✅ Tasks to Complete (1.5 hours)**
1. **Task 2.1:** Set up MLflow locally
```bash
pip install mlflow
mlflow ui
```

2. **Task 2.2:** Create your first experiment
```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Create simple experiment
mlflow.set_experiment("my_first_experiment")

with mlflow.start_run():
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=4)
    
    # Train model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Log parameters and metrics
    mlflow.log_param("solver", "liblinear")
    mlflow.log_metric("accuracy", 0.95)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")

print("✅ First MLflow experiment completed!")
```

3. **Task 2.3:** Explore MLflow UI
   - Open http://localhost:5000
   - Navigate through your experiment
   - Take screenshots of different views

#### **🔍 Interview Prep (30 minutes)**
**Practice Questions:**
- "How would you track experiments for a team of 10 data scientists?"
- "What information should be logged for each ML experiment?"

---

### **DAY 3: Docker Fundamentals**
#### **📖 Learning Phase (2 hours)**
**Topics to Study:**
- Docker basics: Images, containers, Dockerfile
- Multi-stage builds for ML models
- Docker best practices for ML

**Study Material:**
```dockerfile
# Read and understand docker_k8s_scenarios/Dockerfile.ml-model
# Focus on:
- Multi-stage build benefits
- Security considerations (non-root user)
- Health checks
- Environment variables
```

#### **✅ Tasks to Complete (2 hours)**
1. **Task 3.1:** Create your first ML Docker image
```dockerfile
# Create Dockerfile.simple
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model code
COPY model_server.py .

EXPOSE 8000

CMD ["python", "model_server.py"]
```

2. **Task 3.2:** Create a simple model server
```python
# Create model_server.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model (create a dummy one for now)
model = {"type": "dummy", "version": "1.0"}

@app.route('/health')
def health():
    return {"status": "healthy", "model": model}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Dummy prediction
    prediction = {"result": 0.85, "confidence": 0.92}
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

3. **Task 3.3:** Build and test Docker image
```bash
# Create requirements.txt
echo "flask==2.0.1" > requirements.txt

# Build image
docker build -t my-ml-model:v1 -f Dockerfile.simple .

# Run container
docker run -p 8000:8000 my-ml-model:v1

# Test in another terminal
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": [1,2,3,4]}'
```

#### **🔍 Interview Prep (30 minutes)**
**Practice Questions:**
- "Why use Docker for ML models?"
- "How do you optimize Docker images for production?"

---

### **DAY 4: Kubernetes Basics**
#### **📖 Learning Phase (2 hours)**
**Topics to Study:**
- Kubernetes core concepts: Pods, Deployments, Services
- Scaling and load balancing
- Health checks and rolling updates

**Study Material:**
```yaml
# Read and understand k8s-ml-deployment.yaml
# Focus on:
- Deployment configuration
- Resource limits and requests
- Health checks (liveness/readiness probes)
- Auto-scaling configuration
```

#### **✅ Tasks to Complete (2 hours)**
1. **Task 4.1:** Set up local Kubernetes
```bash
# Option 1: Using minikube
minikube start

# Option 2: Using Docker Desktop (enable Kubernetes)
# Or use kind/k3s if you prefer
```

2. **Task 4.2:** Create simple Kubernetes deployment
```yaml
# Create simple-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-simple
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: model
        image: my-ml-model:v1
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

3. **Task 4.3:** Deploy and test
```bash
# Deploy to Kubernetes
kubectl apply -f simple-deployment.yaml

# Check pods
kubectl get pods

# Check service
kubectl get services

# Test the service
kubectl port-forward service/ml-model-service 8080:80
curl http://localhost:8080/health
```

#### **🔍 Interview Prep (30 minutes)**
**Practice Questions:**
- "How do you ensure high availability for ML models in Kubernetes?"
- "What happens when a pod crashes?"

---

### **DAY 5: Python Automation Basics**
#### **📖 Learning Phase (2 hours)**
**Topics to Study:**
- Automation scripts for ML workflows
- Error handling and logging
- Configuration management

**Study Material:**
```python
# Read these sections in 05_python_automation_challenges.py:
- Challenge 1: Model Deployment Automation (ModelDeploymentAutomator class)
- Focus on the structure and TODO comments
- Understand the workflow: Build → Push → Deploy → Health Check
```

#### **✅ Tasks to Complete (2 hours)**
1. **Task 5.1:** Create a simple deployment automation script
```python
# Create deploy_automation.py
import subprocess
import time
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDeployment:
    def __init__(self, image_name, version):
        self.image_name = image_name
        self.version = version
        self.image_tag = f"{image_name}:{version}"
    
    def build_image(self):
        logger.info(f"Building image: {self.image_tag}")
        result = subprocess.run([
            "docker", "build", "-t", self.image_tag, "."
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Image built successfully")
            return True
        else:
            logger.error(f"❌ Build failed: {result.stderr}")
            return False
    
    def run_container(self):
        logger.info(f"Running container: {self.image_tag}")
        result = subprocess.run([
            "docker", "run", "-d", "-p", "8000:8000", 
            "--name", f"ml-model-{self.version}", self.image_tag
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            container_id = result.stdout.strip()
            logger.info(f"✅ Container started: {container_id}")
            return True
        else:
            logger.error(f"❌ Container start failed: {result.stderr}")
            return False
    
    def health_check(self, max_retries=10):
        logger.info("Performing health check...")
        
        for attempt in range(max_retries):
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Health check passed")
                    return True
            except requests.exceptions.RequestException:
                logger.info(f"Health check attempt {attempt + 1} failed, retrying...")
                time.sleep(5)
        
        logger.error("❌ Health check failed after all retries")
        return False
    
    def deploy(self):
        logger.info(f"Starting deployment for {self.image_tag}")
        
        # Step 1: Build
        if not self.build_image():
            return False
        
        # Step 2: Run
        if not self.run_container():
            return False
        
        # Step 3: Health Check
        if not self.health_check():
            return False
        
        logger.info("🎉 Deployment completed successfully!")
        return True

# Usage
if __name__ == "__main__":
    deployment = SimpleDeployment("my-ml-model", "v1")
    deployment.deploy()
```

2. **Task 5.2:** Test the automation script
```bash
python deploy_automation.py
```

3. **Task 5.3:** Add error handling and rollback
```python
# Add this method to SimpleDeployment class
def rollback(self):
    logger.info("Performing rollback...")
    subprocess.run(["docker", "stop", f"ml-model-{self.version}"])
    subprocess.run(["docker", "rm", f"ml-model-{self.version}"])
    logger.info("✅ Rollback completed")
```

#### **🔍 Interview Prep (30 minutes)**
**Practice Questions:**
- "How do you handle failures in automated deployment?"
- "What would you log during deployment for debugging?"

---

### **DAY 6: Shell Scripting for MLOps**
#### **📖 Learning Phase (1.5 hours)**
**Topics to Study:**
- Shell scripting best practices
- Error handling with `set -euo pipefail`
- Logging and monitoring scripts

**Study Material:**
```bash
# Read shell_scripts/01_deployment_automation.sh
# Focus on:
- Script structure and organization
- Configuration management
- Error handling patterns
- Logging functions
```

#### **✅ Tasks to Complete (2 hours)**
1. **Task 6.1:** Create a simple monitoring script
```bash
# Create monitor.sh
#!/bin/bash

set -euo pipefail

# Configuration
LOG_FILE="/tmp/ml_monitor.log"
SERVICE_URL="http://localhost:8000"

# Logging functions
log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "$LOG_FILE"
}

# Health check function
check_service_health() {
    local url="$1"
    
    if curl -s -f --max-time 10 "$url/health" > /dev/null; then
        log_info "✅ Service is healthy"
        return 0
    else
        log_error "❌ Service health check failed"
        return 1
    fi
}

# System metrics
check_system_metrics() {
    local cpu_usage
    local memory_usage
    
    # Get CPU usage (simplified)
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}' || echo "unknown")
    
    # Get memory usage
    memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}' || echo "unknown")
    
    log_info "System metrics - CPU: ${cpu_usage}%, Memory: ${memory_usage}%"
}

# Main monitoring loop
main() {
    log_info "Starting ML service monitoring..."
    
    while true; do
        check_service_health "$SERVICE_URL"
        check_system_metrics
        
        echo "Sleeping for 30 seconds..."
        sleep 30
    done
}

# Run main function
main "$@"
```

2. **Task 6.2:** Make it executable and test
```bash
chmod +x monitor.sh
./monitor.sh
```

3. **Task 6.3:** Create a deployment script
```bash
# Create deploy.sh
#!/bin/bash

set -euo pipefail

VERSION="${1:-v1}"
IMAGE_NAME="my-ml-model"

log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

deploy_model() {
    local version="$1"
    
    log_info "🚀 Starting deployment of version: $version"
    
    # Build Docker image
    log_info "Building Docker image..."
    if docker build -t "${IMAGE_NAME}:${version}" .; then
        log_info "✅ Docker build successful"
    else
        log_info "❌ Docker build failed"
        exit 1
    fi
    
    # Stop existing container
    log_info "Stopping existing container..."
    docker stop "ml-model-current" 2>/dev/null || true
    docker rm "ml-model-current" 2>/dev/null || true
    
    # Start new container
    log_info "Starting new container..."
    if docker run -d --name "ml-model-current" -p 8000:8000 "${IMAGE_NAME}:${version}"; then
        log_info "✅ Container started successfully"
    else
        log_info "❌ Container start failed"
        exit 1
    fi
    
    log_info "🎉 Deployment completed for version: $version"
}

# Usage
deploy_model "$VERSION"
```

#### **🔍 Interview Prep (30 minutes)**
**Practice Questions:**
- "How do you make shell scripts robust and maintainable?"
- "What's your approach to error handling in deployment scripts?"

---

### **DAY 7: Week 1 Review & Integration**
#### **📖 Review Phase (2 hours)**
**Review all concepts from Week 1:**
- MLOps pipeline concepts
- MLflow experiment tracking
- Docker containerization
- Kubernetes deployment
- Python automation
- Shell scripting

#### **✅ Integration Tasks (2 hours)**
1. **Task 7.1:** Create end-to-end workflow
```bash
# Combine everything into one workflow
./deploy.sh v2      # Deploy new version
./monitor.sh &      # Start monitoring
kubectl get pods    # Check Kubernetes status
```

2. **Task 7.2:** Document your learning
```markdown
# Week 1 Learning Summary

## What I Learned:
1. MLOps vs DevOps differences
2. MLflow experiment tracking basics
3. Docker for ML model deployment
4. Kubernetes fundamentals
5. Python automation scripts
6. Shell scripting for MLOps

## Skills Gained:
- Can set up MLflow experiments
- Can containerize ML models
- Can deploy to Kubernetes
- Can write automation scripts
- Can create monitoring systems

## Questions I Can Answer:
- What is MLOps and why is it important?
- How do you track ML experiments?
- How do you deploy ML models at scale?
- How do you automate ML workflows?
```

#### **🔍 Week 1 Mock Interview (1 hour)**
**Practice these questions:**
1. "Walk me through deploying an ML model from code to production"
2. "How would you track experiments for a team?"
3. "What happens when your model container crashes?"
4. "How do you monitor ML models in production?"

---

## 🎯 **WEEK 2: Advanced MLOps**

### **DAY 8: Advanced MLflow**
#### **📖 Learning Phase (2 hours)**
**Topics to Study:**
- Model registry workflows
- MLflow deployment patterns
- Custom model wrappers

**Study Material:**
```python
# Complete Challenge 2 from 02_mlflow_coding_challenges.py
# Focus on: Model Registry Management
# Implement the actual functions
```

#### **✅ Tasks to Complete (2 hours)**
1. **Task 8.1:** Implement model registry workflow
```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register a model
def register_model_version():
    model_name = "fraud_detection_model"
    
    # Assuming you have a trained model
    with mlflow.start_run():
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=model_name
        )
        
        run_id = mlflow.active_run().info.run_id
        
    # Get model version
    model_version = client.get_latest_versions(
        model_name, 
        stages=["None"]
    )[0]
    
    print(f"Model registered: {model_name} version {model_version.version}")
    return model_version

# Transition model to staging
def promote_to_staging(model_name, version):
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging"
    )
    print(f"Model {model_name} v{version} promoted to Staging")

# Transition to production
def promote_to_production(model_name, version):
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    print(f"Model {model_name} v{version} promoted to Production")
```

2. **Task 8.2:** Create model comparison script
```python
def compare_model_versions(model_name):
    versions = client.search_model_versions(f"name='{model_name}'")
    
    print(f"Comparing versions for {model_name}:")
    for version in versions:
        print(f"Version {version.version}: Stage={version.current_stage}")
        
        # Get run metrics
        run = client.get_run(version.run_id)
        metrics = run.data.metrics
        print(f"  Metrics: {metrics}")
```

#### **🔍 Interview Prep (30 minutes)**
**Practice Questions:**
- "How do you manage model versions in a team environment?"
- "What's your model promotion strategy?"

---

### **DAY 9: Google Cloud Platform - Part 1**
#### **📖 Learning Phase (2 hours)**
**Topics to Study:**
- Vertex AI overview
- Cloud ML services landscape
- Training jobs on GCP

**Study Material:**
```python
# Study 04_gcp_ml_integration.py
# Focus on Exercise 1: Vertex AI Custom Training Job
# Understand the concepts even if you can't run it
```

#### **✅ Tasks to Complete (2 hours)**
1. **Task 9.1:** Create GCP account and explore services
```bash
# If you have access to GCP:
# 1. Create a new project
# 2. Enable Vertex AI API
# 3. Explore the Vertex AI console
```

2. **Task 9.2:** Design a training job configuration
```python
# Create training_config.py
training_job_config = {
    "display_name": "fraud-detection-training",
    "python_package_gcs_uri": "gs://your-bucket/trainer.tar.gz",
    "python_module": "trainer.task",
    "container_uri": "gcr.io/cloud-aiplatform/training/pytorch-gpu.1-9:latest",
    "args": [
        "--epochs=100",
        "--batch-size=32",
        "--learning-rate=0.001",
        "--model-dir=gs://your-bucket/models/"
    ],
    "machine_type": "n1-standard-4",
    "accelerator_type": "NVIDIA_TESLA_T4",
    "accelerator_count": 1,
    "replica_count": 1
}

def create_training_job(config):
    """
    Pseudocode for creating training job
    """
    print("Creating training job with config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # In real implementation:
    # aiplatform.CustomTrainingJob.from_local_script(...)
    
    print("Training job would be submitted to Vertex AI")
```

3. **Task 9.3:** Design model serving configuration
```python
# Create serving_config.py
serving_config = {
    "model_display_name": "fraud-detection-model",
    "artifact_uri": "gs://your-bucket/model-artifacts/",
    "serving_container_image_uri": "gcr.io/cloud-aiplatform/prediction/sklearn-cpu.0-24:latest",
    "endpoint_display_name": "fraud-detection-endpoint",
    "machine_type": "n1-standard-2",
    "min_replica_count": 1,
    "max_replica_count": 10,
    "traffic_percentage": 100
}

def deploy_model_to_endpoint(config):
    """
    Pseudocode for model deployment
    """
    print("Deploying model with config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # In real implementation:
    # model = aiplatform.Model.upload(...)
    # endpoint = aiplatform.Endpoint.create(...)
    # model.deploy(endpoint=endpoint, ...)
    
    print("Model would be deployed to Vertex AI endpoint")
```

#### **🔍 Interview Prep (30 minutes)**
**Practice Questions:**
- "How would you choose between different GCP ML services?"
- "What are the benefits of managed ML platforms?"

---

### **DAY 10: Advanced Kubernetes**
#### **📖 Learning Phase (2 hours)**
**Topics to Study:**
- Advanced Kubernetes concepts for ML
- Auto-scaling and resource management
- Monitoring and logging

**Study Material:**
```yaml
# Study k8s-ml-deployment.yaml in detail
# Focus on:
- HorizontalPodAutoscaler configuration
- Resource requests and limits
- Pod disruption budgets
- Health checks and probes
```

#### **✅ Tasks to Complete (2 hours)**
1. **Task 10.1:** Implement auto-scaling
```yaml
# Create autoscaling.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

2. **Task 10.2:** Create monitoring configuration
```yaml
# Create monitoring.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: monitoring-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'ml-models'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
```

3. **Task 10.3:** Test scaling behavior
```bash
# Apply configurations
kubectl apply -f autoscaling.yaml
kubectl apply -f monitoring.yaml

# Generate load to test scaling
kubectl run -i --tty load-generator --rm --image=busybox --restart=Never -- /bin/sh

# Inside the load generator:
while true; do wget -q -O- http://ml-model-service/predict; done
```

#### **🔍 Interview Prep (30 minutes)**
**Practice Questions:**
- "How do you handle traffic spikes for ML models?"
- "What metrics would you use for auto-scaling ML services?"

---

### **DAY 11: Advanced Python Automation**
#### **📖 Learning Phase (2 hours)**
**Topics to Study:**
- Advanced automation patterns
- Error handling and retries
- Configuration management

**Study Material:**
```python
# Study Challenge 3 and 4 from 05_python_automation_challenges.py
# Focus on:
- Model Performance Tracking
- Resource Management Automation
```

#### **✅ Tasks to Complete (2.5 hours)**
1. **Task 11.1:** Implement performance monitoring
```python
# Create performance_monitor.py
import time
import requests
import sqlite3
from datetime import datetime
from typing import Dict, List

class ModelPerformanceMonitor:
    def __init__(self, db_path: str = "performance.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                metric_name TEXT,
                metric_value REAL,
                model_version TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_metric(self, metric_name: str, value: float, model_version: str = "current"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_metrics (timestamp, metric_name, metric_value, model_version)
            VALUES (?, ?, ?, ?)
        """, (datetime.now(), metric_name, value, model_version))
        
        conn.commit()
        conn.close()
    
    def check_model_latency(self, endpoint: str) -> float:
        start_time = time.time()
        try:
            response = requests.post(
                f"{endpoint}/predict",
                json={"features": [1, 2, 3, 4]},
                timeout=30
            )
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            if response.status_code == 200:
                self.record_metric("latency_ms", latency)
                self.record_metric("availability", 1.0)
                return latency
            else:
                self.record_metric("availability", 0.0)
                return -1
                
        except Exception as e:
            self.record_metric("availability", 0.0)
            print(f"Health check failed: {e}")
            return -1
    
    def get_recent_metrics(self, metric_name: str, hours: int = 24) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, metric_value FROM performance_metrics
            WHERE metric_name = ? AND timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        """.format(hours), (metric_name,))
        
        results = [{"timestamp": row[0], "value": row[1]} for row in cursor.fetchall()]
        conn.close()
        
        return results
    
    def calculate_sla_compliance(self, latency_threshold: float = 200.0) -> float:
        recent_latencies = self.get_recent_metrics("latency_ms", 24)
        
        if not recent_latencies:
            return 0.0
        
        compliant_requests = sum(1 for metric in recent_latencies if metric["value"] <= latency_threshold)
        total_requests = len(recent_latencies)
        
        return (compliant_requests / total_requests) * 100

# Usage
monitor = ModelPerformanceMonitor()

# Monitor continuously
while True:
    latency = monitor.check_model_latency("http://localhost:8000")
    if latency > 0:
        print(f"Model latency: {latency:.2f}ms")
    
    sla_compliance = monitor.calculate_sla_compliance()
    print(f"SLA compliance (24h): {sla_compliance:.1f}%")
    
    time.sleep(60)  # Check every minute
```

2. **Task 11.2:** Create alerting system
```python
# Create alerting.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertManager:
    def __init__(self, smtp_server: str = "localhost", smtp_port: int = 587):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.alert_thresholds = {
            "latency_ms": 200.0,
            "availability": 0.95,
            "sla_compliance": 99.0
        }
    
    def check_thresholds(self, metrics: Dict[str, float]) -> List[str]:
        alerts = []
        
        for metric_name, value in metrics.items():
            threshold = self.alert_thresholds.get(metric_name)
            if threshold is None:
                continue
            
            if metric_name == "latency_ms" and value > threshold:
                alerts.append(f"High latency detected: {value:.2f}ms (threshold: {threshold}ms)")
            elif metric_name == "availability" and value < threshold:
                alerts.append(f"Low availability: {value:.2%} (threshold: {threshold:.2%})")
            elif metric_name == "sla_compliance" and value < threshold:
                alerts.append(f"SLA compliance below threshold: {value:.1f}% (threshold: {threshold}%)")
        
        return alerts
    
    def send_alert(self, message: str, recipient: str = "admin@company.com"):
        # In production, you'd use real SMTP configuration
        print(f"🚨 ALERT: {message}")
        print(f"📧 Would send email to: {recipient}")
        
        # Real implementation would be:
        # msg = MIMEText(message)
        # msg['Subject'] = 'ML Model Alert'
        # msg['From'] = 'alerts@company.com'
        # msg['To'] = recipient
        # 
        # server = smtplib.SMTP(self.smtp_server, self.smtp_port)
        # server.send_message(msg)
        # server.quit()
```

#### **🔍 Interview Prep (30 minutes)**
**Practice Questions:**
- "How do you monitor ML model performance in production?"
- "What would trigger an alert for your ML system?"

---

### **DAY 12: System Design Basics**
#### **📖 Learning Phase (2 hours)**
**Topics to Study:**
- System design principles for ML
- Scalability patterns
- Trade-offs in distributed systems

**Study Material:**
```markdown
# Read 08_system_design_questions.md
# Focus on Question 1: Real-Time Fraud Detection System
# Understand the components and data flow
```

#### **✅ Tasks to Complete (2 hours)**
1. **Task 12.1:** Design a simple recommendation system
```
Problem: Design a basic recommendation system for 1000 users and 10000 items

Your design should include:
1. Data collection layer
2. Model training pipeline  
3. Serving infrastructure
4. Monitoring system

Draw diagrams and explain your choices.
```

2. **Task 12.2:** Practice system design interview format
```
1. Clarify requirements (5 minutes)
   - How many users?
   - What type of recommendations?
   - Latency requirements?
   - Accuracy requirements?

2. High-level design (10 minutes)
   - Draw major components
   - Show data flow
   - Identify key services

3. Detailed design (20 minutes)
   - Database schema
   - API design
   - Algorithm choice
   - Caching strategy

4. Scale and optimize (10 minutes)
   - Bottlenecks
   - Scaling solutions
   - Monitoring

5. Handle edge cases (5 minutes)
   - Failure scenarios
   - Data quality issues
   - Cold start problem
```

#### **🔍 Interview Prep (1 hour)**
**Practice Question:** "Design a system to detect fraudulent transactions in real-time"
- Practice the 45-minute format
- Focus on ML-specific considerations
- Explain trade-offs clearly

---

### **DAY 13: Troubleshooting Skills**
#### **📖 Learning Phase (2 hours)**
**Topics to Study:**
- Production debugging techniques
- Common ML system failures
- Incident response procedures

**Study Material:**
```markdown
# Read 07_troubleshooting_scenarios.md
# Focus on:
- Scenario 1: Production Model Performance Degradation
- Scenario 2: Kubernetes Pod Crashes
# Understand the debugging methodology
```

#### **✅ Tasks to Complete (2 hours)**
1. **Task 13.1:** Create a debugging checklist
```markdown
# ML System Debugging Checklist

## When Model Performance Drops:
- [ ] Check recent data for distribution changes
- [ ] Compare feature statistics (training vs current)
- [ ] Review model serving logs for errors
- [ ] Check infrastructure resource usage
- [ ] Validate data pipeline health
- [ ] Review recent deployments/changes

## When Service is Down:
- [ ] Check pod/container status
- [ ] Review application logs
- [ ] Check resource limits and usage
- [ ] Verify network connectivity
- [ ] Check external dependencies
- [ ] Review recent configuration changes

## When Latency is High:
- [ ] Check model inference time
- [ ] Review database query performance
- [ ] Check network latency
- [ ] Monitor CPU/Memory usage
- [ ] Check for memory leaks
- [ ] Review caching effectiveness
```

2. **Task 13.2:** Practice incident response
```
Scenario: You receive an alert that model accuracy dropped from 95% to 78%

Your 30-minute response plan:
1. Minutes 0-5: Assess impact and gather initial data
2. Minutes 5-15: Investigate potential causes
3. Minutes 15-25: Implement mitigation
4. Minutes 25-30: Communicate status and next steps

Document your approach for each phase.
```

3. **Task 13.3:** Create diagnostic scripts
```bash
# Create debug_model.sh
#!/bin/bash

echo "🔍 ML Model Diagnostic Script"
echo "=============================="

# Check model service health
echo "1. Checking model service..."
curl -s http://localhost:8000/health || echo "❌ Service unreachable"

# Check system resources
echo "2. System resources:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')"
echo "Memory: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "Disk: $(df -h / | awk 'NR==2 {print $5}')"

# Check recent logs
echo "3. Recent error logs:"
docker logs ml-model-current --tail=10 | grep -i error || echo "No recent errors"

# Check model metrics
echo "4. Model performance:"
python -c "
import sqlite3
conn = sqlite3.connect('performance.db')
cursor = conn.cursor()
cursor.execute('SELECT AVG(metric_value) FROM performance_metrics WHERE metric_name=\"latency_ms\" AND timestamp > datetime(\"now\", \"-1 hour\")')
avg_latency = cursor.fetchone()[0]
print(f'Average latency (1h): {avg_latency:.2f}ms' if avg_latency else 'No recent data')
"
```

#### **🔍 Interview Prep (1 hour)**
**Practice Questions:**
- "Walk me through debugging a model performance issue"
- "How do you prioritize investigation when multiple alerts fire?"

---

### **DAY 14: Week 2 Review & Advanced Integration**
#### **📖 Review Phase (2 hours)**
**Review all Week 2 concepts:**
- Advanced MLflow workflows
- GCP ML services
- Advanced Kubernetes
- Performance monitoring
- System design principles
- Troubleshooting methodology

#### **✅ Integration Tasks (2 hours)**
1. **Task 14.1:** Build end-to-end monitoring system
```python
# Create complete_monitoring.py
from performance_monitor import ModelPerformanceMonitor
from alerting import AlertManager
import time

class MLOpsMonitoringSystem:
    def __init__(self):
        self.performance_monitor = ModelPerformanceMonitor()
        self.alert_manager = AlertManager()
        self.running = True
    
    def monitor_loop(self):
        while self.running:
            # Check performance
            latency = self.performance_monitor.check_model_latency("http://localhost:8000")
            sla_compliance = self.performance_monitor.calculate_sla_compliance()
            
            # Check for alerts
            metrics = {
                "latency_ms": latency,
                "sla_compliance": sla_compliance
            }
            
            alerts = self.alert_manager.check_thresholds(metrics)
            for alert in alerts:
                self.alert_manager.send_alert(alert)
            
            # Report status
            print(f"Monitoring - Latency: {latency:.2f}ms, SLA: {sla_compliance:.1f}%")
            
            time.sleep(60)

# Run monitoring system
monitoring = MLOpsMonitoringSystem()
monitoring.monitor_loop()
```

2. **Task 14.2:** Create deployment pipeline
```bash
# Create complete_pipeline.sh
#!/bin/bash

set -euo pipefail

VERSION="$1"
ENVIRONMENT="${2:-staging}"

echo "🚀 Complete MLOps Deployment Pipeline"
echo "Version: $VERSION"
echo "Environment: $ENVIRONMENT"

# Step 1: Build and test
echo "Step 1: Building and testing..."
docker build -t "ml-model:$VERSION" .
docker run --rm "ml-model:$VERSION" python -m pytest tests/

# Step 2: Deploy to Kubernetes
echo "Step 2: Deploying to Kubernetes..."
sed "s/{{VERSION}}/$VERSION/g" k8s-template.yaml | kubectl apply -f -

# Step 3: Wait for rollout
echo "Step 3: Waiting for rollout..."
kubectl rollout status deployment/ml-model-deployment

# Step 4: Health check
echo "Step 4: Health check..."
sleep 30
kubectl port-forward service/ml-model-service 8080:80 &
PORT_FORWARD_PID=$!

sleep 5
if curl -f http://localhost:8080/health; then
    echo "✅ Deployment successful!"
else
    echo "❌ Health check failed, rolling back..."
    kubectl rollout undo deployment/ml-model-deployment
    exit 1
fi

kill $PORT_FORWARD_PID
```

#### **🔍 Week 2 Mock Interview (1 hour)**
**Advanced Questions:**
1. "Design a system to serve 1000+ ML models with different resource requirements"
2. "How would you debug a gradual performance degradation over weeks?"
3. "Walk me through your MLOps monitoring strategy"
4. "How do you handle model deployment failures in production?"

---

## 🎯 **WEEK 3: Production Systems**

### **DAY 15: Data Structures & Algorithms for MLOps**
#### **📖 Learning Phase (2 hours)**
**Topics to Study:**
- System-relevant algorithms
- Data structures for ML systems
- Performance optimization

**Study Material:**
```python
# Read 09_algorithms_data_structures.py
# Focus on:
- Problem 1: Consistent Hashing
- Problem 2: LRU Cache
- Problem 3: Rate Limiter
```

#### **✅ Tasks to Complete (2.5 hours)**
1. **Task 15.1:** Implement LRU Cache
```python
class LRUCache:
    class Node:
        def __init__(self, key=None, value=None):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        
        # Create dummy head and tail
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node):
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """Remove an existing node"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _move_to_head(self, node):
        """Move node to head"""
        self._remove_node(node)
        self._add_node(node)
    
    def _pop_tail(self):
        """Remove last node"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key: str):
        node = self.cache.get(key)
        if node:
            self._move_to_head(node)
            return node.value
        return None
    
    def put(self, key: str, value):
        node = self.cache.get(key)
        
        if node:
            node.value = value
            self._move_to_head(node)
        else:
            new_node = self.Node(key, value)
            
            if len(self.cache) >= self.capacity:
                tail = self._pop_tail()
                del self.cache[tail.key]
            
            self.cache[key] = new_node
            self._add_node(new_node)

# Test the implementation
cache = LRUCache(3)
cache.put("model1", "weights1")
cache.put("model2", "weights2")
cache.put("model3", "weights3")
print(cache.get("model1"))  # Should return "weights1"
cache.put("model4", "weights4")  # Should evict model2
print(cache.get("model2"))  # Should return None
```

2. **Task 15.2:** Implement Rate Limiter
```python
import time
import threading

class TokenBucketRateLimiter:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def get_available_tokens(self) -> int:
        with self.lock:
            self._refill()
            return int(self.tokens)

# Test rate limiter
limiter = TokenBucketRateLimiter(capacity=10, refill_rate=2)

for i in range(15):
    success = limiter.consume(1)
    available = limiter.get_available_tokens()
    print(f"Request {i+1}: {'✅' if success else '❌'} (Available: {available})")
    time.sleep(0.1)
```

#### **🔍 Interview Prep (30 minutes)**
**Practice Questions:**
- "How would you implement caching for ML model artifacts?"
- "Design a rate limiter for ML API endpoints"

---

### **DAY 16-21: Continue with remaining days...**

[Continue this pattern for the remaining days, covering all aspects systematically]

---

## 📊 **PROGRESS TRACKING**

### **Daily Checklist Template:**
```markdown
## Day X: [Topic]

### Learning Phase ✅
- [ ] Read assigned materials
- [ ] Understand key concepts
- [ ] Take notes on important points

### Tasks Phase ✅
- [ ] Complete Task X.1
- [ ] Complete Task X.2  
- [ ] Complete Task X.3
- [ ] Document learnings

### Interview Prep ✅
- [ ] Practice questions
- [ ] Explain concepts out loud
- [ ] Note areas for improvement

### Reflection
- What did I learn today?
- What was challenging?
- What do I need to review?
```

---

## 🎯 **SUCCESS METRICS**

### **Week 1 Goals:**
- [ ] Can set up MLflow experiments
- [ ] Can containerize ML models  
- [ ] Can deploy to Kubernetes
- [ ] Can write basic automation scripts

### **Week 2 Goals:**
- [ ] Can design MLOps pipelines
- [ ] Can implement monitoring systems
- [ ] Can troubleshoot production issues
- [ ] Can explain system trade-offs

### **Week 3 Goals:**
- [ ] Can solve algorithm problems
- [ ] Can design large-scale systems
- [ ] Can handle complex scenarios
- [ ] Can communicate technical decisions

This framework ensures you learn systematically with clear daily goals and practical hands-on experience!