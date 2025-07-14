# 🚀 Quick Start Guide - Fraud Detection System

## **Get Up and Running in 5 Minutes**

This production-grade MLOps project demonstrates real-time fraud detection with complete automation, monitoring, and deployment.

---

## 📋 **Prerequisites**

```bash
# Required
- Python 3.8+ 
- Git

# Optional (for full features)  
- Docker & Docker Compose
- Kubernetes (minikube/kind)
```

---

## ⚡ **Super Quick Start (2 minutes)**

### **Option 1: Automated Setup**
```bash
# Clone and setup everything automatically
cd mlops_interview_prep/production_projects/project1_fraud_detection

# Run automated setup
./scripts/setup.sh --full

# That's it! 🎉
```

### **Option 2: Manual Steps**
```bash
# 1. Setup Python environment
make env-create
source fraud_detection_env/bin/activate

# 2. Install dependencies  
make install

# 3. Train model (quick version)
make train-quick

# 4. Start API server
make serve
```

### **🧪 Test It Works**
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test fraud prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "amount": 150.0,
       "merchant": "Amazon", 
       "merchant_category": "online",
       "hour": 14,
       "day_of_week": 2,
       "user_age": 35,
       "account_age_days": 456
     }'
```

**Expected Response:**
```json
{
  "transaction_id": "txn_...",
  "fraud_probability": 0.15,
  "risk_level": "low",
  "confidence": 0.87,
  "latency_ms": 45.2,
  "model_version": "1.0.0"
}
```

---

## 🐳 **Docker Setup (3 minutes)**

### **Start Full Stack**
```bash
# Build and start all services
make docker-build
make docker-up

# Services will be available at:
# 🌐 API: http://localhost:8000
# 🔬 MLflow: http://localhost:5000  
# 📊 Grafana: http://localhost:3000
# 📈 Prometheus: http://localhost:9090
```

### **Test Docker Setup**
```bash
# Wait for services to start (30 seconds)
sleep 30

# Test the API
curl http://localhost:8000/health

# View logs
make docker-logs
```

---

## 📊 **Key Endpoints**

| Endpoint | Purpose | Example |
|----------|---------|---------|
| `GET /health` | Health check | `curl localhost:8000/health` |
| `GET /ready` | Readiness probe | `curl localhost:8000/ready` |
| `POST /predict` | Fraud prediction | See example above |
| `POST /predict/batch` | Batch predictions | Multiple transactions |
| `GET /metrics` | System metrics | Performance data |
| `GET /docs` | API documentation | Swagger UI |

---

## 🧠 **Training Your Own Model**

### **Quick Training (Demo)**
```bash
make train-quick    # 5,000 samples - 1 minute
```

### **Production Training**
```bash  
make train          # 50,000 samples - 5 minutes
make train-large    # 100,000 samples - 10 minutes
```

### **Monitor Training**
```bash
# View MLflow experiments
open http://localhost:5000

# Check logs
tail -f logs/fraud_detection.log
```

---

## 🧪 **Testing**

### **Run All Tests**
```bash
make test           # Unit + integration tests
```

### **Specific Tests**
```bash
make test-unit      # Unit tests only
make test-api       # API endpoint tests  
make test-load      # Load testing (requires locust)
```

### **Manual API Testing**
```bash
# Start server
make serve &

# Test endpoints
make test-api

# Stop server
pkill -f "python src/api/main.py"
```

---

## 📈 **Monitoring & Observability**

### **View Dashboards**
```bash
make start-monitoring

# Access dashboards:
# 📊 Grafana: http://localhost:3000 (admin/admin)
# 📈 Prometheus: http://localhost:9090
# 🔬 MLflow: http://localhost:5000
```

### **Check System Status**
```bash
make status         # Overall system status
make logs          # Application logs
curl localhost:8000/metrics  # API metrics
```

---

## 🚀 **Production Deployment**

### **Kubernetes Deployment**
```bash
# Deploy to Kubernetes
make k8s-deploy

# Check status
make k8s-status

# View services
kubectl get all -n fraud-detection
```

### **Staging Deployment**
```bash
make deploy-staging
```

### **Production Deployment**
```bash
make deploy-production  # Interactive confirmation required
```

---

## 🛠️ **Development Workflow**

### **Daily Development**
```bash
# Activate environment
source fraud_detection_env/bin/activate

# Make changes...

# Format code
make format

# Run tests
make test

# Start development server
make serve
```

### **Code Quality**
```bash
make lint           # Code linting
make format         # Auto-formatting
make security-scan  # Security check
```

---

## 📋 **Project Structure Overview**

```
fraud_detection/
├── src/                    # Source code
│   ├── api/               # REST API (FastAPI)
│   ├── training/          # Model training
│   ├── inference/         # Model serving
│   ├── monitoring/        # Observability
│   └── shared/           # Utilities
├── tests/                 # Test suites
├── docker/               # Docker configs
├── k8s/                  # Kubernetes manifests
├── scripts/              # Automation scripts
└── data/                 # Data and models
```

---

## 🔧 **Common Issues & Solutions**

### **API Won't Start**
```bash
# Check if model exists
ls data/models/

# Train model if missing
make train-quick

# Check logs
tail logs/fraud_detection.log
```

### **Docker Issues**
```bash
# Clean up Docker
make docker-down
docker system prune -f

# Rebuild
make docker-build
make docker-up
```

### **Port Conflicts**
```bash
# Check what's using port 8000
lsof -i :8000

# Use different port
export API_PORT=8001
make serve
```

### **Permission Issues**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Fix Docker permissions (Linux)
sudo chown -R $USER:$USER .
```

---

## 📚 **What You'll Learn**

### **MLOps Skills**
- ✅ End-to-end ML pipeline development
- ✅ Production model serving with FastAPI
- ✅ Docker containerization & orchestration
- ✅ Kubernetes deployment & scaling
- ✅ MLflow experiment tracking
- ✅ Monitoring & observability
- ✅ CI/CD for ML systems

### **Production Engineering**
- ✅ API design & development
- ✅ Database integration
- ✅ Caching strategies
- ✅ Load balancing
- ✅ Health checks & probes
- ✅ Error handling & logging
- ✅ Security best practices

### **Interview Readiness**
- ✅ Working production system to demo
- ✅ Complete MLOps pipeline understanding
- ✅ Hands-on cloud-native experience
- ✅ Performance optimization knowledge
- ✅ Troubleshooting & debugging skills

---

## 🎯 **Next Steps**

### **Extend the Project**
1. **Add Features:**
   - Real-time data streaming
   - Advanced model explainability
   - A/B testing framework
   - Multi-model serving

2. **Scale It:**
   - Deploy to cloud (AWS/GCP/Azure)
   - Add load balancing
   - Implement auto-scaling
   - Set up monitoring alerts

3. **Production Hardening:**
   - Add authentication
   - Implement rate limiting
   - Set up backup strategies
   - Create disaster recovery plan

### **Use for Interviews**
1. **Demo the System:** Show working fraud detection API
2. **Explain Architecture:** Walk through end-to-end pipeline
3. **Discuss Decisions:** Explain technology choices & trade-offs
4. **Show Monitoring:** Demonstrate observability setup
5. **Handle Questions:** Be ready to extend or modify live

---

## ❓ **Need Help?**

### **Documentation**
- `make help` - See all available commands
- `open http://localhost:8000/docs` - API documentation
- Check `docs/` folder for detailed guides

### **Troubleshooting**
- Check logs: `make logs`
- View status: `make status`
- Clean and restart: `make clean && make setup`

### **Resources**
- MLflow UI: http://localhost:5000
- API Docs: http://localhost:8000/docs
- Grafana: http://localhost:3000

---

## 🎉 **You're Ready!**

You now have a **production-grade MLOps system** that you can:
- ✅ **Demo in interviews**
- ✅ **Extend for portfolios** 
- ✅ **Learn from for career growth**
- ✅ **Use as reference architecture**

**Happy coding and good luck with your interviews! 🚀**