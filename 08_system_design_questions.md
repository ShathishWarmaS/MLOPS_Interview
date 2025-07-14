# MLOps System Design Interview Questions

## **Question 1: Design a Real-Time Fraud Detection System**

### **Problem Statement:**
Design a system that can detect fraudulent transactions in real-time for a payment processing company that handles 100,000 transactions per second globally.

### **Requirements:**
- **Latency:** < 50ms for fraud detection
- **Throughput:** 100K+ TPS (transactions per second)
- **Accuracy:** > 99% precision, > 95% recall
- **Availability:** 99.99% uptime
- **Global deployment:** Multi-region support
- **Compliance:** PCI DSS, GDPR compliant
- **Scalability:** Handle 10x traffic spikes during Black Friday

### **Key Components to Design:**

#### **1. Data Ingestion Layer**
```
Transaction Event → Kafka/Pulsar → Stream Processing (Flink/Spark) → Feature Engineering
```

**Discussion Points:**
- Event schema design and evolution
- Partitioning strategy for global scale
- Handling late-arriving events
- Exactly-once processing guarantees

#### **2. Feature Engineering Pipeline**
```
Real-time Features:
- Transaction velocity (last 1min, 5min, 1hr)
- Merchant risk score
- Device fingerprinting
- Location anomaly detection

Batch Features:
- User spending patterns (historical)
- Merchant category analysis
- Seasonal patterns
```

**Technical Considerations:**
- Feature store architecture (online/offline)
- Feature freshness and consistency
- Point-in-time correctness
- Feature versioning and rollback

#### **3. Model Serving Architecture**
```
Load Balancer → Model Serving Cluster → Feature Store → Response
                     ↓
              Model Registry & A/B Testing
```

**Design Decisions:**
- Model deployment strategy (blue-green, canary)
- Auto-scaling based on traffic patterns
- Model warming strategies
- Circuit breaker patterns

#### **4. Data Flow Architecture**
```
Transaction → Rules Engine → ML Model → Risk Score → Decision → Action
     ↓              ↓            ↓           ↓           ↓
Kafka Topic → Feature Store → Prediction → Database → Notification
```

### **Interview Deep Dive Questions:**

#### **Scalability:**
- "How would you handle a 10x increase in traffic?"
- "What's your sharding strategy for the feature store?"
- "How do you ensure consistent performance across regions?"

#### **Latency Optimization:**
- "How would you achieve sub-50ms latency?"
- "What caching strategies would you implement?"
- "How do you handle model inference at scale?"

#### **Model Management:**
- "How do you deploy new models without downtime?"
- "What's your A/B testing strategy for fraud models?"
- "How do you handle model drift in real-time?"

#### **Data Consistency:**
- "How do you ensure training/serving feature consistency?"
- "What's your strategy for handling late-arriving events?"
- "How do you maintain data quality at scale?"

### **Expected Architecture Diagram:**
```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Transactions  │───▶│  Kafka/Pulsar │───▶│ Stream Processor │
└─────────────────┘    └──────────────┘    └─────────────────┘
                                                     │
                       ┌─────────────────┐          ▼
                       │  Feature Store  │◀─┌─────────────────┐
                       │   (Redis/DDB)   │  │ Feature Engineer │
                       └─────────────────┘  └─────────────────┘
                               │
                               ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Model Serving │◀───│ Load Balancer │◀───│   API Gateway    │
│    Cluster      │    └──────────────┘    └─────────────────┘
└─────────────────┘
        │
        ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Decision      │───▶│   Database   │───▶│  Notifications   │
│   Engine        │    │  (Postgres)  │    │   (Kafka)       │
└─────────────────┘    └──────────────┘    └─────────────────┘
```

---

## **Question 2: Design a Recommendation System for Netflix**

### **Problem Statement:**
Design a recommendation system that serves personalized content recommendations to 200 million users worldwide with diverse viewing preferences.

### **Requirements:**
- **Scale:** 200M+ users, 100K+ titles
- **Latency:** < 100ms for recommendations
- **Personalization:** Individual user preferences
- **Freshness:** Incorporate recent viewing behavior
- **Diversity:** Avoid filter bubbles
- **Cold start:** Handle new users and content
- **A/B testing:** Support multiple recommendation algorithms

### **System Components:**

#### **1. Data Collection Pipeline**
```
User Interactions → Event Streaming → Data Lake → Feature Engineering
    ↓                    ↓               ↓            ↓
- Views               Kafka          S3/HDFS    Spark/Airflow
- Ratings             Kinesis        BigQuery   Databricks
- Searches            Pulsar         Snowflake  Apache Beam
- Duration
```

#### **2. Model Training Pipeline**
```
Batch Training:
- Collaborative Filtering (Matrix Factorization)
- Content-Based Filtering
- Deep Learning (Neural Collaborative Filtering)

Real-time Training:
- Online learning for trending content
- Bandits for exploration/exploitation
- Context-aware recommendations
```

#### **3. Serving Architecture**
```
User Request → CDN → API Gateway → Recommendation Service
                                        ↓
                               ┌─────────────────┐
                               │ Model Ensemble  │
                               │ - Collaborative │
                               │ - Content-based │
                               │ - Deep Learning │
                               │ - Trending      │
                               └─────────────────┘
                                        ↓
                               ┌─────────────────┐
                               │  Ranking &      │
                               │  Diversification│
                               └─────────────────┘
```

### **Technical Challenges & Solutions:**

#### **Cold Start Problem:**
```
New User:
- Use demographic-based recommendations
- Onboarding flow to gather initial preferences
- Popular content by region/category

New Content:
- Content-based features (genre, actors, director)
- Expert ratings and early user feedback
- Transfer learning from similar content
```

#### **Scalability Solutions:**
```
Data Partitioning:
- User-based sharding for personalization
- Content-based partitioning for metadata
- Geographic distribution for latency

Caching Strategy:
- Pre-computed recommendations (daily batch)
- Real-time adjustments (recent interactions)
- CDN for static content metadata
```

### **Interview Questions:**

#### **Algorithm Design:**
- "How would you handle the exploration vs exploitation trade-off?"
- "What's your approach to recommendation diversity?"
- "How do you incorporate temporal dynamics in recommendations?"

#### **System Architecture:**
- "How would you ensure sub-100ms latency globally?"
- "What's your strategy for handling recommendation model updates?"
- "How do you A/B test different recommendation algorithms?"

#### **Data Engineering:**
- "How do you handle the massive scale of user interaction data?"
- "What's your approach to feature engineering for recommendations?"
- "How do you ensure data quality and consistency?"

---

## **Question 3: Design a Computer Vision Pipeline for Autonomous Vehicles**

### **Problem Statement:**
Design an end-to-end computer vision system for autonomous vehicles that can process camera feeds in real-time for object detection, lane detection, and decision making.

### **Requirements:**
- **Real-time processing:** < 30ms inference latency
- **High accuracy:** 99.99% object detection accuracy
- **Edge deployment:** Run on vehicle hardware
- **Safety critical:** Fail-safe mechanisms
- **Continuous learning:** Improve from fleet data
- **Multi-modal:** Camera, LiDAR, Radar fusion

### **System Architecture:**

#### **1. Edge Processing Pipeline**
```
Camera Feeds → Image Preprocessing → Object Detection → Tracking → Decision
    (8x)            ↓                      ↓             ↓         ↓
                GPU Optimization     TensorRT/ONNX    Kalman     Control
                                                     Filter     System
```

#### **2. Data Collection & Training Pipeline**
```
Fleet Vehicles → Edge Storage → Cloud Upload → Data Lake → Annotation
      ↓              ↓              ↓            ↓            ↓
   Local Cache → Compression → Bandwidth Mgmt → S3/GCS → Labeling Platform
      ↓                                                       ↓
  Critical Events ────────────────────────────────────→ Priority Queue
```

#### **3. Model Development Lifecycle**
```
Data Annotation → Model Training → Validation → Edge Optimization → OTA Deployment
      ↓               ↓              ↓              ↓                  ↓
   Label Studio   PyTorch/TF    Simulation    TensorRT/OpenVINO   Fleet Update
                                Testing       Model Pruning       Version Control
```

### **Technical Considerations:**

#### **Edge Optimization:**
```python
# Model optimization pipeline
def optimize_for_edge(model):
    # Quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # Pruning
    pruned_model = prune_model(quantized_model, sparsity=0.3)
    
    # Compilation for target hardware
    tensorrt_model = torch2trt(pruned_model)
    
    return tensorrt_model
```

#### **Safety & Reliability:**
```
Safety Mechanisms:
- Redundant sensor fusion
- Conservative fallback decisions
- Graceful degradation
- Hardware watchdog timers
- Model uncertainty estimation

Validation Framework:
- Simulation testing (CARLA, AirSim)
- Closed-course testing
- Shadow mode deployment
- Gradual rollout strategy
```

### **Interview Deep Dive:**

#### **Performance Optimization:**
- "How would you achieve sub-30ms inference on edge hardware?"
- "What's your approach to model quantization and pruning?"
- "How do you handle variable lighting and weather conditions?"

#### **Safety & Reliability:**
- "How do you ensure the system fails safely?"
- "What's your validation strategy for safety-critical AI?"
- "How do you handle edge cases and unknown scenarios?"

#### **Continuous Learning:**
- "How do you collect and utilize fleet data for model improvement?"
- "What's your strategy for handling data privacy and security?"
- "How do you deploy model updates to a fleet of vehicles?"

---

## **Question 4: Design a Large Language Model Training Infrastructure**

### **Problem Statement:**
Design the infrastructure to train and serve large language models (100B+ parameters) for a conversational AI service serving millions of users.

### **Requirements:**
- **Model size:** 100B+ parameters
- **Training data:** 1TB+ of text data
- **Distributed training:** Multi-node, multi-GPU
- **Serving latency:** < 200ms for text generation
- **Cost optimization:** Efficient resource utilization
- **Experiment management:** Multiple model variants
- **Deployment:** Blue-green deployment for model updates

### **Training Infrastructure:**

#### **1. Distributed Training Architecture**
```
Data Preprocessing → Distributed Storage → Training Cluster → Model Checkpoints
        ↓                    ↓                   ↓                ↓
   Tokenization         HDFS/S3           Multi-node GPU    Versioned Storage
   Data Cleaning        Sharding          (1000+ GPUs)      Model Registry
   Filtering            Replication       PyTorch DDP       Artifact Store
```

#### **2. Training Pipeline**
```python
# Distributed training configuration
training_config = {
    "model": {
        "type": "transformer",
        "layers": 96,
        "hidden_size": 12288,
        "attention_heads": 96,
        "parameters": "175B"
    },
    "distributed": {
        "strategy": "data_parallel",
        "gradient_checkpointing": True,
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 8
    },
    "optimization": {
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "lr_scheduler": "cosine",
        "warmup_steps": 2000
    }
}
```

#### **3. Serving Infrastructure**
```
User Request → Load Balancer → Model Serving → Response Generation
                    ↓              ↓                  ↓
              Geographic       GPU Clusters      Streaming Response
              Routing          Model Sharding    Response Caching
              Rate Limiting    KV Cache          Quality Filtering
```

### **Technical Challenges:**

#### **Memory Management:**
```
Model Parallelism Strategies:
- Pipeline parallelism (layers across devices)
- Tensor parallelism (weights across devices)
- Expert parallelism (mixture of experts)
- ZeRO optimizer state partitioning

Memory Optimization:
- Gradient checkpointing
- Activation recomputation
- Offloading to CPU/NVMe
- Dynamic loss scaling
```

#### **Serving Optimization:**
```python
# Model serving optimizations
class OptimizedLLMServing:
    def __init__(self):
        self.model_shards = self.load_sharded_model()
        self.kv_cache = self.initialize_kv_cache()
        self.tokenizer = self.load_tokenizer()
    
    def generate(self, prompt, max_length=100):
        # Batched inference
        # KV cache optimization
        # Speculative decoding
        # Parallel sampling
        pass
```

### **Interview Questions:**

#### **Training Efficiency:**
- "How would you optimize training throughput for 100B parameter models?"
- "What's your strategy for handling training failures and restarts?"
- "How do you balance model quality vs training cost?"

#### **Serving Performance:**
- "How would you achieve sub-200ms latency for text generation?"
- "What's your approach to model sharding and load balancing?"
- "How do you handle variable-length generation requests?"

#### **Infrastructure Management:**
- "How do you manage the cost of training and serving large models?"
- "What's your monitoring strategy for distributed training?"
- "How do you handle hardware failures during long training runs?"

---

## **Question 5: Design a Multi-Modal AI Platform**

### **Problem Statement:**
Design a platform that can process and understand multiple data modalities (text, images, audio, video) for content moderation across social media platforms.

### **Requirements:**
- **Modalities:** Text, Images, Audio, Video
- **Scale:** 1B+ posts per day
- **Latency:** < 500ms for content classification
- **Accuracy:** 99%+ for harmful content detection
- **Languages:** Support 50+ languages
- **Real-time:** Live stream processing capability
- **Compliance:** Regional content policies

### **Architecture Overview:**

#### **1. Multi-Modal Ingestion Pipeline**
```
Content Upload → Format Detection → Modal Extraction → Processing Queue
      ↓              ↓                    ↓                ↓
   API Gateway   MIME Analysis      Text/Image/Audio    Kafka/RabbitMQ
   File Upload   Metadata Extract   Video Segments      Priority Queue
   Streaming     Quality Check      Thumbnail Gen       Load Balancing
```

#### **2. Modal-Specific Processing**
```
Text Processing:
- Language Detection → Translation → Sentiment Analysis → Hate Speech Detection

Image Processing:
- Object Detection → Scene Analysis → NSFW Detection → Face Recognition

Audio Processing:
- Speech-to-Text → Language ID → Audio Classification → Music Detection

Video Processing:
- Frame Extraction → Audio Separation → Temporal Analysis → Action Recognition
```

#### **3. Fusion and Decision Engine**
```
Modal Results → Feature Fusion → Ensemble Model → Final Decision → Action
      ↓              ↓              ↓               ↓           ↓
   Confidence    Late/Early      Weighted Vote   Policy Rule  Block/Allow
   Scores        Fusion          Attention       Business     Flag/Review
   Uncertainty   Multi-head      Meta-learning   Logic        Escalate
```

### **Technical Implementation:**

#### **Model Architecture:**
```python
class MultiModalClassifier:
    def __init__(self):
        self.text_encoder = self.load_text_model()      # BERT/RoBERTa
        self.image_encoder = self.load_vision_model()   # ResNet/ViT
        self.audio_encoder = self.load_audio_model()    # Wav2Vec2
        self.video_encoder = self.load_video_model()    # 3D CNN
        self.fusion_layer = self.load_fusion_model()    # Transformer
    
    def classify(self, content):
        text_features = self.text_encoder(content.text)
        image_features = self.image_encoder(content.images)
        audio_features = self.audio_encoder(content.audio)
        video_features = self.video_encoder(content.video)
        
        fused_features = self.fusion_layer([
            text_features, image_features, 
            audio_features, video_features
        ])
        
        return self.classifier_head(fused_features)
```

#### **Scalability Strategy:**
```
Horizontal Scaling:
- Modal-specific processing clusters
- GPU pools for different model types
- Auto-scaling based on queue depth
- Geographic distribution

Optimization Techniques:
- Model quantization and pruning
- Batch processing for efficiency
- Caching for repeated content
- Progressive processing (fast → detailed)
```

### **Interview Deep Dive:**

#### **Multi-Modal Fusion:**
- "How would you handle missing modalities in content?"
- "What's your strategy for temporal alignment in video processing?"
- "How do you balance different modal contributions to final decisions?"

#### **Scalability & Performance:**
- "How would you optimize processing for billion-scale content?"
- "What's your approach to handling live streaming content?"
- "How do you manage GPU resources across different model types?"

#### **Accuracy & Bias:**
- "How do you ensure consistent performance across different cultures?"
- "What's your strategy for handling edge cases and adversarial content?"
- "How do you continuously improve model performance with feedback?"

---

## **General System Design Interview Tips**

### **Structure Your Approach:**
1. **Clarify Requirements** (5 minutes)
2. **High-Level Architecture** (10 minutes)
3. **Detailed Design** (20 minutes)
4. **Scale & Optimize** (10 minutes)
5. **Handle Edge Cases** (10 minutes)

### **Key Areas to Cover:**
- **Data Flow:** How data moves through the system
- **Storage:** Database choices and data modeling
- **Compute:** Processing and serving infrastructure
- **Networking:** Load balancing and geographic distribution
- **Monitoring:** Observability and alerting
- **Security:** Authentication, authorization, data protection

### **ML-Specific Considerations:**
- **Model Lifecycle:** Training, validation, deployment
- **Feature Engineering:** Real-time vs batch processing
- **Model Serving:** Latency, throughput, auto-scaling
- **Experimentation:** A/B testing and gradual rollouts
- **Monitoring:** Model drift, performance degradation
- **Data Quality:** Validation, lineage, freshness

### **Common Pitfalls to Avoid:**
- Diving into implementation details too early
- Ignoring non-functional requirements (scale, latency)
- Not considering failure scenarios
- Overlooking data consistency and quality
- Forgetting about monitoring and observability
- Not discussing trade-offs and alternatives