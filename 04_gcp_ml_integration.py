"""
Google Cloud Platform ML Services Integration Exercises
Master GCP ML services for MLOps engineering interviews
"""

from google.cloud import aiplatform
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import functions_v1
import json
import os
from typing import Dict, List, Any

# ==============================================================================
# EXERCISE 1: Vertex AI Custom Training Job
# ==============================================================================

def exercise_1_vertex_ai_training():
    """
    Scenario: Deploy custom training job on Vertex AI
    
    Requirements:
    1. Create custom training container
    2. Configure training job with hyperparameters
    3. Use managed datasets
    4. Implement distributed training
    5. Save model to Vertex AI Model Registry
    """
    
    # Initialize Vertex AI
    aiplatform.init(
        project="your-project-id",
        location="us-central1",
        staging_bucket="gs://your-staging-bucket"
    )
    
    # TODO: Implement custom training job
    def create_custom_training_job():
        """
        Create and submit custom training job
        """
        
        # Define training job configuration
        job_config = {
            "display_name": "fraud-detection-training",
            "python_package_gcs_uri": "gs://your-bucket/trainer.tar.gz",
            "python_module": "trainer.task",
            "container_uri": "gcr.io/cloud-aiplatform/training/pytorch-gpu.1-9:latest",
            "requirements": ["scikit-learn==1.0.2", "pandas==1.3.0"],
            "args": [
                "--epochs=100",
                "--batch-size=32",
                "--learning-rate=0.001"
            ]
        }
        
        # TODO: Configure machine types, accelerators
        # TODO: Set up distributed training
        # TODO: Configure output directories
        
        pass
    
    # Interview Questions:
    # 1. How would you optimize training costs?
    # 2. What's your strategy for handling training failures?
    # 3. How do you manage different Python dependencies?

# ==============================================================================
# EXERCISE 2: Vertex AI Model Deployment and Serving
# ==============================================================================

def exercise_2_model_deployment():
    """
    Scenario: Deploy trained model for online and batch prediction
    
    Requirements:
    1. Create Vertex AI Model from artifacts
    2. Deploy to managed endpoint
    3. Configure auto-scaling
    4. Implement A/B testing
    5. Set up batch prediction jobs
    """
    
    def deploy_model_to_endpoint():
        """
        Deploy model to Vertex AI managed endpoint
        """
        
        # TODO: Create model from saved artifacts
        model = aiplatform.Model.upload(
            display_name="fraud-detection-model",
            artifact_uri="gs://your-bucket/model/",
            serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/pytorch-gpu.1-9:latest"
        )
        
        # TODO: Create endpoint and deploy model
        # TODO: Configure traffic splitting for A/B testing
        # TODO: Set up auto-scaling policies
        
        pass
    
    def create_batch_prediction_job():
        """
        Create batch prediction job for large datasets
        """
        
        # TODO: Configure batch prediction
        # TODO: Handle input from BigQuery
        # TODO: Write results back to BigQuery
        
        pass
    
    # Interview Questions:
    # 1. How do you handle cold starts in serving?
    # 2. What's your approach to model versioning in production?
    # 3. How do you monitor model drift?

# ==============================================================================
# EXERCISE 3: Vertex AI Pipelines (Kubeflow)
# ==============================================================================

def exercise_3_vertex_pipelines():
    """
    Scenario: Create end-to-end ML pipeline using Vertex AI Pipelines
    
    Requirements:
    1. Define pipeline components
    2. Implement data validation
    3. Create training and evaluation steps
    4. Implement model deployment
    5. Set up pipeline scheduling
    """
    
    from kfp.v2 import dsl
    from kfp.v2.dsl import component, pipeline, Artifact, Model, Dataset, Input, Output
    
    @component(
        packages_to_install=["pandas", "scikit-learn"],
        base_image="python:3.9"
    )
    def data_preprocessing(
        input_data: Input[Dataset],
        processed_data: Output[Dataset]
    ):
        """Data preprocessing component"""
        
        # TODO: Implement data preprocessing logic
        # TODO: Data validation and quality checks
        # TODO: Feature engineering
        
        pass
    
    @component(
        packages_to_install=["scikit-learn", "mlflow"],
        base_image="python:3.9"
    )
    def model_training(
        training_data: Input[Dataset],
        model: Output[Model],
        metrics: Output[Artifact]
    ):
        """Model training component"""
        
        # TODO: Implement training logic
        # TODO: Hyperparameter tuning
        # TODO: Model validation
        # TODO: Save metrics and model artifacts
        
        pass
    
    @component
    def model_evaluation(
        model: Input[Model],
        test_data: Input[Dataset],
        evaluation_metrics: Output[Artifact]
    ):
        """Model evaluation component"""
        
        # TODO: Implement evaluation logic
        # TODO: Calculate performance metrics
        # TODO: Generate evaluation report
        
        pass
    
    @pipeline(
        name="ml-training-pipeline",
        description="End-to-end ML training pipeline"
    )
    def ml_pipeline(
        input_data_uri: str,
        model_display_name: str = "fraud-detection-model"
    ):
        """Define the complete ML pipeline"""
        
        # TODO: Connect all components
        # TODO: Add conditional logic for model deployment
        # TODO: Implement pipeline parameters
        
        pass
    
    # Interview Questions:
    # 1. How do you handle pipeline failures and retries?
    # 2. What's your strategy for pipeline testing?
    # 3. How do you manage pipeline versions?

# ==============================================================================
# EXERCISE 4: Feature Store Integration
# ==============================================================================

def exercise_4_feature_store():
    """
    Scenario: Implement feature store using Vertex AI Feature Store
    
    Requirements:
    1. Create feature store and entity types
    2. Ingest features from streaming and batch sources
    3. Serve features for online prediction
    4. Implement feature monitoring
    5. Handle feature versioning
    """
    
    def create_feature_store():
        """Create and configure feature store"""
        
        # TODO: Create feature store
        # TODO: Define entity types (user, transaction, etc.)
        # TODO: Create feature groups
        
        pass
    
    def ingest_features():
        """Ingest features from various sources"""
        
        # TODO: Batch ingestion from BigQuery
        # TODO: Streaming ingestion from Pub/Sub
        # TODO: Handle feature transformations
        
        pass
    
    def serve_features():
        """Serve features for online prediction"""
        
        # TODO: Implement online serving
        # TODO: Handle feature freshness
        # TODO: Implement caching strategy
        
        pass
    
    # Interview Questions:
    # 1. How do you ensure feature consistency between training and serving?
    # 2. What's your approach to feature lifecycle management?
    # 3. How do you handle feature store scaling?

# ==============================================================================
# EXERCISE 5: AutoML Integration
# ==============================================================================

def exercise_5_automl_integration():
    """
    Scenario: Use Vertex AI AutoML for rapid prototyping
    
    Requirements:
    1. Create AutoML training job
    2. Compare with custom models
    3. Export AutoML models
    4. Integrate with existing pipeline
    """
    
    def create_automl_job():
        """Create AutoML training job"""
        
        # TODO: Configure AutoML dataset
        # TODO: Set training objectives
        # TODO: Configure model constraints
        
        pass
    
    def export_automl_model():
        """Export AutoML model for custom deployment"""
        
        # TODO: Export model artifacts
        # TODO: Convert to custom serving format
        # TODO: Deploy to custom infrastructure
        
        pass
    
    # Interview Questions:
    # 1. When would you choose AutoML over custom models?
    # 2. How do you handle AutoML model explainability?
    # 3. What are the limitations of AutoML?

# ==============================================================================
# EXERCISE 6: BigQuery ML Integration
# ==============================================================================

def exercise_6_bigquery_ml():
    """
    Scenario: Leverage BigQuery ML for large-scale analytics
    
    Requirements:
    1. Create ML models in BigQuery
    2. Integrate with Vertex AI
    3. Export models for serving
    4. Implement feature engineering in SQL
    """
    
    def create_bqml_model():
        """Create machine learning model in BigQuery"""
        
        sql_query = """
        CREATE OR REPLACE MODEL `project.dataset.fraud_model`
        OPTIONS(
            model_type='LOGISTIC_REG',
            auto_class_weights=true,
            input_label_cols=['is_fraud']
        ) AS
        SELECT
            amount,
            merchant_category,
            time_of_day,
            user_age,
            is_fraud
        FROM `project.dataset.transactions`
        WHERE date >= '2023-01-01'
        """
        
        # TODO: Execute BigQuery ML training
        # TODO: Evaluate model performance
        # TODO: Export for serving
        
        pass
    
    def integrate_with_vertex():
        """Integrate BigQuery ML with Vertex AI"""
        
        # TODO: Export BQML model to Vertex AI
        # TODO: Create prediction endpoint
        # TODO: Set up batch scoring
        
        pass

# ==============================================================================
# EXERCISE 7: Cloud Functions for Model Serving
# ==============================================================================

def exercise_7_cloud_functions():
    """
    Scenario: Deploy lightweight models using Cloud Functions
    
    Requirements:
    1. Create serverless prediction function
    2. Handle cold starts
    3. Implement error handling
    4. Set up monitoring and logging
    """
    
    def create_prediction_function():
        """Create Cloud Function for model serving"""
        
        function_code = """
        import functions_framework
        import pickle
        import pandas as pd
        from google.cloud import storage
        
        # Global variable to cache model
        model = None
        
        def load_model():
            global model
            if model is None:
                # TODO: Load model from Cloud Storage
                pass
            return model
        
        @functions_framework.http
        def predict(request):
            try:
                # TODO: Parse request data
                # TODO: Load model if not cached
                # TODO: Make prediction
                # TODO: Return response
                pass
            except Exception as e:
                # TODO: Error handling and logging
                pass
        """
        
        # TODO: Deploy function with proper configuration
        # TODO: Set up IAM permissions
        # TODO: Configure scaling and timeouts
        
        pass

# ==============================================================================
# EXERCISE 8: Monitoring and Observability
# ==============================================================================

def exercise_8_monitoring():
    """
    Scenario: Implement comprehensive monitoring for ML workloads
    
    Requirements:
    1. Set up model performance monitoring
    2. Implement drift detection
    3. Create custom metrics and alerts
    4. Set up logging and tracing
    """
    
    def setup_model_monitoring():
        """Configure Vertex AI Model Monitoring"""
        
        # TODO: Configure monitoring job
        # TODO: Set up drift detection thresholds
        # TODO: Create alerting policies
        
        pass
    
    def create_custom_metrics():
        """Create custom metrics for business KPIs"""
        
        # TODO: Define custom metrics
        # TODO: Set up Cloud Monitoring dashboards
        # TODO: Create alert policies
        
        pass

# ==============================================================================
# COMPREHENSIVE GCP INTERVIEW SCENARIOS
# ==============================================================================

"""
SCENARIO 1: COST OPTIMIZATION
Question: How would you optimize ML workload costs on GCP?
Expected Discussion:
- Preemptible instances for training
- Spot instances for batch workloads
- Auto-scaling configurations
- Resource scheduling
- Reserved capacity planning

SCENARIO 2: MULTI-REGION DEPLOYMENT
Question: Design a multi-region ML serving architecture
Expected Discussion:
- Regional model deployments
- Data residency requirements
- Latency optimization
- Disaster recovery
- Global load balancing

SCENARIO 3: SECURITY AND COMPLIANCE
Question: How do you secure ML workloads on GCP?
Expected Discussion:
- IAM and service accounts
- VPC and network security
- Data encryption
- Audit logging
- Compliance frameworks

SCENARIO 4: DATA PIPELINE INTEGRATION
Question: Integrate ML with existing data infrastructure
Expected Discussion:
- Dataflow for stream processing
- Cloud Composer for orchestration
- Pub/Sub for event streaming
- Data catalog and lineage
- Schema evolution

SCENARIO 5: HYBRID AND EDGE DEPLOYMENT
Question: Deploy models to edge and on-premises
Expected Discussion:
- Anthos for hybrid deployments
- Edge TPU optimization
- Model optimization techniques
- Connectivity and sync strategies
- Local inference capabilities
"""

if __name__ == "__main__":
    print("GCP ML Services Integration Exercises")
    print("Complete these exercises to master GCP ML services")
    print("Practice explaining architectural decisions and trade-offs")