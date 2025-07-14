# 🔄 Project 2: Complete MLOps Pipeline with CI/CD

## **Enterprise-Grade MLOps Automation from Scratch**

Build a complete MLOps pipeline that automatically trains, validates, deploys, and monitors ML models with **zero manual intervention** and **enterprise-grade reliability**.

---

## 🎯 **Project Overview**

### **What You'll Build**
A fully automated ML pipeline that:
- ✅ **Automatically triggers** on data/code changes
- ✅ **Trains multiple models** and selects the best
- ✅ **Validates model quality** with comprehensive tests
- ✅ **Deploys to multiple environments** (staging → production)
- ✅ **Monitors performance** and triggers retraining
- ✅ **Handles rollbacks** automatically on failures
- ✅ **Manages infrastructure** as code
- ✅ **Provides full observability** and audit trails

### **Business Value**
- Reduce model deployment time from weeks to minutes
- Ensure consistent model quality across environments
- Enable rapid experimentation and iteration
- Provide complete audit trail for compliance
- Scale ML operations across multiple teams
- Minimize manual errors and operational overhead

---

## 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLOps Pipeline Architecture                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Data      │───▶│  Training   │───▶│  Model      │───▶│  Serving    │
│  Sources    │    │  Pipeline   │    │ Validation  │    │  Pipeline   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Data       │    │  Experiment │    │  Model      │    │ Performance │
│ Validation  │    │  Tracking   │    │  Registry   │    │ Monitoring  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

                    ┌─────────────────────────────────┐
                    │         CI/CD Pipeline          │
                    │                                 │
                    │ GitHub Actions / GitLab CI     │
                    │ ├─ Code Quality Checks          │
                    │ ├─ Automated Testing            │
                    │ ├─ Model Training               │
                    │ ├─ Model Validation             │
                    │ ├─ Deployment Automation        │
                    │ └─ Infrastructure Updates       │
                    └─────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │     Infrastructure as Code      │
                    │                                 │
                    │ Terraform / Kubernetes         │
                    │ ├─ Cloud Resources              │
                    │ ├─ Monitoring Stack             │
                    │ ├─ Security Policies            │
                    │ └─ Network Configuration        │
                    └─────────────────────────────────┘
```

---

## 🚀 **Quick Start (10 minutes)**

### **1. Setup Repository**
```bash
cd mlops_interview_prep/production_projects/project2_mlops_pipeline

# Initialize with automation
./tools/setup-pipeline.sh
```

### **2. Configure CI/CD**
```bash
# Setup GitHub Actions (or GitLab CI)
cp .github/workflows/mlops-pipeline.yml.template .github/workflows/mlops-pipeline.yml

# Configure secrets
gh secret set DOCKER_REGISTRY_URL
gh secret set KUBECONFIG_DATA
gh secret set MLFLOW_TRACKING_URI
```

### **3. Trigger First Pipeline**
```bash
# Make a change to trigger pipeline
echo "# Pipeline test" >> training/models/README.md
git add . && git commit -m "feat: trigger initial pipeline"
git push
```

### **4. Monitor Pipeline**
```bash
# Watch GitHub Actions
open https://github.com/your-repo/actions

# Check pipeline logs
gh workflow list
gh run watch
```

---

## 📂 **Project Structure**

```
project2_mlops_pipeline/
├── README.md                           # This file
├── .github/workflows/                  # CI/CD pipelines
│   ├── mlops-pipeline.yml             # Main MLOps pipeline
│   ├── data-pipeline.yml              # Data processing pipeline
│   ├── model-deployment.yml           # Model deployment pipeline
│   ├── infrastructure.yml             # Infrastructure updates
│   └── security-scan.yml              # Security scanning
│
├── training/                          # Training pipeline
│   ├── pipelines/                     # Training pipeline definitions
│   │   ├── data_pipeline.py           # Data processing pipeline
│   │   ├── training_pipeline.py       # Model training pipeline
│   │   └── evaluation_pipeline.py     # Model evaluation pipeline
│   ├── models/                        # Model definitions
│   │   ├── baseline_model.py          # Baseline model
│   │   ├── advanced_model.py          # Advanced model
│   │   └── ensemble_model.py          # Ensemble model
│   ├── experiments/                   # Experiment configurations
│   │   ├── config/                    # Training configurations
│   │   └── hyperparams/               # Hyperparameter definitions
│   └── utils/                         # Training utilities
│
├── serving/                           # Model serving
│   ├── api/                          # Serving API
│   │   ├── main.py                   # FastAPI application
│   │   ├── models.py                 # Request/response models
│   │   └── routes/                   # API routes
│   ├── batch/                        # Batch inference
│   │   ├── batch_predictor.py        # Batch prediction service
│   │   └── schedulers/               # Batch job schedulers
│   └── streaming/                    # Real-time inference
│       ├── stream_processor.py       # Stream processing
│       └── kafka_consumer.py         # Kafka message consumer
│
├── monitoring/                       # Monitoring & observability
│   ├── model_monitor.py              # Model performance monitoring
│   ├── data_monitor.py               # Data quality monitoring
│   ├── drift_detector.py             # Concept/data drift detection
│   ├── alerting/                     # Alert management
│   │   ├── alert_manager.py          # Alert orchestration
│   │   └── notifications/            # Notification channels
│   └── dashboards/                   # Monitoring dashboards
│       ├── grafana/                  # Grafana dashboards
│       └── custom/                   # Custom dashboards
│
├── infrastructure/                   # Infrastructure as Code
│   ├── terraform/                    # Terraform configurations
│   │   ├── environments/             # Environment-specific configs
│   │   ├── modules/                  # Reusable modules
│   │   └── providers/                # Cloud provider configs
│   ├── kubernetes/                   # Kubernetes manifests
│   │   ├── base/                     # Base configurations
│   │   ├── overlays/                 # Environment overlays
│   │   └── operators/                # Custom operators
│   ├── helm/                         # Helm charts
│   │   ├── mlops-platform/           # Main platform chart
│   │   └── monitoring/               # Monitoring stack chart
│   └── scripts/                      # Infrastructure scripts
│
├── tools/                            # Development & deployment tools
│   ├── setup-pipeline.sh             # Pipeline setup automation
│   ├── deploy.sh                     # Deployment automation
│   ├── validate-model.py             # Model validation tools
│   ├── load-test.py                  # Load testing tools
│   └── backup-restore.sh             # Backup/restore tools
│
├── tests/                            # Comprehensive test suite
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   ├── e2e/                          # End-to-end tests
│   ├── performance/                  # Performance tests
│   └── security/                     # Security tests
│
├── docs/                             # Documentation
│   ├── architecture.md               # System architecture
│   ├── deployment.md                 # Deployment guide
│   ├── monitoring.md                 # Monitoring guide
│   ├── troubleshooting.md            # Troubleshooting guide
│   └── api/                          # API documentation
│
├── config/                           # Configuration management
│   ├── environments/                 # Environment configs
│   ├── models/                       # Model configurations
│   └── pipelines/                    # Pipeline configurations
│
└── data/                             # Data management
    ├── schemas/                      # Data schemas
    ├── samples/                      # Sample datasets
    └── validation/                   # Data validation rules
```

---

## 🔄 **Pipeline Components**

### **1. Data Pipeline**
```python
# training/pipelines/data_pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import (
    extract_raw_data,
    validate_data_quality,
    clean_and_transform,
    feature_engineering,
    split_data
)

def create_data_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=extract_raw_data,
            inputs=["params:data_sources", "params:date_range"],
            outputs="raw_data",
            name="extract_raw_data_node"
        ),
        node(
            func=validate_data_quality,
            inputs=["raw_data", "params:quality_rules"],
            outputs="validated_data",
            name="validate_data_quality_node"
        ),
        node(
            func=clean_and_transform,
            inputs=["validated_data", "params:transform_config"],
            outputs="clean_data",
            name="clean_and_transform_node"
        ),
        node(
            func=feature_engineering,
            inputs=["clean_data", "params:feature_config"],
            outputs="features",
            name="feature_engineering_node"
        ),
        node(
            func=split_data,
            inputs=["features", "params:split_config"],
            outputs=["train_data", "val_data", "test_data"],
            name="split_data_node"
        )
    ])
```

### **2. Training Pipeline**
```python
# training/pipelines/training_pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import (
    train_baseline_model,
    train_advanced_models,
    evaluate_models,
    select_best_model,
    register_model
)

def create_training_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=train_baseline_model,
            inputs=["train_data", "params:baseline_config"],
            outputs="baseline_model",
            name="train_baseline_model_node"
        ),
        node(
            func=train_advanced_models,
            inputs=["train_data", "params:advanced_configs"],
            outputs="advanced_models",
            name="train_advanced_models_node"
        ),
        node(
            func=evaluate_models,
            inputs=["baseline_model", "advanced_models", "val_data"],
            outputs="model_evaluations",
            name="evaluate_models_node"
        ),
        node(
            func=select_best_model,
            inputs=["model_evaluations", "params:selection_criteria"],
            outputs="best_model",
            name="select_best_model_node"
        ),
        node(
            func=register_model,
            inputs=["best_model", "test_data", "params:registry_config"],
            outputs="registered_model",
            name="register_model_node"
        )
    ])
```

### **3. CI/CD Pipeline**
```yaml
# .github/workflows/mlops-pipeline.yml
name: MLOps Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

env:
  PYTHON_VERSION: '3.9'
  DOCKER_REGISTRY: ghcr.io
  KUBECONFIG_PATH: ~/.kube/config

jobs:
  # Code Quality & Security
  quality-checks:
    name: Code Quality & Security
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
      
      - name: Lint code
        run: |
          flake8 training/ serving/ monitoring/
          black --check training/ serving/ monitoring/
          isort --check-only training/ serving/ monitoring/
      
      - name: Type checking
        run: mypy training/ serving/ monitoring/
      
      - name: Security scan
        run: |
          bandit -r training/ serving/ monitoring/
          safety check

  # Data Validation
  data-validation:
    name: Data Quality Validation
    runs-on: ubuntu-latest
    needs: quality-checks
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Validate data schemas
        run: |
          python tools/validate-data-schemas.py
      
      - name: Check data quality
        run: |
          python training/pipelines/data_validation.py
      
      - name: Generate data report
        run: |
          python tools/generate-data-report.py
        
      - name: Upload data report
        uses: actions/upload-artifact@v3
        with:
          name: data-quality-report
          path: reports/data-quality.html

  # Model Training
  model-training:
    name: Model Training & Validation
    runs-on: ubuntu-latest
    needs: [quality-checks, data-validation]
    strategy:
      matrix:
        model: [baseline, advanced, ensemble]
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Configure MLflow
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          export MLFLOW_TRACKING_USERNAME=${{ secrets.MLFLOW_USERNAME }}
          export MLFLOW_TRACKING_PASSWORD=${{ secrets.MLFLOW_PASSWORD }}
      
      - name: Train model
        run: |
          python training/train_model.py \
            --model-type ${{ matrix.model }} \
            --config config/training/${{ matrix.model }}.yaml
      
      - name: Validate model
        run: |
          python tools/validate-model.py \
            --model-type ${{ matrix.model }} \
            --validation-config config/validation/model-validation.yaml
      
      - name: Upload model artifacts
        uses: actions/upload-artifact@v3
        with:
          name: model-${{ matrix.model }}
          path: models/${{ matrix.model }}/

  # Model Selection
  model-selection:
    name: Model Selection & Registration
    runs-on: ubuntu-latest
    needs: model-training
    steps:
      - uses: actions/checkout@v3
      - name: Download all model artifacts
        uses: actions/download-artifact@v3
      
      - name: Compare models
        run: |
          python tools/compare-models.py \
            --models baseline,advanced,ensemble \
            --metrics accuracy,precision,recall,f1,auc
      
      - name: Select best model
        id: model-selection
        run: |
          BEST_MODEL=$(python tools/select-best-model.py)
          echo "best_model=$BEST_MODEL" >> $GITHUB_OUTPUT
      
      - name: Register model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          python tools/register-model.py \
            --model-name ${{ steps.model-selection.outputs.best_model }} \
            --stage Production \
            --version ${{ github.sha }}

  # Container Build
  build-containers:
    name: Build & Push Containers
    runs-on: ubuntu-latest
    needs: model-selection
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.DOCKER_REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ env.DOCKER_REGISTRY }}/${{ github.repository }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix={{branch}}-
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./serving/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # Staging Deployment
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build-containers
    environment: staging
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Kubernetes
        run: |
          echo "${{ secrets.KUBECONFIG_DATA }}" | base64 -d > ${{ env.KUBECONFIG_PATH }}
      
      - name: Deploy to staging
        run: |
          helm upgrade --install mlops-staging infrastructure/helm/mlops-platform \
            --namespace mlops-staging \
            --create-namespace \
            --set image.tag=${{ needs.build-containers.outputs.image-tag }} \
            --set environment=staging \
            --wait --timeout=600s
      
      - name: Run smoke tests
        run: |
          python tests/e2e/smoke_tests.py \
            --environment staging \
            --endpoint https://staging.mlops.company.com

  # Integration Tests
  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: deploy-staging
    steps:
      - uses: actions/checkout@v3
      
      - name: Run integration tests
        run: |
          python -m pytest tests/integration/ \
            --environment staging \
            --junit-xml=integration-test-results.xml
      
      - name: Run performance tests
        run: |
          python tests/performance/load_test.py \
            --environment staging \
            --duration 300 \
            --users 100
      
      - name: Publish test results
        uses: dorny/test-reporter@v1
        if: always()
        with:
          name: Integration Test Results
          path: integration-test-results.xml
          reporter: java-junit

  # Production Deployment
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [deploy-staging, integration-tests]
    environment: production
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Kubernetes
        run: |
          echo "${{ secrets.KUBECONFIG_DATA }}" | base64 -d > ${{ env.KUBECONFIG_PATH }}
      
      - name: Deploy to production
        run: |
          helm upgrade --install mlops-production infrastructure/helm/mlops-platform \
            --namespace mlops-production \
            --create-namespace \
            --set image.tag=${{ needs.build-containers.outputs.image-tag }} \
            --set environment=production \
            --set replicaCount=5 \
            --wait --timeout=600s
      
      - name: Verify deployment
        run: |
          kubectl rollout status deployment/mlops-production -n mlops-production
          python tests/e2e/production_verification.py

  # Post-Deployment Monitoring
  post-deployment:
    name: Post-Deployment Setup
    runs-on: ubuntu-latest
    needs: deploy-production
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup monitoring
        run: |
          python monitoring/setup_monitoring.py \
            --environment production \
            --model-version ${{ github.sha }}
      
      - name: Configure alerts
        run: |
          python monitoring/configure_alerts.py \
            --environment production
      
      - name: Send deployment notification
        run: |
          python tools/send-notification.py \
            --type deployment-success \
            --environment production \
            --version ${{ github.sha }}
```

---

## 🛠️ **Implementation Details**

### **Training Pipeline Components**

#### **1. Data Processing Pipeline**
```python
# training/pipelines/data_pipeline.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import great_expectations as ge
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Production data processing pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_quality_suite = self._load_quality_suite()
        
    def extract_raw_data(self, sources: Dict, date_range: Tuple[str, str]) -> pd.DataFrame:
        """Extract data from multiple sources"""
        logger.info(f"Extracting data for range: {date_range}")
        
        dfs = []
        for source_name, source_config in sources.items():
            logger.info(f"Extracting from source: {source_name}")
            
            if source_config['type'] == 'database':
                df = self._extract_from_database(source_config, date_range)
            elif source_config['type'] == 's3':
                df = self._extract_from_s3(source_config, date_range)
            elif source_config['type'] == 'api':
                df = self._extract_from_api(source_config, date_range)
            else:
                raise ValueError(f"Unsupported source type: {source_config['type']}")
            
            df['source'] = source_name
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Extracted {len(combined_df)} records")
        
        return combined_df
    
    def validate_data_quality(self, data: pd.DataFrame, quality_rules: Dict) -> pd.DataFrame:
        """Validate data quality using Great Expectations"""
        logger.info("Running data quality validation")
        
        # Convert to Great Expectations dataset
        ge_df = ge.from_pandas(data)
        
        # Run expectations
        validation_results = []
        
        for rule_name, rule_config in quality_rules.items():
            logger.debug(f"Running quality rule: {rule_name}")
            
            if rule_config['type'] == 'completeness':
                result = ge_df.expect_column_values_to_not_be_null(
                    column=rule_config['column']
                )
            elif rule_config['type'] == 'range':
                result = ge_df.expect_column_values_to_be_between(
                    column=rule_config['column'],
                    min_value=rule_config['min'],
                    max_value=rule_config['max']
                )
            elif rule_config['type'] == 'uniqueness':
                result = ge_df.expect_column_values_to_be_unique(
                    column=rule_config['column']
                )
            elif rule_config['type'] == 'custom':
                # Custom validation logic
                result = self._run_custom_validation(ge_df, rule_config)
            
            validation_results.append({
                'rule': rule_name,
                'success': result.success,
                'details': result.result
            })
        
        # Check if all validations passed
        failed_validations = [r for r in validation_results if not r['success']]
        if failed_validations:
            logger.error(f"Data quality validation failed: {failed_validations}")
            
            # Decide whether to fail or continue with warnings
            if self.config.get('strict_validation', True):
                raise ValueError("Data quality validation failed")
            else:
                logger.warning("Continuing with data quality warnings")
        
        logger.info("Data quality validation completed successfully")
        return data
    
    def clean_and_transform(self, data: pd.DataFrame, transform_config: Dict) -> pd.DataFrame:
        """Clean and transform data"""
        logger.info("Cleaning and transforming data")
        
        df = data.copy()
        
        # Handle missing values
        missing_config = transform_config.get('missing_values', {})
        for column, strategy in missing_config.items():
            if strategy == 'drop':
                df = df.dropna(subset=[column])
            elif strategy == 'fill_mean':
                df[column] = df[column].fillna(df[column].mean())
            elif strategy == 'fill_median':
                df[column] = df[column].fillna(df[column].median())
            elif strategy == 'fill_mode':
                df[column] = df[column].fillna(df[column].mode()[0])
            elif isinstance(strategy, (int, float, str)):
                df[column] = df[column].fillna(strategy)
        
        # Remove outliers
        outlier_config = transform_config.get('outliers', {})
        for column, method in outlier_config.items():
            if method == 'iqr':
                df = self._remove_outliers_iqr(df, column)
            elif method == 'zscore':
                df = self._remove_outliers_zscore(df, column)
        
        # Data type conversions
        dtype_config = transform_config.get('dtypes', {})
        for column, dtype in dtype_config.items():
            df[column] = df[column].astype(dtype)
        
        # Custom transformations
        custom_transforms = transform_config.get('custom_transforms', [])
        for transform in custom_transforms:
            df = self._apply_custom_transform(df, transform)
        
        logger.info(f"Data cleaning completed. Shape: {df.shape}")
        return df
    
    def feature_engineering(self, data: pd.DataFrame, feature_config: Dict) -> pd.DataFrame:
        """Create engineered features"""
        logger.info("Creating engineered features")
        
        df = data.copy()
        
        # Time-based features
        time_features = feature_config.get('time_features', {})
        if 'timestamp_column' in time_features:
            ts_col = time_features['timestamp_column']
            df[ts_col] = pd.to_datetime(df[ts_col])
            
            if time_features.get('extract_hour', False):
                df['hour'] = df[ts_col].dt.hour
            if time_features.get('extract_day_of_week', False):
                df['day_of_week'] = df[ts_col].dt.dayofweek
            if time_features.get('extract_month', False):
                df['month'] = df[ts_col].dt.month
            if time_features.get('extract_quarter', False):
                df['quarter'] = df[ts_col].dt.quarter
        
        # Categorical encoding
        categorical_features = feature_config.get('categorical_encoding', {})
        for column, encoding_type in categorical_features.items():
            if encoding_type == 'onehot':
                df = pd.get_dummies(df, columns=[column], prefix=column)
            elif encoding_type == 'label':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[f'{column}_encoded'] = le.fit_transform(df[column])
            elif encoding_type == 'target':
                # Target encoding implementation
                df = self._apply_target_encoding(df, column, feature_config.get('target_column'))
        
        # Numerical features
        numerical_features = feature_config.get('numerical_features', {})
        for feature_name, feature_def in numerical_features.items():
            if feature_def['type'] == 'ratio':
                df[feature_name] = df[feature_def['numerator']] / df[feature_def['denominator']]
            elif feature_def['type'] == 'binning':
                df[feature_name] = pd.cut(df[feature_def['column']], bins=feature_def['bins'])
            elif feature_def['type'] == 'polynomial':
                df[feature_name] = df[feature_def['column']] ** feature_def['degree']
            elif feature_def['type'] == 'interaction':
                df[feature_name] = df[feature_def['column1']] * df[feature_def['column2']]
        
        # Aggregation features
        aggregation_features = feature_config.get('aggregation_features', {})
        for feature_name, agg_def in aggregation_features.items():
            groupby_cols = agg_def['groupby']
            agg_col = agg_def['column']
            agg_func = agg_def['function']
            
            agg_df = df.groupby(groupby_cols)[agg_col].agg(agg_func).reset_index()
            agg_df.columns = groupby_cols + [feature_name]
            df = df.merge(agg_df, on=groupby_cols, how='left')
        
        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        return df
    
    def split_data(self, data: pd.DataFrame, split_config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets"""
        logger.info("Splitting data into train/validation/test sets")
        
        # Get target column
        target_col = split_config['target_column']
        
        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # First split: train+val vs test
        test_size = split_config.get('test_size', 0.2)
        stratify = y if split_config.get('stratify', False) else None
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=split_config.get('random_state', 42),
            stratify=stratify
        )
        
        # Second split: train vs val
        val_size = split_config.get('val_size', 0.2)
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        
        stratify_temp = y_temp if split_config.get('stratify', False) else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=split_config.get('random_state', 42),
            stratify=stratify_temp
        )
        
        # Combine features and targets
        train_data = pd.concat([X_train, y_train], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {len(train_data)} samples")
        logger.info(f"  Validation: {len(val_data)} samples")
        logger.info(f"  Test: {len(test_data)} samples")
        
        return train_data, val_data, test_data
    
    # Helper methods
    def _extract_from_database(self, config: Dict, date_range: Tuple[str, str]) -> pd.DataFrame:
        """Extract data from database"""
        # Implementation for database extraction
        pass
    
    def _extract_from_s3(self, config: Dict, date_range: Tuple[str, str]) -> pd.DataFrame:
        """Extract data from S3"""
        # Implementation for S3 extraction
        pass
    
    def _extract_from_api(self, config: Dict, date_range: Tuple[str, str]) -> pd.DataFrame:
        """Extract data from API"""
        # Implementation for API extraction
        pass
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    def _remove_outliers_zscore(self, df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using Z-score method"""
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return df[z_scores <= threshold]
```

This is just the beginning of the comprehensive MLOps pipeline. Would you like me to continue with the model training components, CI/CD implementation, and infrastructure setup?