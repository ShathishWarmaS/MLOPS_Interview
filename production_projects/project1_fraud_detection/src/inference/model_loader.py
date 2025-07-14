#!/usr/bin/env python3
"""
Model Loader - Production Model Loading and Management
Handles loading trained models from various sources (local, MLflow, cloud storage)
"""

import os
import logging
import joblib
import mlflow
import mlflow.sklearn
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

from shared.config import config
from training.data_processor import FraudDataProcessor

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Production model loader with multiple source support
    
    Features:
    - Local filesystem loading
    - MLflow model registry integration
    - Cloud storage support (S3, GCS, Azure)
    - Model validation and testing
    - Automatic fallback mechanisms
    - Model metadata management
    """
    
    def __init__(self):
        self.supported_sources = ['local', 'mlflow', 's3', 'gcs', 'azure']
        self.model_cache = {}
        self.model_metadata_cache = {}
        
    async def load_model(
        self,
        source: str = 'local',
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        version: Optional[str] = None
    ) -> Tuple[BaseEstimator, FraudDataProcessor, Dict[str, Any]]:
        """
        Load model from specified source
        
        Args:
            source: Source type ('local', 'mlflow', 's3', 'gcs', 'azure')
            model_path: Path to model (for local/cloud sources)
            model_name: Model name (for MLflow)
            version: Model version (for MLflow)
            
        Returns:
            Tuple of (model, data_processor, metadata)
        """
        logger.info(f"Loading model from {source}")
        
        # Check cache first
        cache_key = f"{source}_{model_path}_{model_name}_{version}"
        if cache_key in self.model_cache:
            logger.info("Model found in cache")
            return self.model_cache[cache_key]
        
        # Load based on source
        if source == 'local':
            result = await self._load_from_local(model_path)
        elif source == 'mlflow':
            result = await self._load_from_mlflow(model_name, version)
        elif source == 's3':
            result = await self._load_from_s3(model_path)
        elif source == 'gcs':
            result = await self._load_from_gcs(model_path)
        elif source == 'azure':
            result = await self._load_from_azure(model_path)
        else:
            raise ValueError(f"Unsupported source: {source}")
        
        # Cache the result
        self.model_cache[cache_key] = result
        
        # Validate model
        await self._validate_model(result[0], result[1])
        
        logger.info(f"Model loaded successfully from {source}")
        return result
    
    async def _load_from_local(self, model_path: Optional[str] = None) -> Tuple[BaseEstimator, FraudDataProcessor, Dict[str, Any]]:
        """Load model from local filesystem"""
        try:
            # Determine model path
            if model_path is None:
                model_path = "data/models"
            
            model_dir = Path(model_path)
            
            # Check if directory exists
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
            # Load model
            model_file = model_dir / "best_fraud_model.pkl"
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            model = joblib.load(model_file)
            logger.info(f"Model loaded from {model_file}")
            
            # Load data processor
            processor_file = model_dir / "data_processor.pkl"
            if processor_file.exists():
                data_processor = joblib.load(processor_file)
                logger.info(f"Data processor loaded from {processor_file}")
            else:
                logger.warning("Data processor not found, creating new one")
                data_processor = FraudDataProcessor()
            
            # Load metadata
            metadata = {
                'source': 'local',
                'model_path': str(model_file),
                'load_time': time.time(),
                'model_type': type(model).__name__
            }
            
            # Try to load feature columns
            feature_file = model_dir / "feature_columns.txt"
            if feature_file.exists():
                with open(feature_file, 'r') as f:
                    feature_columns = [line.strip() for line in f.readlines()]
                metadata['feature_columns'] = feature_columns
            
            return model, data_processor, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model from local: {e}")
            raise
    
    async def _load_from_mlflow(self, model_name: Optional[str] = None, version: Optional[str] = None) -> Tuple[BaseEstimator, FraudDataProcessor, Dict[str, Any]]:
        """Load model from MLflow registry"""
        try:
            # Set MLflow tracking URI
            if hasattr(config, 'MLFLOW_TRACKING_URI'):
                mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            
            # Default model name
            if model_name is None:
                model_name = "fraud_detection_production"
            
            # Build model URI
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            # Load model
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded from MLflow: {model_uri}")
            
            # Get model metadata
            client = mlflow.tracking.MlflowClient()
            if version:
                model_version = client.get_model_version(model_name, version)
            else:
                model_versions = client.get_latest_versions(model_name, stages=["Production"])
                if not model_versions:
                    model_versions = client.get_latest_versions(model_name)
                model_version = model_versions[0]
            
            # Download artifacts
            run_id = model_version.run_id
            
            # Try to load data processor
            try:
                artifact_path = client.download_artifacts(run_id, "data_processor.pkl")
                data_processor = joblib.load(artifact_path)
                logger.info("Data processor loaded from MLflow")
            except Exception as e:
                logger.warning(f"Failed to load data processor from MLflow: {e}")
                data_processor = FraudDataProcessor()
            
            # Build metadata
            metadata = {
                'source': 'mlflow',
                'model_name': model_name,
                'model_version': model_version.version,
                'run_id': run_id,
                'model_uri': model_uri,
                'load_time': time.time(),
                'model_type': type(model).__name__,
                'creation_timestamp': model_version.creation_timestamp,
                'last_updated_timestamp': model_version.last_updated_timestamp
            }
            
            # Add run metadata
            try:
                run = client.get_run(run_id)
                metadata['run_metrics'] = run.data.metrics
                metadata['run_params'] = run.data.params
            except Exception as e:
                logger.warning(f"Failed to get run metadata: {e}")
            
            return model, data_processor, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            raise
    
    async def _load_from_s3(self, model_path: str) -> Tuple[BaseEstimator, FraudDataProcessor, Dict[str, Any]]:
        """Load model from AWS S3"""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # Parse S3 path
            if not model_path.startswith('s3://'):
                raise ValueError("S3 path must start with s3://")
            
            path_parts = model_path[5:].split('/', 1)
            bucket_name = path_parts[0]
            key_prefix = path_parts[1] if len(path_parts) > 1 else ''
            
            # Initialize S3 client
            s3_client = boto3.client('s3')
            
            # Create temporary directory
            import tempfile
            temp_dir = Path(tempfile.mkdtemp())
            
            # Download model artifacts
            model_key = f"{key_prefix}/model.pkl"
            processor_key = f"{key_prefix}/data_processor.pkl"
            metadata_key = f"{key_prefix}/metadata.json"
            
            # Download model
            model_file = temp_dir / "model.pkl"
            s3_client.download_file(bucket_name, model_key, str(model_file))
            model = joblib.load(model_file)
            
            # Download data processor
            processor_file = temp_dir / "data_processor.pkl"
            try:
                s3_client.download_file(bucket_name, processor_key, str(processor_file))
                data_processor = joblib.load(processor_file)
            except ClientError:
                logger.warning("Data processor not found in S3, creating new one")
                data_processor = FraudDataProcessor()
            
            # Download metadata
            metadata_file = temp_dir / "metadata.json"
            try:
                s3_client.download_file(bucket_name, metadata_key, str(metadata_file))
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except ClientError:
                metadata = {}
            
            # Add loading metadata
            metadata.update({
                'source': 's3',
                'model_path': model_path,
                'load_time': time.time(),
                'model_type': type(model).__name__
            })
            
            # Cleanup temporary files
            import shutil
            shutil.rmtree(temp_dir)
            
            return model, data_processor, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model from S3: {e}")
            raise
    
    async def _load_from_gcs(self, model_path: str) -> Tuple[BaseEstimator, FraudDataProcessor, Dict[str, Any]]:
        """Load model from Google Cloud Storage"""
        try:
            from google.cloud import storage
            
            # Parse GCS path
            if not model_path.startswith('gs://'):
                raise ValueError("GCS path must start with gs://")
            
            path_parts = model_path[5:].split('/', 1)
            bucket_name = path_parts[0]
            key_prefix = path_parts[1] if len(path_parts) > 1 else ''
            
            # Initialize GCS client
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            # Create temporary directory
            import tempfile
            temp_dir = Path(tempfile.mkdtemp())
            
            # Download model artifacts
            model_blob = bucket.blob(f"{key_prefix}/model.pkl")
            processor_blob = bucket.blob(f"{key_prefix}/data_processor.pkl")
            metadata_blob = bucket.blob(f"{key_prefix}/metadata.json")
            
            # Download model
            model_file = temp_dir / "model.pkl"
            model_blob.download_to_filename(str(model_file))
            model = joblib.load(model_file)
            
            # Download data processor
            processor_file = temp_dir / "data_processor.pkl"
            try:
                processor_blob.download_to_filename(str(processor_file))
                data_processor = joblib.load(processor_file)
            except Exception:
                logger.warning("Data processor not found in GCS, creating new one")
                data_processor = FraudDataProcessor()
            
            # Download metadata
            metadata_file = temp_dir / "metadata.json"
            try:
                metadata_blob.download_to_filename(str(metadata_file))
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}
            
            # Add loading metadata
            metadata.update({
                'source': 'gcs',
                'model_path': model_path,
                'load_time': time.time(),
                'model_type': type(model).__name__
            })
            
            # Cleanup temporary files
            import shutil
            shutil.rmtree(temp_dir)
            
            return model, data_processor, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model from GCS: {e}")
            raise
    
    async def _load_from_azure(self, model_path: str) -> Tuple[BaseEstimator, FraudDataProcessor, Dict[str, Any]]:
        """Load model from Azure Blob Storage"""
        try:
            from azure.storage.blob import BlobServiceClient
            
            # Parse Azure path
            if not model_path.startswith('azure://'):
                raise ValueError("Azure path must start with azure://")
            
            # Implementation would be similar to S3/GCS
            # This is a placeholder for Azure Blob Storage integration
            raise NotImplementedError("Azure Blob Storage support not implemented yet")
            
        except Exception as e:
            logger.error(f"Failed to load model from Azure: {e}")
            raise
    
    async def _validate_model(self, model: BaseEstimator, data_processor: FraudDataProcessor):
        """Validate loaded model"""
        try:
            logger.info("Validating loaded model...")
            
            # Check if model has required methods
            if not hasattr(model, 'predict'):
                raise ValueError("Model must have predict method")
            
            if not hasattr(model, 'predict_proba'):
                raise ValueError("Model must have predict_proba method")
            
            # Test with dummy data
            dummy_data = pd.DataFrame({
                'amount': [100.0],
                'hour': [14],
                'day_of_week': [2],
                'user_age': [35],
                'account_age_days': [456],
                'merchant_category': ['online']
            })
            
            # Test data processing
            if data_processor.is_fitted:
                processed_data = data_processor.preprocess_features(dummy_data, is_training=False)
                feature_columns = data_processor.get_feature_columns()
                
                # Extract features that exist in processed data
                available_features = [col for col in feature_columns if col in processed_data.columns]
                X_test = processed_data[available_features]
                
                # Test prediction
                prediction = model.predict(X_test)
                probabilities = model.predict_proba(X_test)
                
                # Validate prediction format
                assert len(prediction) == 1, "Prediction should return one result"
                assert probabilities.shape == (1, 2), "Probabilities should be (1, 2) shape"
                assert 0 <= probabilities[0, 1] <= 1, "Probability should be between 0 and 1"
                
                logger.info("✅ Model validation passed")
            else:
                logger.warning("⚠️ Data processor not fitted, skipping full validation")
                
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            raise ValueError(f"Model validation failed: {e}")
    
    def clear_cache(self):
        """Clear model cache"""
        self.model_cache.clear()
        self.model_metadata_cache.clear()
        logger.info("Model cache cleared")
    
    def get_cached_models(self) -> Dict[str, Any]:
        """Get information about cached models"""
        return {
            'cached_models': list(self.model_cache.keys()),
            'cache_size': len(self.model_cache),
            'cache_metadata': {
                key: {
                    'model_type': type(value[0]).__name__,
                    'load_time': value[2].get('load_time', 'unknown')
                }
                for key, value in self.model_cache.items()
            }
        }
    
    async def list_available_models(self, source: str = 'mlflow') -> List[Dict[str, Any]]:
        """List available models from a source"""
        if source == 'mlflow':
            return await self._list_mlflow_models()
        elif source == 'local':
            return await self._list_local_models()
        else:
            raise ValueError(f"Listing not supported for source: {source}")
    
    async def _list_mlflow_models(self) -> List[Dict[str, Any]]:
        """List models from MLflow registry"""
        try:
            if hasattr(config, 'MLFLOW_TRACKING_URI'):
                mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            
            client = mlflow.tracking.MlflowClient()
            models = []
            
            # Get all registered models
            for rm in client.search_registered_models():
                model_info = {
                    'name': rm.name,
                    'description': rm.description,
                    'latest_versions': []
                }
                
                # Get latest versions
                latest_versions = client.get_latest_versions(rm.name)
                for version in latest_versions:
                    model_info['latest_versions'].append({
                        'version': version.version,
                        'stage': version.current_stage,
                        'creation_time': version.creation_timestamp,
                        'last_updated': version.last_updated_timestamp
                    })
                
                models.append(model_info)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list MLflow models: {e}")
            return []
    
    async def _list_local_models(self) -> List[Dict[str, Any]]:
        """List models from local filesystem"""
        try:
            models = []
            model_dir = Path("data/models")
            
            if model_dir.exists():
                for model_file in model_dir.glob("*.pkl"):
                    if model_file.name.endswith('_model.pkl'):
                        models.append({
                            'name': model_file.stem,
                            'path': str(model_file),
                            'size': model_file.stat().st_size,
                            'modified': model_file.stat().st_mtime
                        })
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list local models: {e}")
            return []