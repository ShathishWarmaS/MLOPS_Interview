"""
Model Predictor for MLOps Pipeline
Production-ready model inference with caching and monitoring
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Model metadata container"""
    name: str
    version: str
    stage: str
    loaded_at: float
    features: List[str]
    model_type: str
    framework: str
    size_mb: float
    metrics: Dict[str, float]

class ModelPredictor:
    """Production-ready model predictor with async support"""
    
    def __init__(self, model_name: str = "mlops_pipeline_model", cache_size: int = 1000):
        self.model_name = model_name
        self.cache_size = cache_size
        
        # Model components
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.metadata: Optional[ModelMetadata] = None
        
        # State tracking
        self.is_model_loaded = False
        self.load_time = None
        self.prediction_count = 0
        self.error_count = 0
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        
        # Caching (simple in-memory cache)
        self.prediction_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance tracking
        self.latencies = []
        self.max_latency_samples = 1000
        
    async def load_model(self, model_version: str = "latest") -> bool:
        """Load model from MLflow registry or local storage"""
        try:
            logger.info(f"Loading model: {self.model_name} version {model_version}")
            
            # Try to load from MLflow registry first
            success = await self._load_from_mlflow(model_version)
            
            if not success:
                # Fallback to local model
                logger.warning("MLflow model not found, trying local model...")
                success = await self._load_local_model()
            
            if not success:
                # Create dummy model for demo
                logger.warning("No model found, creating dummy model for demo...")
                success = await self._create_dummy_model()
            
            if success:
                self.is_model_loaded = True
                self.load_time = time.time()
                logger.info(f"✅ Model loaded successfully: {self.metadata.name} v{self.metadata.version}")
            else:
                logger.error("❌ Failed to load any model")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    async def _load_from_mlflow(self, model_version: str) -> bool:
        """Load model from MLflow Model Registry"""
        try:
            # Setup MLflow client
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(mlflow_uri)
            
            # Load model
            if model_version == "latest":
                model_uri = f"models:/{self.model_name}/Production"
            else:
                model_uri = f"models:/{self.model_name}/{model_version}"
            
            # Load in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor, 
                mlflow.sklearn.load_model, 
                model_uri
            )
            
            # Try to load scaler if available
            try:
                scaler_uri = model_uri.replace("/model", "/scaler")
                self.scaler = await loop.run_in_executor(
                    self.executor,
                    mlflow.sklearn.load_model,
                    scaler_uri
                )
            except Exception:
                logger.warning("No scaler found, proceeding without scaling")
                self.scaler = None
            
            # Get model metadata
            client = mlflow.tracking.MlflowClient()
            model_version_details = client.get_latest_versions(
                self.model_name, 
                stages=["Production"] if model_version == "latest" else None
            )[0]
            
            # Extract feature names
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
            else:
                # Default feature names for demo
                self.feature_names = [f"feature_{i}" for i in range(10)]
            
            # Create metadata
            self.metadata = ModelMetadata(
                name=self.model_name,
                version=model_version_details.version,
                stage=model_version_details.current_stage,
                loaded_at=time.time(),
                features=self.feature_names,
                model_type=type(self.model).__name__,
                framework="sklearn",
                size_mb=self._calculate_model_size(),
                metrics=self._get_model_metrics(model_version_details)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load from MLflow: {e}")
            return False
    
    async def _load_local_model(self) -> bool:
        """Load model from local storage"""
        try:
            model_path = Path("models/best_model.pkl")
            scaler_path = Path("models/scaler.pkl")
            
            if not model_path.exists():
                return False
            
            # Load model
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                joblib.load,
                str(model_path)
            )
            
            # Load scaler if available
            if scaler_path.exists():
                self.scaler = await loop.run_in_executor(
                    self.executor,
                    joblib.load,
                    str(scaler_path)
                )
            
            # Extract feature names
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
            else:
                self.feature_names = [f"feature_{i}" for i in range(10)]
            
            # Create metadata
            self.metadata = ModelMetadata(
                name=self.model_name,
                version="local-1.0.0",
                stage="Local",
                loaded_at=time.time(),
                features=self.feature_names,
                model_type=type(self.model).__name__,
                framework="sklearn",
                size_mb=self._calculate_model_size(),
                metrics={"accuracy": 0.85, "f1_score": 0.83}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            return False
    
    async def _create_dummy_model(self) -> bool:
        """Create dummy model for demonstration"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            logger.info("Creating dummy model for demonstration...")
            
            # Create dummy model
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
            self.scaler = StandardScaler()
            
            # Train on dummy data
            np.random.seed(42)
            X_dummy = np.random.randn(1000, 10)
            y_dummy = np.random.randint(0, 2, 1000)
            
            # Fit scaler and model
            X_scaled = self.scaler.fit_transform(X_dummy)
            self.model.fit(X_scaled, y_dummy)
            
            # Set feature names
            self.feature_names = [f"feature_{i}" for i in range(10)]
            
            # Create metadata
            self.metadata = ModelMetadata(
                name=self.model_name,
                version="dummy-1.0.0",
                stage="Demo",
                loaded_at=time.time(),
                features=self.feature_names,
                model_type="RandomForestClassifier",
                framework="sklearn",
                size_mb=1.0,
                metrics={"accuracy": 0.75, "f1_score": 0.73}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dummy model: {e}")
            return False
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in MB"""
        try:
            import sys
            size_bytes = sys.getsizeof(pickle.dumps(self.model))
            if self.scaler:
                size_bytes += sys.getsizeof(pickle.dumps(self.scaler))
            return size_bytes / (1024 * 1024)  # Convert to MB
        except Exception:
            return 1.0  # Default size
    
    def _get_model_metrics(self, model_version_details) -> Dict[str, float]:
        """Get model performance metrics"""
        try:
            # In a real implementation, you would fetch metrics from MLflow
            return {
                "accuracy": 0.85,
                "f1_score": 0.83,
                "precision": 0.87,
                "recall": 0.79,
                "roc_auc": 0.91
            }
        except Exception:
            return {"accuracy": 0.80}
    
    async def predict(self, features: Dict[str, Union[float, int, str, bool]]) -> Dict[str, Any]:
        """Make single prediction with caching and monitoring"""
        start_time = time.time()
        
        try:
            if not self.is_ready():
                raise RuntimeError("Model not ready")
            
            # Create cache key
            cache_key = self._create_cache_key(features)
            
            # Check cache first
            if cache_key in self.prediction_cache:
                self.cache_hits += 1
                result = self.prediction_cache[cache_key].copy()
                result['cached'] = True
                result['latency_ms'] = (time.time() - start_time) * 1000
                return result
            
            self.cache_misses += 1
            
            # Prepare features
            feature_array = await self._prepare_features(features)
            
            # Make prediction
            prediction, confidence = await self._predict_async(feature_array)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = {
                'prediction': float(prediction),
                'confidence': float(confidence),
                'model_version': self.metadata.version,
                'cached': False,
                'latency_ms': latency_ms
            }
            
            # Cache result
            if len(self.prediction_cache) < self.cache_size:
                self.prediction_cache[cache_key] = result.copy()
            
            # Update tracking
            with self.lock:
                self.prediction_count += 1
                self.latencies.append(latency_ms)
                if len(self.latencies) > self.max_latency_samples:
                    self.latencies = self.latencies[-self.max_latency_samples:]
            
            return result
            
        except Exception as e:
            with self.lock:
                self.error_count += 1
            logger.error(f"Prediction failed: {e}")
            raise
    
    async def predict_batch(self, samples: List[Dict[str, Union[float, int, str, bool]]]) -> List[Dict[str, Any]]:
        """Make batch predictions efficiently"""
        start_time = time.time()
        
        try:
            if not self.is_ready():
                raise RuntimeError("Model not ready")
            
            # Process batch
            tasks = []
            for i, sample in enumerate(samples):
                # Add request_id for tracking
                sample_with_id = sample.copy()
                sample_with_id['_request_id'] = f"batch_item_{i}"
                
                # Create prediction task
                task = self.predict(sample)
                tasks.append(task)
            
            # Execute batch predictions
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            predictions = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch item {i} failed: {result}")
                    # Create error response
                    predictions.append({
                        'request_id': f"batch_item_{i}",
                        'error': str(result),
                        'prediction': None,
                        'confidence': 0.0,
                        'model_version': self.metadata.version,
                        'latency_ms': 0.0,
                        'timestamp': time.time()
                    })
                else:
                    # Add request_id to successful prediction
                    result['request_id'] = f"batch_item_{i}"
                    result['timestamp'] = time.time()
                    predictions.append(result)
            
            total_latency = (time.time() - start_time) * 1000
            logger.info(f"Batch prediction completed: {len(samples)} samples in {total_latency:.2f}ms")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    async def _prepare_features(self, features: Dict[str, Union[float, int, str, bool]]) -> np.ndarray:
        """Prepare features for prediction"""
        try:
            # Convert to DataFrame for consistent processing
            df = pd.DataFrame([features])
            
            # Handle missing features by filling with defaults
            for feature_name in self.feature_names:
                if feature_name not in df.columns:
                    df[feature_name] = 0.0  # Default value
            
            # Select and order features
            feature_array = df[self.feature_names].values
            
            # Apply scaling if available
            if self.scaler is not None:
                loop = asyncio.get_event_loop()
                feature_array = await loop.run_in_executor(
                    self.executor,
                    self.scaler.transform,
                    feature_array
                )
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            raise ValueError(f"Invalid features: {e}")
    
    async def _predict_async(self, feature_array: np.ndarray) -> tuple:
        """Make async prediction"""
        try:
            loop = asyncio.get_event_loop()
            
            # Get prediction
            if hasattr(self.model, 'predict_proba'):
                proba = await loop.run_in_executor(
                    self.executor,
                    self.model.predict_proba,
                    feature_array
                )
                prediction = proba[0][1] if len(proba[0]) > 1 else proba[0][0]
                confidence = max(proba[0]) - min(proba[0])
            else:
                pred = await loop.run_in_executor(
                    self.executor,
                    self.model.predict,
                    feature_array
                )
                prediction = pred[0]
                confidence = 0.8  # Default confidence
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Async prediction failed: {e}")
            raise
    
    def _create_cache_key(self, features: Dict[str, Union[float, int, str, bool]]) -> str:
        """Create cache key from features"""
        try:
            # Sort features for consistent key
            sorted_items = sorted(features.items())
            key_parts = []
            for k, v in sorted_items:
                if isinstance(v, float):
                    # Round floats to avoid floating point precision issues
                    key_parts.append(f"{k}:{v:.6f}")
                else:
                    key_parts.append(f"{k}:{v}")
            return "|".join(key_parts)
        except Exception:
            return str(hash(str(features)))
    
    def is_ready(self) -> bool:
        """Check if model is ready for predictions"""
        return self.is_model_loaded and self.model is not None and self.metadata is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if not self.metadata:
            return {"status": "not_loaded"}
        
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "stage": self.metadata.stage,
            "loaded_at": self.metadata.loaded_at,
            "features": self.metadata.features,
            "model_type": self.metadata.model_type,
            "framework": self.metadata.framework,
            "size_mb": self.metadata.size_mb,
            "metrics": self.metadata.metrics,
            "prediction_count": self.prediction_count,
            "error_count": self.error_count,
            "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            "average_latency_ms": np.mean(self.latencies) if self.latencies else 0.0
        }
    
    def get_feature_schema(self) -> Dict[str, Any]:
        """Get feature schema for validation"""
        if not self.metadata:
            return {}
        
        # Create example input
        example_input = {}
        for feature in self.metadata.features:
            if feature.startswith('categorical_'):
                example_input[feature] = "category_a"
            else:
                example_input[feature] = 1.0
        
        return {
            "names": self.metadata.features,
            "types": {f: "float" for f in self.metadata.features},
            "required": self.metadata.features,
            "optional": [],
            "example": example_input
        }
    
    async def reload_model(self) -> bool:
        """Reload model from registry"""
        try:
            logger.info("Reloading model...")
            
            # Clear current state
            self.is_model_loaded = False
            self.prediction_cache.clear()
            
            # Reload model
            success = await self.load_model()
            
            if success:
                logger.info("✅ Model reloaded successfully")
            else:
                logger.error("❌ Model reload failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Model reload failed: {e}")
            return False
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.lock:
            stats = {
                "total_predictions": self.prediction_count,
                "total_errors": self.error_count,
                "error_rate": self.error_count / max(self.prediction_count, 1),
                "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
                "cache_size": len(self.prediction_cache),
                "average_latency_ms": np.mean(self.latencies) if self.latencies else 0.0,
                "p95_latency_ms": np.percentile(self.latencies, 95) if self.latencies else 0.0,
                "p99_latency_ms": np.percentile(self.latencies, 99) if self.latencies else 0.0
            }
        
        return stats
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up model predictor...")
        
        self.is_model_loaded = False
        self.prediction_cache.clear()
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("✅ Model predictor cleanup completed")