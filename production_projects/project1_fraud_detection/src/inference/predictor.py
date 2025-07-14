#!/usr/bin/env python3
"""
Fraud Predictor - Production Model Inference
Real-time fraud detection with MLOps best practices
"""

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import logging
import asyncio
import time
import os
from typing import Dict, Any, Optional, List
import json
from pathlib import Path
import uuid
import warnings
warnings.filterwarnings('ignore')

from api.models import TransactionRequest, RiskLevel
from shared.config import config
from inference.model_loader import ModelLoader
from training.data_processor import FraudDataProcessor

logger = logging.getLogger(__name__)

class FraudPredictor:
    """
    Production-grade fraud detection predictor
    
    Features:
    - Model loading and caching
    - Feature engineering
    - Real-time prediction
    - Performance monitoring
    - Graceful error handling
    """
    
    def __init__(self):
        self.model = None
        self.data_processor = None
        self.model_loader = ModelLoader()
        self.model_version = "1.0.0"
        self.model_metadata = {}
        self.feature_names = []
        self.is_model_loaded = False
        self.prediction_count = 0
        self.load_time = None
        self.model_source = None
        self.model_path = None
    
    async def load_model(
        self, 
        source: str = 'local',
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        version: Optional[str] = None
    ):
        """
        Load ML model from various sources
        
        Args:
            source: Source type ('local', 'mlflow', 's3', 'gcs')
            model_path: Optional path to model (for local/cloud sources)
            model_name: Model name (for MLflow)
            version: Model version (for MLflow)
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Loading fraud detection model from {source}...")
            
            # Use ModelLoader to load model
            self.model, self.data_processor, self.model_metadata = await self.model_loader.load_model(
                source=source,
                model_path=model_path,
                model_name=model_name,
                version=version
            )
            
            # Extract metadata
            self.model_version = self.model_metadata.get('model_version', '1.0.0')
            self.model_source = source
            self.model_path = model_path
            
            # Get feature names from data processor
            if self.data_processor and self.data_processor.is_fitted:
                self.feature_names = self.data_processor.get_feature_columns()
            else:
                # Fallback feature names
                self.feature_names = [
                    'amount', 'hour', 'day_of_week', 'user_age', 'account_age_days', 'merchant_category'
                ]
            
            self.is_model_loaded = True
            self.load_time = time.time()
            
            load_duration = time.time() - start_time
            logger.info(f"✅ Model loaded successfully in {load_duration:.2f}s")
            logger.info(f"Model version: {self.model_version}")
            logger.info(f"Model source: {self.model_source}")
            logger.info(f"Features: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            # Fallback: train a new model
            logger.info("🔄 Falling back to training new model...")
            await self._train_and_save_model()
    
    
    async def _train_and_save_model(self):
        """Train a new model and save it"""
        logger.info("🎯 Training new fraud detection model...")
        
        try:
            # Create and train data processor
            self.data_processor = FraudDataProcessor()
            
            # Generate synthetic training data
            df = self.data_processor.generate_synthetic_data(n_samples=10000)
            
            # Preprocess data
            df_processed = self.data_processor.preprocess_features(df, is_training=True)
            
            # Get feature columns
            feature_columns = self.data_processor.get_feature_columns()
            
            # Prepare features and target
            X = df_processed[feature_columns]
            y = df_processed['is_fraud']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Set feature names
            self.feature_names = feature_columns
            
            # Save model metadata
            self.model_metadata = {
                'model_type': 'RandomForestClassifier',
                'feature_names': self.feature_names,
                'auc_score': auc_score,
                'training_timestamp': time.time(),
                'training_samples': len(X_train)
            }
            
            # Save model artifacts
            await self._save_model_artifacts()
            
            self.is_model_loaded = True
            self.load_time = time.time()
            
            logger.info(f"✅ Model trained successfully - AUC: {auc_score:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to train model: {e}")
            raise
    
    
    async def _save_model_artifacts(self):
        """Save model artifacts locally"""
        try:
            # Create model directory
            model_dir = Path("data/models")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            joblib.dump(self.model, model_dir / "best_fraud_model.pkl")
            
            # Save data processor
            if self.data_processor:
                joblib.dump(self.data_processor, model_dir / "data_processor.pkl")
            
            # Save metadata
            with open(model_dir / "metadata.json", 'w') as f:
                json.dump(self.model_metadata, f, indent=2)
            
            # Save feature columns
            with open(model_dir / "feature_columns.txt", 'w') as f:
                f.write('\n'.join(self.feature_names))
            
            logger.info(f"Model artifacts saved to {model_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save model artifacts: {e}")
    
    async def predict(self, request: TransactionRequest) -> Dict[str, Any]:
        """
        Make fraud prediction for a transaction
        
        Args:
            request: Transaction request data
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.is_model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Convert request to DataFrame
            transaction_data = {
                'amount': request.amount,
                'hour': request.hour,
                'day_of_week': request.day_of_week,
                'user_age': request.user_age,
                'account_age_days': request.account_age_days,
                'merchant_category': request.merchant_category
            }
            
            df = pd.DataFrame([transaction_data])
            
            # Process features using data processor
            if self.data_processor and self.data_processor.is_fitted:
                processed_df = self.data_processor.preprocess_features(df, is_training=False)
                feature_columns = self.data_processor.get_feature_columns()
                
                # Extract features that exist in processed data
                available_features = [col for col in feature_columns if col in processed_df.columns]
                X = processed_df[available_features]
            else:
                # Fallback: use raw features (not recommended for production)
                logger.warning("Data processor not fitted, using raw features")
                feature_columns = ['amount', 'hour', 'day_of_week', 'user_age', 'account_age_days']
                X = df[feature_columns]
            
            # Make prediction
            fraud_probability = self.model.predict_proba(X)[0][1]
            
            # Determine risk level
            risk_level = self._determine_risk_level(fraud_probability)
            
            # Calculate confidence (based on distance from decision boundary)
            confidence = max(fraud_probability, 1 - fraud_probability)
            
            # Increment prediction counter
            self.prediction_count += 1
            
            # Prepare result
            result = {
                'fraud_probability': float(fraud_probability),
                'risk_level': risk_level,
                'confidence': float(confidence),
                'model_version': self.model_version,
                'prediction_id': str(uuid.uuid4()),
                'features_used': self.feature_names,
                'latency_ms': (time.time() - start_time) * 1000
            }
            
            logger.debug(f"Prediction made - Probability: {fraud_probability:.4f}, Risk: {risk_level}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _determine_risk_level(self, probability: float) -> str:
        """Determine risk level based on fraud probability"""
        if probability < 0.3:
            return RiskLevel.LOW
        elif probability < 0.7:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.is_model_loaded and self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_model_loaded:
            return {"status": "not_loaded"}
        
        return {
            "version": self.model_version,
            "model_type": self.model_metadata.get('model_type', 'unknown'),
            "feature_count": len(self.feature_names),
            "features": self.feature_names,
            "prediction_count": self.prediction_count,
            "load_time": self.load_time,
            "auc_score": self.model_metadata.get('auc_score'),
            "training_samples": self.model_metadata.get('training_samples'),
            "status": "loaded"
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            logger.info("🧹 Cleaning up fraud predictor...")
            
            # Clear model from memory
            self.model = None
            self.data_processor = None
            self.model_metadata = {}
            self.is_model_loaded = False
            
            # Clear model loader cache
            if self.model_loader:
                self.model_loader.clear_cache()
            
            logger.info("✅ Fraud predictor cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the predictor"""
        return {
            "status": "healthy" if self.is_loaded() else "unhealthy",
            "model_loaded": self.is_loaded(),
            "prediction_count": self.prediction_count,
            "model_version": self.model_version,
            "uptime_seconds": time.time() - self.load_time if self.load_time else 0
        }
    
    async def reload_model(
        self, 
        source: str = 'local',
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        version: Optional[str] = None
    ):
        """Reload model (for model updates)"""
        logger.info("🔄 Reloading fraud detection model...")
        
        # Cleanup current model
        await self.cleanup()
        
        # Load new model
        await self.load_model(source, model_path, model_name, version)
        
        logger.info("✅ Model reloaded successfully")
    
    async def list_available_models(self, source: str = 'mlflow') -> List[Dict[str, Any]]:
        """List available models from a source"""
        return await self.model_loader.list_available_models(source)
    
    def get_model_cache_info(self) -> Dict[str, Any]:
        """Get information about model cache"""
        return self.model_loader.get_cached_models()