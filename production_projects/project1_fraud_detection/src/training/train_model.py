#!/usr/bin/env python3
"""
Fraud Detection Model Training Pipeline
Production-grade training script with MLflow integration
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
import joblib
import logging
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from training.data_processor import FraudDataProcessor
from training.model_evaluator import ModelEvaluator
from shared.config import Config
from shared.logging import setup_logging

# Setup logging
logger = setup_logging(__name__)

class FraudModelTrainer:
    """Train fraud detection models with MLflow tracking"""
    
    def __init__(self, experiment_name: str = "fraud_detection"):
        # Set MLflow tracking URI and experiment
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        
        self.data_processor = FraudDataProcessor()
        self.evaluator = ModelEvaluator()
        
        # Ensure model directory exists
        os.makedirs("data/models", exist_ok=True)
        
    def train_and_compare_models(self, n_samples: int = 50000):
        """Train multiple models and select the best one"""
        
        logger.info("Starting fraud detection model training pipeline")
        
        # Generate and process data
        logger.info(f"Generating synthetic data with {n_samples} samples...")
        df = self.data_processor.generate_synthetic_data(n_samples=n_samples)
        logger.info(f"Generated data shape: {df.shape}")
        logger.info(f"Fraud rate: {df['is_fraud'].mean():.4f}")
        
        # Preprocess data
        df_processed = self.data_processor.preprocess_features(df, is_training=True)
        
        # Prepare features and target
        feature_columns = [
            'amount_log', 'hour', 'day_of_week', 'user_age', 'account_age_days',
            'merchant_category_encoded', 'is_weekend', 'is_night'
        ]
        
        X = df_processed[feature_columns]
        y = df_processed['is_fraud']
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Features: {feature_columns}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Train multiple models
        models = {
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=42,
                learning_rate=0.1
            )
        }
        
        best_model = None
        best_score = 0
        best_model_name = None
        
        # Parent run for model comparison
        with mlflow.start_run(run_name="model_comparison"):
            
            for model_name, model in models.items():
                
                # Child run for each model
                with mlflow.start_run(run_name=f"fraud_detection_{model_name}", nested=True):
                    
                    logger.info(f"Training {model_name}...")
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    predictions = model.predict(X_test)
                    probabilities = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    metrics = self.evaluator.calculate_metrics(y_test, predictions, probabilities)
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                    metrics['cv_auc_mean'] = cv_scores.mean()
                    metrics['cv_auc_std'] = cv_scores.std()
                    
                    # Log parameters
                    mlflow.log_params(model.get_params())
                    
                    # Log metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                    
                    # Log additional info
                    mlflow.log_param("n_train_samples", len(X_train))
                    mlflow.log_param("n_test_samples", len(X_test))
                    mlflow.log_param("feature_count", len(feature_columns))
                    
                    # Log model
                    mlflow.sklearn.log_model(
                        model,
                        f"fraud_model_{model_name}",
                        registered_model_name="fraud_detection_model",
                        signature=mlflow.models.infer_signature(X_train, predictions)
                    )
                    
                    # Log evaluation artifacts
                    self._log_evaluation_artifacts(y_test, predictions, probabilities, model_name)
                    
                    # Log feature importance for tree-based models
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'feature': feature_columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        feature_importance.to_csv(f'feature_importance_{model_name}.csv', index=False)
                        mlflow.log_artifact(f'feature_importance_{model_name}.csv')
                        
                        # Plot feature importance
                        plt.figure(figsize=(10, 6))
                        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
                        plt.title(f'Top 10 Feature Importance - {model_name}')
                        plt.tight_layout()
                        plt.savefig(f'feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
                        mlflow.log_artifact(f'feature_importance_{model_name}.png')
                        plt.close()
                    
                    logger.info(f"{model_name} completed:")
                    logger.info(f"  AUC: {metrics['auc_score']:.4f}")
                    logger.info(f"  Precision: {metrics['precision']:.4f}")
                    logger.info(f"  Recall: {metrics['recall']:.4f}")
                    logger.info(f"  F1: {metrics['f1_score']:.4f}")
                    
                    # Track best model
                    if metrics['auc_score'] > best_score:
                        best_score = metrics['auc_score']
                        best_model = model
                        best_model_name = model_name
            
            # Log best model info to parent run
            mlflow.log_metric("best_auc", best_score)
            mlflow.log_param("best_model", best_model_name)
        
        # Save best model and preprocessor
        if best_model:
            logger.info(f"Best model: {best_model_name} with AUC: {best_score:.4f}")
            
            # Save model and preprocessor
            joblib.dump(best_model, 'data/models/best_fraud_model.pkl')
            joblib.dump(self.data_processor, 'data/models/data_processor.pkl')
            
            # Save feature columns
            with open('data/models/feature_columns.txt', 'w') as f:
                f.write('\n'.join(feature_columns))
            
            logger.info("Model artifacts saved successfully")
            
            # Register best model in MLflow
            with mlflow.start_run(run_name=f"production_model_{best_model_name}"):
                mlflow.sklearn.log_model(
                    best_model,
                    "model",
                    registered_model_name="fraud_detection_production",
                    signature=mlflow.models.infer_signature(X_train, predictions)
                )
                mlflow.log_metric("production_auc", best_score)
                mlflow.log_param("model_type", best_model_name)
                mlflow.log_param("training_samples", len(X_train))
                
                # Log preprocessor as artifact
                joblib.dump(self.data_processor, 'data_processor.pkl')
                mlflow.log_artifact('data_processor.pkl')
        
        logger.info("Training pipeline completed successfully")
        return best_model_name, best_model, best_score
    
    def _log_evaluation_artifacts(self, y_true, y_pred, y_prob, model_name):
        """Log evaluation artifacts to MLflow"""
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Fraud', 'Fraud'],
                   yticklabels=['Not Fraud', 'Fraud'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact(f'confusion_matrix_{model_name}.png')
        plt.close()
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'roc_curve_{model_name}.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact(f'roc_curve_{model_name}.png')
        plt.close()
        
        # Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {ap_score:.4f})', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'pr_curve_{model_name}.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact(f'pr_curve_{model_name}.png')
        plt.close()
        
        # Classification Report
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'classification_report_{model_name}.csv')
        mlflow.log_artifact(f'classification_report_{model_name}.csv')
        
        # Clean up temporary files
        for file_pattern in [f'confusion_matrix_{model_name}.png', 
                           f'roc_curve_{model_name}.png',
                           f'pr_curve_{model_name}.png',
                           f'classification_report_{model_name}.csv',
                           f'feature_importance_{model_name}.csv',
                           f'feature_importance_{model_name}.png']:
            if os.path.exists(file_pattern):
                os.remove(file_pattern)

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--samples', type=int, default=50000, 
                       help='Number of samples to generate (default: 50000)')
    parser.add_argument('--experiment', type=str, default='fraud_detection',
                       help='MLflow experiment name (default: fraud_detection)')
    
    args = parser.parse_args()
    
    try:
        trainer = FraudModelTrainer(experiment_name=args.experiment)
        model_name, model, score = trainer.train_and_compare_models(n_samples=args.samples)
        
        print("\n🎉 Training completed successfully!")
        print(f"📊 Best model: {model_name}")
        print(f"📈 AUC Score: {score:.4f}")
        print(f"💾 Model saved to: data/models/")
        print(f"🔬 View experiments: {Config.MLFLOW_TRACKING_URI}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()