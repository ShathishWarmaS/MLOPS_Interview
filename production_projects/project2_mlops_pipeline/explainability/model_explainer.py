"""
Advanced Model Explainability and Interpretability Tools
Comprehensive framework for understanding ML model predictions
"""

import asyncio
import logging
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import threading
import pickle
import warnings

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# ML and explainability libraries
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Local imports
from serving.api.predictor import ModelPredictor

logger = logging.getLogger(__name__)

@dataclass
class ExplanationResult:
    """Model explanation result"""
    prediction_id: str
    model_version: str
    prediction: float
    confidence: float
    feature_values: Dict[str, Any]
    explanations: Dict[str, Any]
    explanation_methods: List[str]
    processing_time_ms: float
    timestamp: float

@dataclass
class GlobalExplanation:
    """Global model explanation"""
    model_version: str
    feature_importance: Dict[str, float]
    feature_interactions: Dict[str, float]
    model_complexity: Dict[str, Any]
    explanation_method: str
    confidence_score: float
    timestamp: float

@dataclass
class FeatureContribution:
    """Individual feature contribution"""
    feature_name: str
    feature_value: Any
    contribution: float
    contribution_normalized: float
    rank: int
    confidence: float

class SHAPExplainer:
    """SHAP-based model explainer"""
    
    def __init__(self, model, background_data: pd.DataFrame):
        self.model = model
        self.background_data = background_data
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type"""
        try:
            # Try TreeExplainer first (for tree-based models)
            self.explainer = shap.TreeExplainer(self.model)
            self.explainer_type = "tree"
        except Exception:
            try:
                # Try LinearExplainer for linear models
                self.explainer = shap.LinearExplainer(self.model, self.background_data)
                self.explainer_type = "linear"
            except Exception:
                # Fall back to KernelExplainer (model-agnostic)
                background_sample = shap.sample(self.background_data, min(100, len(self.background_data)))
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background_sample)
                self.explainer_type = "kernel"
        
        logger.info(f"Initialized SHAP {self.explainer_type} explainer")
    
    def explain_instance(self, instance: pd.DataFrame) -> Dict[str, Any]:
        """Explain single prediction instance"""
        try:
            if self.explainer_type == "kernel":
                shap_values = self.explainer.shap_values(instance)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification, take positive class
            else:
                shap_values = self.explainer.shap_values(instance)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            
            # Get expected value
            if hasattr(self.explainer, 'expected_value'):
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
            else:
                expected_value = 0.0
            
            # Create feature contributions
            feature_names = instance.columns.tolist()
            contributions = {}
            
            for i, feature in enumerate(feature_names):
                contributions[feature] = {
                    "value": float(instance.iloc[0, i]),
                    "contribution": float(shap_values[0, i]),
                    "abs_contribution": float(abs(shap_values[0, i]))
                }
            
            return {
                "shap_values": shap_values.tolist(),
                "expected_value": float(expected_value),
                "feature_contributions": contributions,
                "explainer_type": self.explainer_type
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {"error": str(e)}

class LIMEExplainer:
    """LIME-based model explainer"""
    
    def __init__(self, model, training_data: pd.DataFrame, feature_names: List[str]):
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        
        # Initialize LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data.values,
            feature_names=feature_names,
            class_names=['Negative', 'Positive'],
            mode='classification',
            discretize_continuous=True
        )
    
    def explain_instance(self, instance: pd.DataFrame, num_features: int = 10) -> Dict[str, Any]:
        """Explain single prediction instance"""
        try:
            # Get prediction function
            def predict_fn(x):
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(x)
                else:
                    preds = self.model.predict(x)
                    # Convert to probabilities for binary classification
                    probs = np.zeros((len(preds), 2))
                    probs[:, 1] = preds
                    probs[:, 0] = 1 - preds
                    return probs
            
            # Generate explanation
            explanation = self.explainer.explain_instance(
                instance.values[0],
                predict_fn,
                num_features=num_features
            )
            
            # Extract feature contributions
            contributions = {}
            for feature_idx, contribution in explanation.as_list():
                feature_parts = feature_idx.split(' ')
                feature_name = feature_parts[0] if feature_parts else f"feature_{feature_idx}"
                
                # Find actual feature name
                actual_feature = None
                for fname in self.feature_names:
                    if fname in feature_idx or feature_idx in fname:
                        actual_feature = fname
                        break
                
                if actual_feature:
                    contributions[actual_feature] = {
                        "contribution": float(contribution),
                        "description": feature_idx
                    }
                else:
                    contributions[feature_name] = {
                        "contribution": float(contribution),
                        "description": feature_idx
                    }
            
            return {
                "lime_explanation": explanation.as_list(),
                "feature_contributions": contributions,
                "local_prediction": explanation.local_pred[1] if len(explanation.local_pred) > 1 else explanation.local_pred[0],
                "intercept": float(explanation.intercept[1]) if len(explanation.intercept) > 1 else float(explanation.intercept[0])
            }
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {"error": str(e)}

class PermutationExplainer:
    """Permutation importance-based explainer"""
    
    def __init__(self, model, X_val: pd.DataFrame, y_val: pd.Series):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.feature_importance = None
        self._calculate_importance()
    
    def _calculate_importance(self):
        """Calculate permutation importance"""
        try:
            importance_result = permutation_importance(
                self.model, self.X_val, self.y_val,
                n_repeats=10, random_state=42, n_jobs=-1
            )
            
            self.feature_importance = {
                'importances_mean': importance_result.importances_mean,
                'importances_std': importance_result.importances_std,
                'feature_names': self.X_val.columns.tolist()
            }
            
        except Exception as e:
            logger.error(f"Permutation importance calculation failed: {e}")
            self.feature_importance = None
    
    def explain_instance(self, instance: pd.DataFrame) -> Dict[str, Any]:
        """Explain instance using permutation importance"""
        if self.feature_importance is None:
            return {"error": "Permutation importance not available"}
        
        try:
            # Calculate feature contributions based on importance and feature values
            contributions = {}
            
            for i, feature in enumerate(self.feature_importance['feature_names']):
                importance = self.feature_importance['importances_mean'][i]
                importance_std = self.feature_importance['importances_std'][i]
                feature_value = instance[feature].iloc[0]
                
                # Normalize feature value (simple min-max based on validation data)
                feature_min = self.X_val[feature].min()
                feature_max = self.X_val[feature].max()
                
                if feature_max != feature_min:
                    normalized_value = (feature_value - feature_min) / (feature_max - feature_min)
                else:
                    normalized_value = 0.5
                
                # Calculate contribution (importance * normalized value)
                contribution = importance * (normalized_value - 0.5) * 2  # Scale to [-1, 1]
                
                contributions[feature] = {
                    "value": float(feature_value),
                    "importance": float(importance),
                    "importance_std": float(importance_std),
                    "contribution": float(contribution),
                    "normalized_value": float(normalized_value)
                }
            
            return {
                "feature_contributions": contributions,
                "global_importance": {
                    feature: float(imp) for feature, imp in 
                    zip(self.feature_importance['feature_names'], 
                        self.feature_importance['importances_mean'])
                }
            }
            
        except Exception as e:
            logger.error(f"Permutation explanation failed: {e}")
            return {"error": str(e)}

class ModelExplainer:
    """Comprehensive model explainer framework"""
    
    def __init__(self, 
                 model_predictor: ModelPredictor,
                 reference_data: pd.DataFrame,
                 target_column: str = None):
        
        self.predictor = model_predictor
        self.reference_data = reference_data
        self.target_column = target_column
        
        # Feature information
        self.feature_names = [col for col in reference_data.columns if col != target_column]
        self.X_reference = reference_data[self.feature_names] if target_column else reference_data
        self.y_reference = reference_data[target_column] if target_column else None
        
        # Explainers
        self.explainers = {}
        self.global_explanations = {}
        
        # Storage
        self.explanation_cache = {}
        self.lock = threading.Lock()
        
        # Executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    async def initialize(self):
        """Initialize explainers"""
        try:
            logger.info("Initializing model explainers...")
            
            # Get the underlying model from predictor
            model = await self._get_model_from_predictor()
            
            if model is None:
                logger.warning("Could not extract model from predictor. Limited explanations available.")
                return
            
            # Initialize SHAP explainer
            try:
                self.explainers['shap'] = SHAPExplainer(model, self.X_reference)
                logger.info("✅ SHAP explainer initialized")
            except Exception as e:
                logger.warning(f"SHAP explainer initialization failed: {e}")
            
            # Initialize LIME explainer
            try:
                self.explainers['lime'] = LIMEExplainer(model, self.X_reference, self.feature_names)
                logger.info("✅ LIME explainer initialized")
            except Exception as e:
                logger.warning(f"LIME explainer initialization failed: {e}")
            
            # Initialize Permutation explainer (if target data available)
            if self.y_reference is not None:
                try:
                    self.explainers['permutation'] = PermutationExplainer(
                        model, self.X_reference, self.y_reference
                    )
                    logger.info("✅ Permutation explainer initialized")
                except Exception as e:
                    logger.warning(f"Permutation explainer initialization failed: {e}")
            
            # Calculate global explanations
            await self._calculate_global_explanations()
            
            logger.info("Model explainers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize explainers: {e}")
            raise
    
    async def _get_model_from_predictor(self):
        """Extract the underlying model from predictor"""
        try:
            # Try to access the model directly
            if hasattr(self.predictor, 'model') and self.predictor.model is not None:
                return self.predictor.model
            
            # Try to load the model
            await self.predictor.load_model()
            if hasattr(self.predictor, 'model') and self.predictor.model is not None:
                return self.predictor.model
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not extract model: {e}")
            return None
    
    async def _calculate_global_explanations(self):
        """Calculate global model explanations"""
        try:
            # SHAP global explanation
            if 'shap' in self.explainers:
                await self._calculate_shap_global_explanation()
            
            # Permutation global explanation
            if 'permutation' in self.explainers:
                await self._calculate_permutation_global_explanation()
            
        except Exception as e:
            logger.error(f"Failed to calculate global explanations: {e}")
    
    async def _calculate_shap_global_explanation(self):
        """Calculate SHAP global explanation"""
        try:
            explainer = self.explainers['shap']
            
            # Sample data for global explanation
            sample_size = min(100, len(self.X_reference))
            sample_data = self.X_reference.sample(n=sample_size, random_state=42)
            
            # Calculate SHAP values for sample
            if explainer.explainer_type == "kernel":
                shap_values = explainer.explainer.shap_values(sample_data)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                shap_values = explainer.explainer.shap_values(sample_data)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            
            # Calculate mean absolute SHAP values (feature importance)
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            feature_importance = {
                feature: float(importance) 
                for feature, importance in zip(self.feature_names, mean_abs_shap)
            }
            
            self.global_explanations['shap'] = GlobalExplanation(
                model_version=getattr(self.predictor, 'model_version', 'unknown'),
                feature_importance=feature_importance,
                feature_interactions={},  # Could add interaction analysis
                model_complexity={"shap_explainer_type": explainer.explainer_type},
                explanation_method="shap",
                confidence_score=0.9,  # SHAP is generally reliable
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"SHAP global explanation failed: {e}")
    
    async def _calculate_permutation_global_explanation(self):
        """Calculate permutation global explanation"""
        try:
            explainer = self.explainers['permutation']
            
            if explainer.feature_importance is None:
                return
            
            feature_importance = {
                feature: float(importance) 
                for feature, importance in zip(
                    explainer.feature_importance['feature_names'],
                    explainer.feature_importance['importances_mean']
                )
            }
            
            # Calculate model complexity metrics
            importance_values = list(feature_importance.values())
            complexity_metrics = {
                "n_features": len(self.feature_names),
                "importance_concentration": float(np.std(importance_values) / np.mean(importance_values)),
                "effective_features": int(sum(1 for imp in importance_values if imp > 0.01))
            }
            
            self.global_explanations['permutation'] = GlobalExplanation(
                model_version=getattr(self.predictor, 'model_version', 'unknown'),
                feature_importance=feature_importance,
                feature_interactions={},
                model_complexity=complexity_metrics,
                explanation_method="permutation",
                confidence_score=0.8,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Permutation global explanation failed: {e}")
    
    async def explain_prediction(self, 
                                features: Dict[str, Any],
                                methods: List[str] = None,
                                include_visualization: bool = False) -> ExplanationResult:
        """Explain a single prediction"""
        try:
            start_time = time.time()
            
            # Default methods
            if methods is None:
                methods = list(self.explainers.keys())
            
            # Create instance DataFrame
            instance_df = pd.DataFrame([features])[self.feature_names]
            
            # Get prediction from predictor
            prediction_result = await self.predictor.predict(features)
            
            # Generate cache key
            cache_key = self._generate_cache_key(features, methods)
            
            # Check cache
            with self.lock:
                if cache_key in self.explanation_cache:
                    cached_result = self.explanation_cache[cache_key]
                    logger.debug("Using cached explanation")
                    return cached_result
            
            # Run explanations
            explanations = {}
            
            for method in methods:
                if method in self.explainers:
                    try:
                        explanation = await asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            self.explainers[method].explain_instance,
                            instance_df
                        )
                        explanations[method] = explanation
                        
                    except Exception as e:
                        logger.warning(f"Explanation method {method} failed: {e}")
                        explanations[method] = {"error": str(e)}
            
            # Create unified feature contributions
            unified_contributions = self._unify_explanations(explanations, features)
            
            # Create explanation result
            processing_time = (time.time() - start_time) * 1000
            
            result = ExplanationResult(
                prediction_id=str(int(time.time() * 1000)),
                model_version=getattr(self.predictor, 'model_version', 'unknown'),
                prediction=prediction_result.prediction,
                confidence=prediction_result.confidence,
                feature_values=features,
                explanations={
                    "methods": explanations,
                    "unified_contributions": unified_contributions,
                    "summary": self._generate_explanation_summary(unified_contributions)
                },
                explanation_methods=list(explanations.keys()),
                processing_time_ms=processing_time,
                timestamp=time.time()
            )
            
            # Cache result
            with self.lock:
                self.explanation_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction explanation failed: {e}")
            raise
    
    def _generate_cache_key(self, features: Dict[str, Any], methods: List[str]) -> str:
        """Generate cache key for explanation"""
        features_str = json.dumps(features, sort_keys=True)
        methods_str = "|".join(sorted(methods))
        return f"{hash(features_str)}_{hash(methods_str)}"
    
    def _unify_explanations(self, 
                           explanations: Dict[str, Any], 
                           features: Dict[str, Any]) -> List[FeatureContribution]:
        """Unify explanations from different methods"""
        feature_contributions = {}
        
        # Collect contributions from all methods
        for method, explanation in explanations.items():
            if "error" in explanation:
                continue
            
            method_contributions = explanation.get("feature_contributions", {})
            
            for feature, contrib_data in method_contributions.items():
                if feature not in feature_contributions:
                    feature_contributions[feature] = {
                        "contributions": [],
                        "value": features.get(feature, contrib_data.get("value", 0))
                    }
                
                # Extract contribution value
                if isinstance(contrib_data, dict):
                    contribution = contrib_data.get("contribution", 0)
                else:
                    contribution = contrib_data
                
                feature_contributions[feature]["contributions"].append({
                    "method": method,
                    "contribution": float(contribution)
                })
        
        # Calculate unified contributions
        unified = []
        
        for feature, data in feature_contributions.items():
            if not data["contributions"]:
                continue
            
            # Average contributions across methods
            contributions = [c["contribution"] for c in data["contributions"]]
            mean_contribution = np.mean(contributions)
            std_contribution = np.std(contributions) if len(contributions) > 1 else 0
            
            # Calculate confidence based on agreement between methods
            confidence = 1.0 - min(std_contribution / max(abs(mean_contribution), 1e-6), 1.0)
            
            unified.append(FeatureContribution(
                feature_name=feature,
                feature_value=data["value"],
                contribution=float(mean_contribution),
                contribution_normalized=0.0,  # Will be calculated after sorting
                rank=0,  # Will be calculated after sorting
                confidence=float(confidence)
            ))
        
        # Sort by absolute contribution and assign ranks
        unified.sort(key=lambda x: abs(x.contribution), reverse=True)
        
        # Normalize contributions and assign ranks
        max_abs_contribution = max([abs(c.contribution) for c in unified], default=1.0)
        
        for i, contrib in enumerate(unified):
            contrib.rank = i + 1
            contrib.contribution_normalized = contrib.contribution / max_abs_contribution
        
        return unified
    
    def _generate_explanation_summary(self, contributions: List[FeatureContribution]) -> Dict[str, Any]:
        """Generate explanation summary"""
        if not contributions:
            return {"message": "No feature contributions available"}
        
        # Top positive and negative contributors
        positive_contribs = [c for c in contributions if c.contribution > 0]
        negative_contribs = [c for c in contributions if c.contribution < 0]
        
        summary = {
            "top_positive_features": [
                {
                    "feature": c.feature_name,
                    "contribution": c.contribution,
                    "value": c.feature_value,
                    "confidence": c.confidence
                }
                for c in positive_contribs[:3]
            ],
            "top_negative_features": [
                {
                    "feature": c.feature_name,
                    "contribution": c.contribution,
                    "value": c.feature_value,
                    "confidence": c.confidence
                }
                for c in negative_contribs[:3]
            ],
            "total_positive_contribution": sum(c.contribution for c in positive_contribs),
            "total_negative_contribution": sum(c.contribution for c in negative_contribs),
            "feature_count": len(contributions),
            "confidence_score": np.mean([c.confidence for c in contributions])
        }
        
        return summary
    
    async def generate_explanation_report(self, 
                                        features: Dict[str, Any],
                                        output_path: str = None) -> str:
        """Generate comprehensive explanation report"""
        try:
            # Get explanation
            explanation = await self.explain_prediction(features)
            
            # Create report
            report = {
                "prediction_details": {
                    "prediction": explanation.prediction,
                    "confidence": explanation.confidence,
                    "model_version": explanation.model_version,
                    "timestamp": datetime.fromtimestamp(explanation.timestamp).isoformat()
                },
                "feature_values": explanation.feature_values,
                "explanations": explanation.explanations,
                "global_context": {
                    method: asdict(global_exp) 
                    for method, global_exp in self.global_explanations.items()
                },
                "processing_time_ms": explanation.processing_time_ms,
                "explanation_methods": explanation.explanation_methods
            }
            
            # Export to file
            if not output_path:
                output_path = f"explanation_report_{explanation.prediction_id}.json"
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Explanation report exported to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate explanation report: {e}")
            raise
    
    async def get_global_explanations(self) -> Dict[str, GlobalExplanation]:
        """Get global model explanations"""
        return self.global_explanations.copy()
    
    async def compare_explanations(self, 
                                 features_list: List[Dict[str, Any]],
                                 methods: List[str] = None) -> Dict[str, Any]:
        """Compare explanations across multiple instances"""
        try:
            explanations = []
            
            for features in features_list:
                explanation = await self.explain_prediction(features, methods)
                explanations.append(explanation)
            
            # Analyze consistency
            consistency_analysis = self._analyze_explanation_consistency(explanations)
            
            return {
                "explanations": [asdict(exp) for exp in explanations],
                "consistency_analysis": consistency_analysis,
                "comparison_summary": self._generate_comparison_summary(explanations)
            }
            
        except Exception as e:
            logger.error(f"Explanation comparison failed: {e}")
            raise
    
    def _analyze_explanation_consistency(self, explanations: List[ExplanationResult]) -> Dict[str, Any]:
        """Analyze consistency across explanations"""
        if len(explanations) < 2:
            return {"message": "At least 2 explanations required for consistency analysis"}
        
        # Extract feature contributions
        all_contributions = {}
        
        for exp in explanations:
            unified_contribs = exp.explanations.get("unified_contributions", [])
            for contrib in unified_contribs:
                feature = contrib.feature_name
                if feature not in all_contributions:
                    all_contributions[feature] = []
                all_contributions[feature].append(contrib.contribution)
        
        # Calculate consistency metrics
        consistency_scores = {}
        
        for feature, contributions in all_contributions.items():
            if len(contributions) > 1:
                mean_contrib = np.mean(contributions)
                std_contrib = np.std(contributions)
                cv = std_contrib / max(abs(mean_contrib), 1e-6)  # Coefficient of variation
                consistency_scores[feature] = 1.0 - min(cv, 1.0)
            else:
                consistency_scores[feature] = 1.0
        
        overall_consistency = np.mean(list(consistency_scores.values()))
        
        return {
            "overall_consistency": float(overall_consistency),
            "feature_consistency": consistency_scores,
            "most_consistent_features": sorted(
                consistency_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            "least_consistent_features": sorted(
                consistency_scores.items(), 
                key=lambda x: x[1]
            )[:5]
        }
    
    def _generate_comparison_summary(self, explanations: List[ExplanationResult]) -> Dict[str, Any]:
        """Generate summary of explanation comparison"""
        predictions = [exp.prediction for exp in explanations]
        confidences = [exp.confidence for exp in explanations]
        
        return {
            "n_explanations": len(explanations),
            "prediction_stats": {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "range": [float(np.min(predictions)), float(np.max(predictions))]
            },
            "confidence_stats": {
                "mean": float(np.mean(confidences)),
                "std": float(np.std(confidences)),
                "range": [float(np.min(confidences)), float(np.max(confidences))]
            },
            "processing_time_stats": {
                "mean_ms": float(np.mean([exp.processing_time_ms for exp in explanations])),
                "max_ms": float(np.max([exp.processing_time_ms for exp in explanations]))
            }
        }

def main():
    """Main function for explainability demonstration"""
    
    async def run_explainability_demo():
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic dataset
        data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.uniform(0, 10, n_samples),
            'feature_3': np.random.choice([0, 1], n_samples),
            'feature_4': np.random.exponential(1, n_samples),
            'feature_5': np.random.normal(5, 2, n_samples)
        })
        
        # Create target variable
        data['target'] = (
            0.3 * data['feature_1'] + 
            0.2 * data['feature_2'] - 
            0.5 * data['feature_3'] + 
            0.1 * data['feature_4'] + 
            np.random.normal(0, 0.1, n_samples)
        ) > 0
        data['target'] = data['target'].astype(int)
        
        # Train a simple model (in production, this would be your trained model)
        X = data.drop('target', axis=1)
        y = data['target']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Create mock predictor
        class MockPredictor:
            def __init__(self, model):
                self.model = model
                self.model_version = "demo_v1.0"
                
            async def predict(self, features):
                feature_df = pd.DataFrame([features])
                prediction = self.model.predict_proba(feature_df)[0, 1]
                confidence = max(self.model.predict_proba(feature_df)[0])
                
                return type('PredictionResult', (), {
                    'prediction': prediction,
                    'confidence': confidence
                })()
        
        predictor = MockPredictor(model)
        
        # Initialize explainer
        explainer = ModelExplainer(predictor, data, 'target')
        await explainer.initialize()
        
        # Test explanation
        test_features = {
            'feature_1': 1.5,
            'feature_2': 7.0,
            'feature_3': 1,
            'feature_4': 0.8,
            'feature_5': 4.2
        }
        
        print("Generating explanation...")
        explanation = await explainer.explain_prediction(test_features)
        
        print(f"\nPrediction: {explanation.prediction:.3f}")
        print(f"Confidence: {explanation.confidence:.3f}")
        print(f"Processing time: {explanation.processing_time_ms:.1f}ms")
        
        print("\nTop feature contributions:")
        unified_contribs = explanation.explanations["unified_contributions"]
        for contrib in unified_contribs[:5]:
            print(f"  {contrib.feature_name}: {contrib.contribution:.3f} "
                  f"(value: {contrib.feature_value}, confidence: {contrib.confidence:.2f})")
        
        # Generate report
        report_path = await explainer.generate_explanation_report(test_features)
        print(f"\nDetailed report saved to: {report_path}")
        
        # Get global explanations
        global_explanations = await explainer.get_global_explanations()
        
        print(f"\nGlobal explanations available: {list(global_explanations.keys())}")
        
        for method, global_exp in global_explanations.items():
            print(f"\n{method.upper()} Global Feature Importance:")
            sorted_features = sorted(
                global_exp.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for feature, importance in sorted_features[:5]:
                print(f"  {feature}: {importance:.3f}")
    
    # Run demo
    asyncio.run(run_explainability_demo())

if __name__ == "__main__":
    main()