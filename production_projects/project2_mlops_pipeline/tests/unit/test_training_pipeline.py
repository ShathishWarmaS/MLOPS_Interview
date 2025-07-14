"""
Unit tests for training pipeline
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from training.pipelines.training_pipeline import TrainingPipeline
from training.pipelines.nodes import DataProcessor


@pytest.mark.unit
class TestTrainingPipeline:
    """Test suite for TrainingPipeline class."""
    
    def test_init(self):
        """Test TrainingPipeline initialization."""
        pipeline = TrainingPipeline()
        assert hasattr(pipeline, 'models')
        assert isinstance(pipeline.models, dict)
        assert 'random_forest' in pipeline.models
        assert 'logistic_regression' in pipeline.models
        assert 'gradient_boosting' in pipeline.models
    
    def test_get_model_by_type_random_forest(self, training_pipeline):
        """Test getting random forest model."""
        model = training_pipeline._get_model_by_type('random_forest', {'random_state': 42})
        assert isinstance(model, RandomForestClassifier)
        assert model.random_state == 42
    
    def test_get_model_by_type_logistic_regression(self, training_pipeline):
        """Test getting logistic regression model."""
        model = training_pipeline._get_model_by_type('logistic_regression', {'random_state': 42})
        assert isinstance(model, LogisticRegression)
        assert model.random_state == 42
    
    def test_get_model_by_type_gradient_boosting(self, training_pipeline):
        """Test getting gradient boosting model."""
        model = training_pipeline._get_model_by_type('gradient_boosting', {'random_state': 42})
        assert isinstance(model, GradientBoostingClassifier)
        assert model.random_state == 42
    
    def test_get_model_by_type_invalid(self, training_pipeline):
        """Test getting invalid model type."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            training_pipeline._get_model_by_type('invalid_model', {})
    
    @patch('training.pipelines.training_pipeline.TrainingPipeline._prepare_features')
    def test_prepare_data(self, mock_prepare_features, training_pipeline, sample_data):
        """Test data preparation."""
        # Mock the feature preparation
        features = sample_data.drop('target', axis=1)
        target = sample_data['target']
        mock_prepare_features.return_value = features
        
        X, y = training_pipeline._prepare_data(sample_data, 'target', ['feature_1', 'feature_2'])
        
        mock_prepare_features.assert_called_once_with(sample_data, ['feature_1', 'feature_2'])
        pd.testing.assert_frame_equal(X, features)
        pd.testing.assert_series_equal(y, target, check_names=False)
    
    def test_prepare_features_subset(self, training_pipeline, sample_data):
        """Test feature preparation with feature subset."""
        selected_features = ['feature_1', 'feature_2']
        result = training_pipeline._prepare_features(sample_data, selected_features)
        
        assert list(result.columns) == selected_features
        assert len(result) == len(sample_data)
    
    def test_prepare_features_all(self, training_pipeline, sample_data):
        """Test feature preparation with all features."""
        # Remove target column for this test
        feature_data = sample_data.drop('target', axis=1)
        result = training_pipeline._prepare_features(feature_data, None)
        
        pd.testing.assert_frame_equal(result, feature_data)
    
    def test_train_single_model(self, training_pipeline, sample_data):
        """Test training a single model."""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        model_config = {
            'type': 'random_forest',
            'n_estimators': 10,
            'random_state': 42
        }
        
        model, metrics = training_pipeline._train_single_model(X, y, model_config)
        
        assert isinstance(model, RandomForestClassifier)
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        
        # Check metrics
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert 0 <= metrics[metric] <= 1
    
    def test_train_single_model_with_validation(self, training_pipeline, sample_data):
        """Test training a single model with validation data."""
        # Split data
        train_data = sample_data.iloc[:800]
        val_data = sample_data.iloc[800:]
        
        X_train = train_data.drop('target', axis=1)
        y_train = train_data['target']
        X_val = val_data.drop('target', axis=1)
        y_val = val_data['target']
        
        model_config = {
            'type': 'logistic_regression',
            'random_state': 42
        }
        
        model, metrics = training_pipeline._train_single_model(
            X_train, y_train, model_config, X_val, y_val
        )
        
        assert isinstance(model, LogisticRegression)
        
        # Should have both training and validation metrics
        assert 'train_accuracy' in metrics
        assert 'val_accuracy' in metrics
    
    def test_evaluate_model(self, training_pipeline, sample_data):
        """Test model evaluation."""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        metrics = training_pipeline._evaluate_model(model, X, y)
        
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert 0 <= metrics[metric] <= 1
    
    def test_compare_models(self, training_pipeline):
        """Test model comparison."""
        models_with_metrics = {
            'model_1': {
                'model': Mock(),
                'metrics': {'accuracy': 0.85, 'f1_score': 0.83}
            },
            'model_2': {
                'model': Mock(),
                'metrics': {'accuracy': 0.90, 'f1_score': 0.88}
            },
            'model_3': {
                'model': Mock(),
                'metrics': {'accuracy': 0.82, 'f1_score': 0.85}
            }
        }
        
        best_model_name, best_model_info = training_pipeline._compare_models(
            models_with_metrics, 'accuracy'
        )
        
        assert best_model_name == 'model_2'
        assert best_model_info['metrics']['accuracy'] == 0.90
    
    def test_compare_models_f1_score(self, training_pipeline):
        """Test model comparison using F1 score."""
        models_with_metrics = {
            'model_1': {
                'model': Mock(),
                'metrics': {'accuracy': 0.85, 'f1_score': 0.90}  # Higher F1
            },
            'model_2': {
                'model': Mock(),
                'metrics': {'accuracy': 0.90, 'f1_score': 0.88}  # Higher accuracy
            }
        }
        
        best_model_name, best_model_info = training_pipeline._compare_models(
            models_with_metrics, 'f1_score'
        )
        
        assert best_model_name == 'model_1'
        assert best_model_info['metrics']['f1_score'] == 0.90
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_metric')
    @patch('mlflow.log_param')
    @patch('mlflow.sklearn.log_model')
    def test_log_to_mlflow(self, mock_log_model, mock_log_param, mock_log_metric, 
                          mock_start_run, training_pipeline):
        """Test MLflow logging."""
        # Setup mocks
        mock_run_context = MagicMock()
        mock_start_run.return_value.__enter__ = Mock(return_value=mock_run_context)
        mock_start_run.return_value.__exit__ = Mock(return_value=None)
        
        model = Mock()
        metrics = {'accuracy': 0.85, 'f1_score': 0.83}
        model_config = {'type': 'random_forest', 'n_estimators': 100}
        
        training_pipeline._log_to_mlflow(model, metrics, model_config, 'test_model')
        
        # Verify MLflow calls
        mock_start_run.assert_called_once()
        mock_log_param.assert_called()
        mock_log_metric.assert_called()
        mock_log_model.assert_called_once()
    
    @patch('training.pipelines.training_pipeline.DataProcessor')
    def test_run_training_pipeline_basic(self, mock_data_processor_class, 
                                        training_pipeline, training_config):
        """Test basic training pipeline execution."""
        # Mock data processor
        mock_processor = Mock()
        mock_data_processor_class.return_value = mock_processor
        
        # Mock generated data
        mock_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0, 1, 100),
            'target': np.random.binomial(1, 0.3, 100)
        })
        mock_processor.generate_synthetic_data.return_value = mock_data
        mock_processor.preprocess_features.return_value = mock_data
        
        # Mock train/val/test split
        train_data = mock_data.iloc[:60]
        val_data = mock_data.iloc[60:80]
        test_data = mock_data.iloc[80:]
        mock_processor.split_data.return_value = (train_data, val_data, test_data)
        
        with patch.object(training_pipeline, '_log_to_mlflow'):
            result = training_pipeline.run_training_pipeline(training_config)
        
        assert 'best_model' in result
        assert 'best_model_name' in result
        assert 'metrics' in result
        assert 'comparison' in result
        
        # Verify data processor calls
        mock_processor.generate_synthetic_data.assert_called_once()
        mock_processor.preprocess_features.assert_called_once()
        mock_processor.split_data.assert_called_once()
    
    @patch('training.pipelines.training_pipeline.DataProcessor')
    def test_run_training_pipeline_multiple_models(self, mock_data_processor_class, 
                                                   training_pipeline):
        """Test training pipeline with multiple models."""
        # Mock data processor
        mock_processor = Mock()
        mock_data_processor_class.return_value = mock_processor
        
        # Mock data
        mock_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0, 1, 100),
            'target': np.random.binomial(1, 0.3, 100)
        })
        mock_processor.generate_synthetic_data.return_value = mock_data
        mock_processor.preprocess_features.return_value = mock_data
        
        # Mock splits
        train_data = mock_data.iloc[:60]
        val_data = mock_data.iloc[60:80]
        test_data = mock_data.iloc[80:]
        mock_processor.split_data.return_value = (train_data, val_data, test_data)
        
        # Configuration for multiple models
        config = {
            'data_config': {
                'n_samples': 100,
                'test_size': 0.2,
                'val_size': 0.2,
                'random_state': 42
            },
            'model_configs': [
                {
                    'name': 'rf_model',
                    'type': 'random_forest',
                    'n_estimators': 10,
                    'random_state': 42
                },
                {
                    'name': 'lr_model',
                    'type': 'logistic_regression',
                    'random_state': 42
                }
            ],
            'feature_config': {
                'numerical_features': ['feature_1', 'feature_2']
            }
        }
        
        with patch.object(training_pipeline, '_log_to_mlflow'):
            result = training_pipeline.run_training_pipeline(config)
        
        assert 'best_model' in result
        assert 'best_model_name' in result
        assert 'comparison' in result
        
        # Should have results for multiple models
        comparison = result['comparison']
        assert len(comparison) == 2
        assert 'rf_model' in comparison
        assert 'lr_model' in comparison
    
    def test_run_training_pipeline_missing_target(self, training_pipeline):
        """Test training pipeline with missing target column."""
        config = {
            'data_config': {'n_samples': 100},
            'model_config': {'type': 'random_forest'},
            'target_column': 'missing_target'
        }
        
        with pytest.raises(KeyError):
            training_pipeline.run_training_pipeline(config)
    
    def test_run_training_pipeline_invalid_model_type(self, training_pipeline):
        """Test training pipeline with invalid model type."""
        config = {
            'data_config': {'n_samples': 100},
            'model_config': {'type': 'invalid_model'},
            'target_column': 'target'
        }
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            training_pipeline.run_training_pipeline(config)
    
    @patch('training.pipelines.training_pipeline.DataProcessor')
    def test_run_training_pipeline_no_validation_data(self, mock_data_processor_class, 
                                                     training_pipeline):
        """Test training pipeline without validation data."""
        # Mock data processor
        mock_processor = Mock()
        mock_data_processor_class.return_value = mock_processor
        
        # Mock data
        mock_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'target': np.random.binomial(1, 0.3, 100)
        })
        mock_processor.generate_synthetic_data.return_value = mock_data
        mock_processor.preprocess_features.return_value = mock_data
        
        # Mock splits - no validation data
        train_data = mock_data.iloc[:80]
        test_data = mock_data.iloc[80:]
        mock_processor.split_data.return_value = (train_data, None, test_data)
        
        config = {
            'data_config': {'n_samples': 100},
            'model_config': {'type': 'random_forest', 'n_estimators': 10},
            'target_column': 'target'
        }
        
        with patch.object(training_pipeline, '_log_to_mlflow'):
            result = training_pipeline.run_training_pipeline(config)
        
        assert 'best_model' in result
        assert result['best_model'] is not None
    
    def test_cross_validation(self, training_pipeline, sample_data):
        """Test cross-validation functionality."""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        model_config = {
            'type': 'random_forest',
            'n_estimators': 10,
            'random_state': 42
        }
        
        cv_scores = training_pipeline._cross_validate_model(X, y, model_config, cv=3)
        
        assert len(cv_scores) == 3
        for score in cv_scores:
            assert isinstance(score, (int, float))
            assert 0 <= score <= 1
    
    def test_feature_importance_extraction(self, training_pipeline, sample_data):
        """Test feature importance extraction."""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        # Train a tree-based model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        importance = training_pipeline._get_feature_importance(model, X.columns)
        
        assert len(importance) == len(X.columns)
        assert all(isinstance(imp, (int, float)) for imp in importance.values())
        assert all(imp >= 0 for imp in importance.values())
        assert sum(importance.values()) == pytest.approx(1.0, abs=1e-6)
    
    def test_hyperparameter_tuning(self, training_pipeline, sample_data):
        """Test hyperparameter tuning."""
        X = sample_data.drop('target', axis=1)
        y = sample_data['target']
        
        param_grid = {
            'n_estimators': [5, 10],
            'max_depth': [3, 5]
        }
        
        best_model, best_params, best_score = training_pipeline._tune_hyperparameters(
            'random_forest', X, y, param_grid, cv=3
        )
        
        assert isinstance(best_model, RandomForestClassifier)
        assert isinstance(best_params, dict)
        assert isinstance(best_score, (int, float))
        assert 0 <= best_score <= 1
        
        # Check that best parameters are from the grid
        assert best_params['n_estimators'] in param_grid['n_estimators']
        assert best_params['max_depth'] in param_grid['max_depth']