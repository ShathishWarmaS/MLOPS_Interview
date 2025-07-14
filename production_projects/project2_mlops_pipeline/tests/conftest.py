"""
Pytest configuration and shared fixtures for MLOps pipeline tests
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import os
import sys
from typing import Generator, Dict, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.pipelines.nodes import DataProcessor
from training.pipelines.training_pipeline import TrainingPipeline
from serving.api.main import app
from monitoring.model_monitor import ModelMonitor
from monitoring.drift_detector import DriftDetector


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_data():
    """Generate sample dataset for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.exponential(2, n_samples),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature_4': np.random.randint(0, 10, n_samples),
        'target': np.random.binomial(1, 0.3, n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def large_sample_data():
    """Generate larger sample dataset for performance testing."""
    np.random.seed(42)
    n_samples = 50000
    
    data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.exponential(2, n_samples),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature_4': np.random.randint(0, 10, n_samples),
        'target': np.random.binomial(1, 0.3, n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def data_processor_config():
    """Configuration for data processor."""
    return {
        'strict_validation': False,
        'missing_values': {
            'feature_1': 'fill_mean',
            'feature_2': 'fill_median'
        },
        'outliers': {
            'feature_1': 'iqr',
            'feature_2': 'zscore'
        },
        'dtypes': {
            'feature_4': 'int64'
        }
    }


@pytest.fixture
def data_processor(data_processor_config):
    """Create a data processor instance."""
    return DataProcessor(data_processor_config)


@pytest.fixture
def training_config():
    """Configuration for training pipeline."""
    return {
        'data_config': {
            'n_samples': 1000,
            'test_size': 0.2,
            'val_size': 0.2,
            'random_state': 42
        },
        'model_config': {
            'type': 'random_forest',
            'n_estimators': 10,  # Small for faster testing
            'random_state': 42
        },
        'feature_config': {
            'numerical_features': ['feature_1', 'feature_2', 'feature_4'],
            'categorical_features': ['feature_3']
        }
    }


@pytest.fixture
def training_pipeline():
    """Create a training pipeline instance."""
    return TrainingPipeline()


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    with pytest.mock.patch('mlflow.start_run') as mock_start_run, \
         pytest.mock.patch('mlflow.log_metric') as mock_log_metric, \
         pytest.mock.patch('mlflow.log_param') as mock_log_param, \
         pytest.mock.patch('mlflow.sklearn.log_model') as mock_log_model:
        
        mock_start_run.return_value.__enter__ = Mock()
        mock_start_run.return_value.__exit__ = Mock()
        
        yield {
            'start_run': mock_start_run,
            'log_metric': mock_log_metric,
            'log_param': mock_log_param,
            'log_model': mock_log_model
        }


@pytest.fixture
def model_monitor_config():
    """Configuration for model monitor."""
    return {
        'database_url': 'sqlite:///test_monitoring.db',
        'batch_size': 100,
        'drift_threshold': 0.05,
        'alert_threshold': 0.1
    }


@pytest.fixture
async def model_monitor(model_monitor_config):
    """Create a model monitor instance."""
    monitor = ModelMonitor(model_monitor_config)
    await monitor.initialize()
    yield monitor
    await monitor.close()


@pytest.fixture
def drift_detector():
    """Create a drift detector instance."""
    return DriftDetector()


@pytest.fixture
def api_client():
    """Create a test client for the API."""
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def mock_database():
    """Mock database connection."""
    with pytest.mock.patch('sqlalchemy.create_engine') as mock_engine:
        mock_connection = Mock()
        mock_engine.return_value.connect.return_value = mock_connection
        yield mock_connection


@pytest.fixture
def mock_redis():
    """Mock Redis connection."""
    with pytest.mock.patch('redis.Redis') as mock_redis:
        mock_client = Mock()
        mock_redis.return_value = mock_client
        yield mock_client


@pytest.fixture
def environment_variables():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    test_env = {
        'MLFLOW_TRACKING_URI': 'file:///tmp/mlruns',
        'DATABASE_URL': 'sqlite:///test.db',
        'REDIS_URL': 'redis://localhost:6379/1',
        'ENVIRONMENT': 'test',
        'DEBUG': 'True'
    }
    
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def prediction_request_data():
    """Sample prediction request data."""
    return {
        'feature_1': 1.5,
        'feature_2': 2.3,
        'feature_3': 'A',
        'feature_4': 5
    }


@pytest.fixture
def batch_prediction_request_data():
    """Sample batch prediction request data."""
    return {
        'predictions': [
            {
                'feature_1': 1.5,
                'feature_2': 2.3,
                'feature_3': 'A',
                'feature_4': 5
            },
            {
                'feature_1': -0.5,
                'feature_2': 1.8,
                'feature_3': 'B',
                'feature_4': 3
            },
            {
                'feature_1': 2.1,
                'feature_2': 3.2,
                'feature_3': 'C',
                'feature_4': 7
            }
        ]
    }


@pytest.fixture
def model_artifacts(temp_dir):
    """Create mock model artifacts."""
    artifacts_dir = temp_dir / "model_artifacts"
    artifacts_dir.mkdir()
    
    # Create dummy model file
    model_file = artifacts_dir / "model.pkl"
    model_file.write_text("dummy_model_content")
    
    # Create dummy metadata
    metadata_file = artifacts_dir / "metadata.json"
    metadata = {
        'model_type': 'RandomForestClassifier',
        'version': '1.0.0',
        'training_timestamp': '2023-01-01T00:00:00',
        'metrics': {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87,
            'f1_score': 0.85
        }
    }
    metadata_file.write_text(str(metadata))
    
    return artifacts_dir


@pytest.fixture
def performance_benchmark():
    """Performance benchmarks for testing."""
    return {
        'max_prediction_latency': 0.5,  # 500ms
        'max_training_time': 60,  # 60 seconds
        'max_batch_size': 1000,
        'min_throughput': 100  # requests per second
    }


class AsyncContextManager:
    """Helper class for async context managers in tests."""
    
    def __init__(self, async_func):
        self.async_func = async_func
        
    async def __aenter__(self):
        return await self.async_func()
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def async_mock():
    """Create async mock helper."""
    return AsyncMock


# Test data generators
def generate_drift_data(base_data: pd.DataFrame, drift_amount: float = 0.5) -> pd.DataFrame:
    """Generate data with concept drift."""
    drifted_data = base_data.copy()
    
    # Add drift to numerical features
    for col in ['feature_1', 'feature_2']:
        if col in drifted_data.columns:
            drifted_data[col] = drifted_data[col] + np.random.normal(drift_amount, 0.1, len(drifted_data))
    
    # Add drift to categorical features
    if 'feature_3' in drifted_data.columns:
        # Change distribution of categorical feature
        mask = np.random.random(len(drifted_data)) < drift_amount
        drifted_data.loc[mask, 'feature_3'] = np.random.choice(['A', 'B', 'C'], sum(mask))
    
    return drifted_data


def generate_anomaly_data(base_data: pd.DataFrame, anomaly_rate: float = 0.1) -> pd.DataFrame:
    """Generate data with anomalies."""
    anomaly_data = base_data.copy()
    n_anomalies = int(len(anomaly_data) * anomaly_rate)
    anomaly_indices = np.random.choice(len(anomaly_data), n_anomalies, replace=False)
    
    # Inject anomalies
    for idx in anomaly_indices:
        for col in ['feature_1', 'feature_2']:
            if col in anomaly_data.columns:
                # Make values extreme outliers
                anomaly_data.loc[idx, col] = np.random.choice([-10, 10]) * np.abs(anomaly_data[col].std())
    
    return anomaly_data


# Pytest markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.e2e = pytest.mark.e2e
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow


# Test utilities
class TestUtils:
    """Utility functions for tests."""
    
    @staticmethod
    def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = True):
        """Assert two DataFrames are equal with better error messages."""
        try:
            pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
        except AssertionError as e:
            pytest.fail(f"DataFrames are not equal: {e}")
    
    @staticmethod
    def assert_dict_contains(actual: Dict[str, Any], expected: Dict[str, Any]):
        """Assert that actual dict contains all key-value pairs from expected dict."""
        for key, value in expected.items():
            assert key in actual, f"Missing key: {key}"
            assert actual[key] == value, f"Value mismatch for key {key}: {actual[key]} != {value}"
    
    @staticmethod
    def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
        """Wait for a condition to become true."""
        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        return False


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils