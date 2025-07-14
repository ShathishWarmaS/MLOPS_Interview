"""
Unit tests for data processing pipeline
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import great_expectations as ge

from training.pipelines.nodes import DataProcessor


@pytest.mark.unit
class TestDataProcessor:
    """Test suite for DataProcessor class."""
    
    def test_init(self, data_processor_config):
        """Test DataProcessor initialization."""
        processor = DataProcessor(data_processor_config)
        assert processor.config == data_processor_config
        assert processor.data_quality_suite is not None
    
    def test_generate_synthetic_data(self, data_processor):
        """Test synthetic data generation."""
        n_samples = 100
        df = data_processor.generate_synthetic_data(n_samples)
        
        assert len(df) == n_samples
        assert 'target' in df.columns
        assert df['target'].dtype == int
        assert df['target'].isin([0, 1]).all()
        
        # Check data types
        assert df.dtypes['feature_1'] == np.float64
        assert df.dtypes['feature_2'] == np.float64
        assert df.dtypes['feature_3'] == object
        assert df.dtypes['feature_4'] == np.int64
    
    def test_generate_synthetic_data_reproducibility(self, data_processor):
        """Test that synthetic data generation is reproducible."""
        df1 = data_processor.generate_synthetic_data(100)
        df2 = data_processor.generate_synthetic_data(100)
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_validate_data_quality_success(self, data_processor, sample_data):
        """Test successful data quality validation."""
        quality_rules = {
            'completeness_check': {
                'type': 'completeness',
                'column': 'feature_1'
            },
            'range_check': {
                'type': 'range',
                'column': 'feature_4',
                'min': 0,
                'max': 10
            }
        }
        
        # Should not raise an exception
        result = data_processor.validate_data_quality(sample_data, quality_rules)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, sample_data)
    
    def test_validate_data_quality_failure_strict(self, data_processor_config, sample_data):
        """Test data quality validation failure in strict mode."""
        # Create processor with strict validation
        config = data_processor_config.copy()
        config['strict_validation'] = True
        processor = DataProcessor(config)
        
        # Create invalid data
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'feature_1'] = None
        
        quality_rules = {
            'completeness_check': {
                'type': 'completeness',
                'column': 'feature_1'
            }
        }
        
        with pytest.raises(ValueError, match="Data quality validation failed"):
            processor.validate_data_quality(invalid_data, quality_rules)
    
    def test_validate_data_quality_failure_non_strict(self, data_processor, sample_data):
        """Test data quality validation failure in non-strict mode."""
        # Create invalid data
        invalid_data = sample_data.copy()
        invalid_data.loc[0, 'feature_1'] = None
        
        quality_rules = {
            'completeness_check': {
                'type': 'completeness',
                'column': 'feature_1'
            }
        }
        
        # Should not raise an exception in non-strict mode
        result = data_processor.validate_data_quality(invalid_data, quality_rules)
        assert isinstance(result, pd.DataFrame)
    
    def test_clean_and_transform_missing_values(self, data_processor, sample_data):
        """Test cleaning and transformation of missing values."""
        # Introduce missing values
        data_with_missing = sample_data.copy()
        data_with_missing.loc[0:5, 'feature_1'] = np.nan
        data_with_missing.loc[10:15, 'feature_2'] = np.nan
        
        transform_config = {
            'missing_values': {
                'feature_1': 'fill_mean',
                'feature_2': 'fill_median'
            }
        }
        
        result = data_processor.clean_and_transform(data_with_missing, transform_config)
        
        # Check that missing values are filled
        assert not result['feature_1'].isna().any()
        assert not result['feature_2'].isna().any()
        
        # Check that values are reasonable
        assert result['feature_1'].iloc[0] == pytest.approx(sample_data['feature_1'].mean(), abs=1e-6)
    
    def test_clean_and_transform_outliers_iqr(self, data_processor, sample_data):
        """Test outlier removal using IQR method."""
        # Add extreme outliers
        data_with_outliers = sample_data.copy()
        data_with_outliers.loc[0, 'feature_1'] = 100  # Extreme outlier
        data_with_outliers.loc[1, 'feature_1'] = -100  # Extreme outlier
        
        transform_config = {
            'outliers': {
                'feature_1': 'iqr'
            }
        }
        
        result = data_processor.clean_and_transform(data_with_outliers, transform_config)
        
        # Check that outliers are removed
        assert len(result) < len(data_with_outliers)
        assert 100 not in result['feature_1'].values
        assert -100 not in result['feature_1'].values
    
    def test_clean_and_transform_outliers_zscore(self, data_processor, sample_data):
        """Test outlier removal using Z-score method."""
        # Add extreme outliers
        data_with_outliers = sample_data.copy()
        data_with_outliers.loc[0, 'feature_1'] = 100
        data_with_outliers.loc[1, 'feature_1'] = -100
        
        transform_config = {
            'outliers': {
                'feature_1': 'zscore'
            }
        }
        
        result = data_processor.clean_and_transform(data_with_outliers, transform_config)
        
        # Check that outliers are removed
        assert len(result) < len(data_with_outliers)
        assert 100 not in result['feature_1'].values
        assert -100 not in result['feature_1'].values
    
    def test_clean_and_transform_dtype_conversion(self, data_processor, sample_data):
        """Test data type conversion."""
        transform_config = {
            'dtypes': {
                'feature_4': 'float64'
            }
        }
        
        result = data_processor.clean_and_transform(sample_data, transform_config)
        
        assert result['feature_4'].dtype == np.float64
    
    def test_feature_engineering_time_features(self, data_processor):
        """Test time-based feature engineering."""
        # Create data with timestamp
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
            'feature_1': np.random.normal(0, 1, 100),
            'target': np.random.binomial(1, 0.3, 100)
        })
        
        feature_config = {
            'time_features': {
                'timestamp_column': 'timestamp',
                'extract_hour': True,
                'extract_day_of_week': True,
                'extract_month': True,
                'extract_quarter': True
            }
        }
        
        result = data_processor.feature_engineering(data, feature_config)
        
        assert 'hour' in result.columns
        assert 'day_of_week' in result.columns
        assert 'month' in result.columns
        assert 'quarter' in result.columns
        
        # Check value ranges
        assert result['hour'].min() >= 0
        assert result['hour'].max() <= 23
        assert result['day_of_week'].min() >= 0
        assert result['day_of_week'].max() <= 6
    
    def test_feature_engineering_categorical_encoding(self, data_processor, sample_data):
        """Test categorical feature encoding."""
        feature_config = {
            'categorical_encoding': {
                'feature_3': 'onehot'
            }
        }
        
        result = data_processor.feature_engineering(sample_data, feature_config)
        
        # Check that one-hot encoded columns are created
        onehot_columns = [col for col in result.columns if col.startswith('feature_3_')]
        assert len(onehot_columns) > 0
        
        # Check that original column is removed
        assert 'feature_3' not in result.columns
    
    def test_feature_engineering_numerical_features(self, data_processor, sample_data):
        """Test numerical feature engineering."""
        feature_config = {
            'numerical_features': {
                'ratio_feature': {
                    'type': 'ratio',
                    'numerator': 'feature_1',
                    'denominator': 'feature_2'
                },
                'polynomial_feature': {
                    'type': 'polynomial',
                    'column': 'feature_1',
                    'degree': 2
                },
                'interaction_feature': {
                    'type': 'interaction',
                    'column1': 'feature_1',
                    'column2': 'feature_4'
                }
            }
        }
        
        result = data_processor.feature_engineering(sample_data, feature_config)
        
        assert 'ratio_feature' in result.columns
        assert 'polynomial_feature' in result.columns
        assert 'interaction_feature' in result.columns
        
        # Verify calculations
        expected_ratio = sample_data['feature_1'] / sample_data['feature_2']
        expected_polynomial = sample_data['feature_1'] ** 2
        expected_interaction = sample_data['feature_1'] * sample_data['feature_4']
        
        pd.testing.assert_series_equal(result['ratio_feature'], expected_ratio, check_names=False)
        pd.testing.assert_series_equal(result['polynomial_feature'], expected_polynomial, check_names=False)
        pd.testing.assert_series_equal(result['interaction_feature'], expected_interaction, check_names=False)
    
    def test_split_data(self, data_processor, sample_data):
        """Test data splitting."""
        split_config = {
            'target_column': 'target',
            'test_size': 0.2,
            'val_size': 0.2,
            'random_state': 42,
            'stratify': True
        }
        
        train_data, val_data, test_data = data_processor.split_data(sample_data, split_config)
        
        # Check that splits add up to original data
        total_samples = len(train_data) + len(val_data) + len(test_data)
        assert total_samples == len(sample_data)
        
        # Check approximate split sizes
        assert len(test_data) == pytest.approx(len(sample_data) * 0.2, abs=2)
        assert len(val_data) == pytest.approx(len(sample_data) * 0.2, abs=2)
        assert len(train_data) == pytest.approx(len(sample_data) * 0.6, abs=2)
        
        # Check that all splits have the target column
        assert 'target' in train_data.columns
        assert 'target' in val_data.columns
        assert 'target' in test_data.columns
    
    def test_split_data_without_stratification(self, data_processor, sample_data):
        """Test data splitting without stratification."""
        split_config = {
            'target_column': 'target',
            'test_size': 0.3,
            'val_size': 0.2,
            'random_state': 42,
            'stratify': False
        }
        
        train_data, val_data, test_data = data_processor.split_data(sample_data, split_config)
        
        # Check that splits add up to original data
        total_samples = len(train_data) + len(val_data) + len(test_data)
        assert total_samples == len(sample_data)
        
        # Check approximate split sizes
        assert len(test_data) == pytest.approx(len(sample_data) * 0.3, abs=2)
        assert len(val_data) == pytest.approx(len(sample_data) * 0.2, abs=2)
        assert len(train_data) == pytest.approx(len(sample_data) * 0.5, abs=2)
    
    @patch('training.pipelines.nodes.DataProcessor._extract_from_database')
    def test_extract_raw_data_database(self, mock_extract_db, data_processor, sample_data):
        """Test raw data extraction from database."""
        mock_extract_db.return_value = sample_data
        
        sources = {
            'database_source': {
                'type': 'database',
                'connection_string': 'postgresql://user:pass@localhost/db'
            }
        }
        date_range = ('2023-01-01', '2023-01-31')
        
        result = data_processor.extract_raw_data(sources, date_range)
        
        mock_extract_db.assert_called_once()
        assert len(result) == len(sample_data)
        assert 'source' in result.columns
        assert (result['source'] == 'database_source').all()
    
    @patch('training.pipelines.nodes.DataProcessor._extract_from_s3')
    @patch('training.pipelines.nodes.DataProcessor._extract_from_api')
    def test_extract_raw_data_multiple_sources(self, mock_extract_api, mock_extract_s3, 
                                               data_processor, sample_data):
        """Test raw data extraction from multiple sources."""
        # Mock data from different sources
        s3_data = sample_data.iloc[:50].copy()
        api_data = sample_data.iloc[50:].copy()
        
        mock_extract_s3.return_value = s3_data
        mock_extract_api.return_value = api_data
        
        sources = {
            's3_source': {'type': 's3', 'bucket': 'data-bucket'},
            'api_source': {'type': 'api', 'endpoint': 'https://api.example.com'}
        }
        date_range = ('2023-01-01', '2023-01-31')
        
        result = data_processor.extract_raw_data(sources, date_range)
        
        mock_extract_s3.assert_called_once()
        mock_extract_api.assert_called_once()
        
        # Check that data from both sources is included
        assert len(result) == len(sample_data)
        assert 's3_source' in result['source'].values
        assert 'api_source' in result['source'].values
    
    def test_extract_raw_data_unsupported_source(self, data_processor):
        """Test extraction with unsupported source type."""
        sources = {
            'unsupported_source': {
                'type': 'unsupported_type'
            }
        }
        date_range = ('2023-01-01', '2023-01-31')
        
        with pytest.raises(ValueError, match="Unsupported source type"):
            data_processor.extract_raw_data(sources, date_range)
    
    def test_remove_outliers_iqr(self, data_processor):
        """Test IQR outlier removal method."""
        # Create data with known outliers
        data = pd.DataFrame({
            'test_column': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10]  # 100 is outlier
        })
        
        result = data_processor._remove_outliers_iqr(data, 'test_column')
        
        # Check that outlier is removed
        assert 100 not in result['test_column'].values
        assert len(result) < len(data)
    
    def test_remove_outliers_zscore(self, data_processor):
        """Test Z-score outlier removal method."""
        # Create data with known outliers
        data = pd.DataFrame({
            'test_column': [1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10]  # 100 is outlier
        })
        
        result = data_processor._remove_outliers_zscore(data, 'test_column', threshold=2.0)
        
        # Check that outlier is removed
        assert 100 not in result['test_column'].values
        assert len(result) < len(data)
    
    def test_get_stats(self, data_processor):
        """Test getting processor statistics."""
        stats = data_processor.get_stats()
        
        assert 'is_fitted' in stats
        assert 'feature_stats' in stats
        assert 'label_encoders' in stats
        assert 'feature_columns' in stats
        
        assert isinstance(stats['is_fitted'], bool)
        assert isinstance(stats['feature_stats'], dict)
        assert isinstance(stats['label_encoders'], list)
        assert isinstance(stats['feature_columns'], list)
    
    def test_save_and_load(self, data_processor, temp_dir):
        """Test saving and loading processor."""
        filepath = temp_dir / "test_processor.pkl"
        
        # Save processor
        data_processor.save(str(filepath))
        assert filepath.exists()
        
        # Load processor
        loaded_processor = DataProcessor.load(str(filepath))
        assert loaded_processor.config == data_processor.config
    
    def test_preprocess_features_training(self, data_processor, sample_data):
        """Test feature preprocessing for training."""
        result = data_processor.preprocess_features(sample_data, is_training=True)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(sample_data)  # May be less due to outlier removal
        assert data_processor.is_fitted
    
    def test_preprocess_features_inference(self, data_processor, sample_data):
        """Test feature preprocessing for inference."""
        # First fit the processor
        _ = data_processor.preprocess_features(sample_data, is_training=True)
        
        # Then test inference preprocessing
        inference_data = sample_data.iloc[:10].copy()
        result = data_processor.preprocess_features(inference_data, is_training=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(inference_data)