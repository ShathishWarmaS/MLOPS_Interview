"""
MLOps Training Pipeline
Complete pipeline orchestration with Kedro-style implementation
"""

from typing import Dict, Any, Tuple
import logging
from pathlib import Path
import yaml
import pandas as pd
from .nodes import (
    extract_raw_data,
    validate_data_quality,
    clean_and_transform,
    feature_engineering,
    split_data,
    train_baseline_model,
    train_advanced_models,
    evaluate_models,
    select_best_model,
    register_model
)

logger = logging.getLogger(__name__)

class Pipeline:
    """Pipeline class for organizing pipeline execution"""
    
    def __init__(self, nodes: list, name: str = None):
        self.nodes = nodes
        self.name = name or "unnamed_pipeline"
    
    def run(self, data_catalog: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the pipeline"""
        logger.info(f"Running pipeline: {self.name}")
        
        results = data_catalog.copy()
        parameters = parameters or {}
        
        for node in self.nodes:
            logger.info(f"Executing node: {node['name']}")
            
            # Get inputs
            inputs = {}
            for input_name in node['inputs']:
                if input_name.startswith('params:'):
                    # Parameter input
                    param_key = input_name.replace('params:', '')
                    inputs[param_key] = parameters.get(param_key)
                else:
                    # Data input
                    inputs[input_name] = results.get(input_name)
            
            # Execute function
            try:
                if len(node['outputs']) == 1:
                    output = node['func'](**inputs)
                    results[node['outputs'][0]] = output
                else:
                    # Multiple outputs
                    outputs = node['func'](**inputs)
                    for i, output_name in enumerate(node['outputs']):
                        results[output_name] = outputs[i]
                
                logger.info(f"Node {node['name']} completed successfully")
                
            except Exception as e:
                logger.error(f"Node {node['name']} failed: {e}")
                raise
        
        logger.info(f"Pipeline {self.name} completed successfully")
        return results

def create_data_pipeline() -> Pipeline:
    """Create data processing pipeline"""
    return Pipeline([
        {
            'name': 'extract_raw_data_node',
            'func': extract_raw_data,
            'inputs': ['data_sources', 'date_range'],
            'outputs': ['raw_data']
        },
        {
            'name': 'validate_data_quality_node',
            'func': validate_data_quality,
            'inputs': ['raw_data', 'quality_rules'],
            'outputs': ['validated_data']
        },
        {
            'name': 'clean_and_transform_node',
            'func': clean_and_transform,
            'inputs': ['validated_data', 'transform_config'],
            'outputs': ['clean_data']
        },
        {
            'name': 'feature_engineering_node',
            'func': feature_engineering,
            'inputs': ['clean_data', 'feature_config'],
            'outputs': ['features']
        },
        {
            'name': 'split_data_node',
            'func': split_data,
            'inputs': ['features', 'split_config'],
            'outputs': ['train_data', 'val_data', 'test_data']
        }
    ], name="data_pipeline")

def create_training_pipeline() -> Pipeline:
    """Create model training pipeline"""
    return Pipeline([
        {
            'name': 'train_baseline_model_node',
            'func': train_baseline_model,
            'inputs': ['train_data', 'baseline_config'],
            'outputs': ['baseline_model']
        },
        {
            'name': 'train_advanced_models_node',
            'func': train_advanced_models,
            'inputs': ['train_data', 'advanced_configs'],
            'outputs': ['advanced_models']
        },
        {
            'name': 'evaluate_models_node',
            'func': evaluate_models,
            'inputs': ['baseline_model', 'advanced_models', 'val_data'],
            'outputs': ['model_evaluations']
        },
        {
            'name': 'select_best_model_node',
            'func': select_best_model,
            'inputs': ['model_evaluations', 'selection_criteria'],
            'outputs': ['best_model']
        },
        {
            'name': 'register_model_node',
            'func': register_model,
            'inputs': ['best_model', 'test_data', 'registry_config'],
            'outputs': ['registered_model']
        }
    ], name="training_pipeline")

def create_full_pipeline() -> Pipeline:
    """Create complete end-to-end pipeline"""
    data_pipeline_nodes = create_data_pipeline().nodes
    training_pipeline_nodes = create_training_pipeline().nodes
    
    return Pipeline(
        data_pipeline_nodes + training_pipeline_nodes,
        name="full_mlops_pipeline"
    )

class PipelineRunner:
    """Pipeline runner with configuration management"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config/pipeline_config.yaml"
        self.config = self._load_config()
        self.setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = self._get_default_config()
            self._save_default_config(config_file, config)
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration"""
        return {
            'data_sources': {
                'synthetic': {
                    'type': 'synthetic',
                    'n_samples': 10000,
                    'n_features': 10,
                    'required': True
                }
            },
            'date_range': {
                'start_date': '2024-01-01',
                'end_date': '2024-12-31'
            },
            'quality_rules': {
                'completeness_check': {
                    'type': 'completeness',
                    'column': 'target',
                    'threshold': 0.99,
                    'critical': True
                },
                'feature_range_check': {
                    'type': 'range',
                    'column': 'feature_0',
                    'min': -5.0,
                    'max': 5.0,
                    'threshold': 0.95,
                    'critical': False
                }
            },
            'transform_config': {
                'remove_duplicates': True,
                'missing_values': {
                    'feature_0': 'fill_mean',
                    'feature_1': 'fill_median'
                },
                'outliers': {
                    'feature_0': {
                        'method': 'iqr',
                        'threshold': 1.5
                    }
                },
                'dtypes': {
                    'target': 'int64'
                }
            },
            'feature_config': {
                'numerical_features': {
                    'feature_0_squared': {
                        'type': 'polynomial',
                        'column': 'feature_0',
                        'degree': 2
                    },
                    'feature_interaction': {
                        'type': 'interaction',
                        'column1': 'feature_0',
                        'column2': 'feature_1'
                    }
                },
                'statistical_features': {
                    'feature_mean': {
                        'type': 'mean',
                        'columns': ['feature_0', 'feature_1', 'feature_2']
                    }
                }
            },
            'split_config': {
                'target_column': 'target',
                'test_size': 0.2,
                'val_size': 0.2,
                'stratify': True,
                'random_state': 42
            },
            'baseline_config': {
                'target_column': 'target',
                'model_type': 'logistic_regression'
            },
            'advanced_configs': {
                'target_column': 'target',
                'models': {
                    'random_forest': {
                        'n_estimators': 100,
                        'max_depth': 10
                    },
                    'gradient_boosting': {
                        'n_estimators': 100,
                        'learning_rate': 0.1,
                        'max_depth': 6
                    },
                    'logistic_regression': {
                        'C': 1.0
                    }
                }
            },
            'selection_criteria': {
                'primary_metric': 'f1_score',
                'secondary_metrics': ['accuracy', 'roc_auc'],
                'minimize': False
            },
            'registry_config': {
                'target_column': 'target',
                'model_name': 'mlops_pipeline_model',
                'stage': 'Staging'
            }
        }
    
    def _save_default_config(self, config_file: Path, config: Dict[str, Any]):
        """Save default configuration to file"""
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"Default configuration saved to {config_file}")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def run_data_pipeline(self) -> Dict[str, Any]:
        """Run data processing pipeline"""
        logger.info("Starting data pipeline")
        
        pipeline = create_data_pipeline()
        
        # Prepare data catalog
        data_catalog = {}
        
        # Prepare parameters
        parameters = {
            'data_sources': self.config['data_sources'],
            'date_range': self.config['date_range'],
            'quality_rules': self.config['quality_rules'],
            'transform_config': self.config['transform_config'],
            'feature_config': self.config['feature_config'],
            'split_config': self.config['split_config']
        }
        
        results = pipeline.run(data_catalog, parameters)
        
        logger.info("Data pipeline completed successfully")
        return results
    
    def run_training_pipeline(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run training pipeline"""
        logger.info("Starting training pipeline")
        
        pipeline = create_training_pipeline()
        
        # Prepare data catalog with results from data pipeline
        data_catalog = data_results.copy()
        
        # Prepare parameters
        parameters = {
            'baseline_config': self.config['baseline_config'],
            'advanced_configs': self.config['advanced_configs'],
            'selection_criteria': self.config['selection_criteria'],
            'registry_config': self.config['registry_config']
        }
        
        results = pipeline.run(data_catalog, parameters)
        
        logger.info("Training pipeline completed successfully")
        return results
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run complete end-to-end pipeline"""
        logger.info("Starting full MLOps pipeline")
        
        pipeline = create_full_pipeline()
        
        # Prepare data catalog
        data_catalog = {}
        
        # Prepare all parameters
        parameters = {
            # Data pipeline parameters
            'data_sources': self.config['data_sources'],
            'date_range': self.config['date_range'],
            'quality_rules': self.config['quality_rules'],
            'transform_config': self.config['transform_config'],
            'feature_config': self.config['feature_config'],
            'split_config': self.config['split_config'],
            
            # Training pipeline parameters
            'baseline_config': self.config['baseline_config'],
            'advanced_configs': self.config['advanced_configs'],
            'selection_criteria': self.config['selection_criteria'],
            'registry_config': self.config['registry_config']
        }
        
        results = pipeline.run(data_catalog, parameters)
        
        logger.info("Full MLOps pipeline completed successfully")
        return results
    
    def run_pipeline_with_custom_config(self, custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run pipeline with custom configuration"""
        logger.info("Running pipeline with custom configuration")
        
        # Merge custom config with default config
        merged_config = self.config.copy()
        merged_config.update(custom_config)
        
        # Temporarily update config
        original_config = self.config
        self.config = merged_config
        
        try:
            results = self.run_full_pipeline()
        finally:
            # Restore original config
            self.config = original_config
        
        return results
    
    def validate_config(self) -> bool:
        """Validate pipeline configuration"""
        logger.info("Validating pipeline configuration")
        
        required_sections = [
            'data_sources', 'quality_rules', 'transform_config',
            'feature_config', 'split_config', 'baseline_config',
            'advanced_configs', 'selection_criteria', 'registry_config'
        ]
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate data sources
        for source_name, source_config in self.config['data_sources'].items():
            if 'type' not in source_config:
                logger.error(f"Data source {source_name} missing 'type' field")
                return False
        
        # Validate target column consistency
        target_cols = [
            self.config['split_config'].get('target_column'),
            self.config['baseline_config'].get('target_column'),
            self.config['advanced_configs'].get('target_column'),
            self.config['registry_config'].get('target_column')
        ]
        
        if len(set(target_cols)) > 1:
            logger.error("Inconsistent target column definitions across configurations")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get pipeline configuration summary"""
        return {
            'data_sources': list(self.config['data_sources'].keys()),
            'target_column': self.config['split_config']['target_column'],
            'models_to_train': list(self.config['advanced_configs']['models'].keys()),
            'selection_metric': self.config['selection_criteria']['primary_metric'],
            'model_registry_name': self.config['registry_config']['model_name'],
            'quality_rules_count': len(self.config['quality_rules']),
            'feature_engineering_enabled': bool(self.config['feature_config'])
        }

def main():
    """Main function for running the pipeline"""
    import argparse
    import mlflow
    
    parser = argparse.ArgumentParser(description='Run MLOps Training Pipeline')
    parser.add_argument('--config', type=str, help='Path to pipeline configuration file')
    parser.add_argument('--pipeline', type=str, choices=['data', 'training', 'full'], 
                       default='full', help='Pipeline to run')
    parser.add_argument('--experiment-name', type=str, default='mlops_pipeline',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Setup MLflow
    mlflow.set_experiment(args.experiment_name)
    
    # Initialize pipeline runner
    runner = PipelineRunner(args.config)
    
    # Validate configuration
    if not runner.validate_config():
        logger.error("Configuration validation failed")
        return 1
    
    # Print pipeline summary
    summary = runner.get_pipeline_summary()
    logger.info("Pipeline Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")
    
    try:
        if args.pipeline == 'data':
            results = runner.run_data_pipeline()
            logger.info("Data pipeline completed successfully")
        elif args.pipeline == 'training':
            # For training-only, we need to run data pipeline first
            data_results = runner.run_data_pipeline()
            results = runner.run_training_pipeline(data_results)
            logger.info("Training pipeline completed successfully")
        else:  # full
            results = runner.run_full_pipeline()
            logger.info("Full pipeline completed successfully")
        
        # Print final results
        if 'registered_model' in results:
            model_info = results['registered_model']
            logger.info(f"Model registered: {model_info['model_name']}")
            logger.info(f"Test metrics: {model_info['test_metrics']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())