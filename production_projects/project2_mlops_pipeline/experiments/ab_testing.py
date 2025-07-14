"""
A/B Testing Framework for ML Models
Production-ready experimentation platform for model comparison
"""

import asyncio
import logging
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import threading
from enum import Enum
import hashlib
import uuid
from abc import ABC, abstractmethod

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Statistical libraries
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Local imports
from serving.api.predictor import ModelPredictor
from monitoring.model_monitor import ModelMonitor

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    """Experiment status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"

class TrafficSplitStrategy(Enum):
    """Traffic splitting strategies"""
    RANDOM = "random"
    HASH_BASED = "hash_based"
    GEOGRAPHIC = "geographic"
    FEATURE_BASED = "feature_based"

@dataclass
class ExperimentConfig:
    """A/B test experiment configuration"""
    experiment_id: str
    name: str
    description: str
    model_a_version: str  # Control model
    model_b_version: str  # Treatment model
    traffic_split: float  # Percentage for model B (0.0 to 1.0)
    split_strategy: TrafficSplitStrategy
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    success_metrics: List[str] = None
    segment_filters: Dict[str, Any] = None
    ramp_strategy: Optional[Dict[str, Any]] = None

@dataclass
class ExperimentResult:
    """A/B test result"""
    experiment_id: str
    timestamp: float
    user_id: str
    model_version: str
    prediction: float
    confidence: float
    actual_outcome: Optional[float] = None
    metadata: Dict[str, Any] = None
    segment: Optional[str] = None
    processing_time_ms: float = 0.0

@dataclass
class StatisticalTestResult:
    """Statistical test result"""
    metric_name: str
    control_mean: float
    treatment_mean: float
    control_std: float
    treatment_std: float
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    test_statistic: float
    test_type: str
    sample_size_control: int
    sample_size_treatment: int

@dataclass
class ExperimentSummary:
    """Experiment summary with statistical results"""
    experiment_id: str
    experiment_name: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_hours: float
    total_samples: int
    control_samples: int
    treatment_samples: int
    traffic_split_actual: float
    conversion_rate_control: float
    conversion_rate_treatment: float
    statistical_tests: List[StatisticalTestResult]
    recommendation: str
    confidence_score: float

class TrafficSplitter:
    """Traffic splitting logic for A/B tests"""
    
    def __init__(self, strategy: TrafficSplitStrategy):
        self.strategy = strategy
    
    def assign_variant(self, 
                      user_id: str, 
                      experiment_config: ExperimentConfig,
                      request_metadata: Dict[str, Any] = None) -> str:
        """Assign user to control or treatment group"""
        
        if self.strategy == TrafficSplitStrategy.RANDOM:
            return self._random_assignment(experiment_config.traffic_split)
        
        elif self.strategy == TrafficSplitStrategy.HASH_BASED:
            return self._hash_based_assignment(
                user_id, experiment_config.experiment_id, experiment_config.traffic_split
            )
        
        elif self.strategy == TrafficSplitStrategy.GEOGRAPHIC:
            return self._geographic_assignment(
                request_metadata, experiment_config.traffic_split
            )
        
        elif self.strategy == TrafficSplitStrategy.FEATURE_BASED:
            return self._feature_based_assignment(
                request_metadata, experiment_config.traffic_split
            )
        
        else:
            return self._random_assignment(experiment_config.traffic_split)
    
    def _random_assignment(self, traffic_split: float) -> str:
        """Random assignment to control/treatment"""
        return "treatment" if np.random.random() < traffic_split else "control"
    
    def _hash_based_assignment(self, user_id: str, experiment_id: str, traffic_split: float) -> str:
        """Consistent hash-based assignment"""
        # Create deterministic hash
        combined_key = f"{user_id}:{experiment_id}"
        hash_value = int(hashlib.md5(combined_key.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0
        
        return "treatment" if normalized_hash < traffic_split else "control"
    
    def _geographic_assignment(self, metadata: Dict[str, Any], traffic_split: float) -> str:
        """Geographic-based assignment"""
        if not metadata or 'country' not in metadata:
            return self._random_assignment(traffic_split)
        
        country = metadata['country'].lower()
        
        # Example: Higher treatment allocation for specific countries
        high_traffic_countries = ['us', 'ca', 'gb', 'de', 'fr']
        if country in high_traffic_countries:
            adjusted_split = min(traffic_split * 1.5, 1.0)
        else:
            adjusted_split = traffic_split * 0.5
        
        return self._random_assignment(adjusted_split)
    
    def _feature_based_assignment(self, metadata: Dict[str, Any], traffic_split: float) -> str:
        """Feature-based assignment"""
        if not metadata:
            return self._random_assignment(traffic_split)
        
        # Example: Assignment based on user features
        user_tier = metadata.get('user_tier', 'basic')
        
        if user_tier == 'premium':
            adjusted_split = min(traffic_split * 1.2, 1.0)
        elif user_tier == 'enterprise':
            adjusted_split = min(traffic_split * 1.5, 1.0)
        else:
            adjusted_split = traffic_split
        
        return self._random_assignment(adjusted_split)

class ABTestFramework:
    """A/B testing framework for ML models"""
    
    def __init__(self, 
                 storage_backend: str = "local",
                 database_url: Optional[str] = None):
        
        self.storage_backend = storage_backend
        self.database_url = database_url
        
        # Components
        self.predictors: Dict[str, ModelPredictor] = {}
        self.monitor: Optional[ModelMonitor] = None
        self.traffic_splitter = TrafficSplitter(TrafficSplitStrategy.HASH_BASED)
        
        # Experiment storage
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: List[ExperimentResult] = []
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        
        # State
        self.is_running = False
        
        # Storage paths
        self.storage_dir = Path("experiments/ab_tests")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize A/B testing framework"""
        try:
            logger.info("Initializing A/B testing framework...")
            
            # Initialize monitor
            self.monitor = ModelMonitor()
            await self.monitor.initialize()
            
            # Load existing experiments
            await self._load_experiments()
            
            logger.info("✅ A/B testing framework initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize A/B testing framework: {e}")
            raise
    
    async def _load_experiments(self):
        """Load experiments from storage"""
        try:
            experiments_file = self.storage_dir / "experiments.json"
            
            if experiments_file.exists():
                with open(experiments_file, 'r') as f:
                    experiments_data = json.load(f)
                
                for exp_data in experiments_data:
                    config = ExperimentConfig(**exp_data)
                    self.experiments[config.experiment_id] = config
                
                logger.info(f"Loaded {len(self.experiments)} experiments")
        
        except Exception as e:
            logger.warning(f"Failed to load experiments: {e}")
    
    async def _save_experiments(self):
        """Save experiments to storage"""
        try:
            experiments_file = self.storage_dir / "experiments.json"
            
            experiments_data = []
            for config in self.experiments.values():
                exp_dict = asdict(config)
                # Convert datetime objects to ISO strings
                if exp_dict['start_time']:
                    exp_dict['start_time'] = config.start_time.isoformat()
                if exp_dict['end_time']:
                    exp_dict['end_time'] = config.end_time.isoformat()
                experiments_data.append(exp_dict)
            
            with open(experiments_file, 'w') as f:
                json.dump(experiments_data, f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to save experiments: {e}")
    
    async def create_experiment(self, config: ExperimentConfig) -> str:
        """Create new A/B test experiment"""
        try:
            # Validate configuration
            await self._validate_experiment_config(config)
            
            # Generate experiment ID if not provided
            if not config.experiment_id:
                config.experiment_id = str(uuid.uuid4())
            
            # Set default values
            if not config.success_metrics:
                config.success_metrics = ["accuracy", "f1_score"]
            
            # Store experiment
            with self.lock:
                self.experiments[config.experiment_id] = config
            
            # Save to storage
            await self._save_experiments()
            
            # Load model predictors
            await self._load_model_predictors(config)
            
            logger.info(f"Created experiment: {config.experiment_id} - {config.name}")
            
            return config.experiment_id
        
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise
    
    async def _validate_experiment_config(self, config: ExperimentConfig):
        """Validate experiment configuration"""
        if not config.name:
            raise ValueError("Experiment name is required")
        
        if not config.model_a_version or not config.model_b_version:
            raise ValueError("Both model versions are required")
        
        if not 0.0 <= config.traffic_split <= 1.0:
            raise ValueError("Traffic split must be between 0.0 and 1.0")
        
        if config.confidence_level <= 0.0 or config.confidence_level >= 1.0:
            raise ValueError("Confidence level must be between 0.0 and 1.0")
    
    async def _load_model_predictors(self, config: ExperimentConfig):
        """Load model predictors for experiment"""
        try:
            # Load control model (A)
            if config.model_a_version not in self.predictors:
                predictor_a = ModelPredictor()
                await predictor_a.load_model(model_version=config.model_a_version)
                self.predictors[config.model_a_version] = predictor_a
            
            # Load treatment model (B)
            if config.model_b_version not in self.predictors:
                predictor_b = ModelPredictor()
                await predictor_b.load_model(model_version=config.model_b_version)
                self.predictors[config.model_b_version] = predictor_b
            
            logger.info(f"Loaded predictors for experiment {config.experiment_id}")
        
        except Exception as e:
            logger.error(f"Failed to load model predictors: {e}")
            raise
    
    async def start_experiment(self, experiment_id: str):
        """Start an A/B test experiment"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            config = self.experiments[experiment_id]
            config.start_time = datetime.now()
            
            # Update traffic splitter strategy
            self.traffic_splitter = TrafficSplitter(config.split_strategy)
            
            # Save updated config
            await self._save_experiments()
            
            logger.info(f"Started experiment: {experiment_id}")
        
        except Exception as e:
            logger.error(f"Failed to start experiment: {e}")
            raise
    
    async def stop_experiment(self, experiment_id: str):
        """Stop an A/B test experiment"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            config = self.experiments[experiment_id]
            config.end_time = datetime.now()
            
            # Save updated config
            await self._save_experiments()
            
            logger.info(f"Stopped experiment: {experiment_id}")
        
        except Exception as e:
            logger.error(f"Failed to stop experiment: {e}")
            raise
    
    async def get_prediction(self, 
                           experiment_id: str,
                           user_id: str,
                           features: Dict[str, Any],
                           metadata: Dict[str, Any] = None) -> ExperimentResult:
        """Get prediction from A/B test"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            config = self.experiments[experiment_id]
            
            # Check if experiment is active
            if not config.start_time or (config.end_time and datetime.now() > config.end_time):
                raise ValueError(f"Experiment {experiment_id} is not active")
            
            # Assign variant
            variant = self.traffic_splitter.assign_variant(user_id, config, metadata)
            
            # Select model version
            model_version = config.model_b_version if variant == "treatment" else config.model_a_version
            
            # Get prediction
            start_time = time.time()
            predictor = self.predictors[model_version]
            prediction_result = await predictor.predict(features)
            processing_time = (time.time() - start_time) * 1000
            
            # Create experiment result
            result = ExperimentResult(
                experiment_id=experiment_id,
                timestamp=time.time(),
                user_id=user_id,
                model_version=model_version,
                prediction=prediction_result.prediction,
                confidence=prediction_result.confidence,
                metadata=metadata or {},
                processing_time_ms=processing_time
            )
            
            # Store result
            with self.lock:
                self.results.append(result)
            
            # Log to monitor
            await self.monitor.log_prediction(
                request={"features": features, "user_id": user_id},
                response=result,
                latency_ms=processing_time
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to get prediction: {e}")
            raise
    
    async def add_outcome(self, 
                         experiment_id: str, 
                         user_id: str, 
                         actual_outcome: float):
        """Add actual outcome for result evaluation"""
        try:
            # Find the corresponding result
            with self.lock:
                for result in self.results:
                    if (result.experiment_id == experiment_id and 
                        result.user_id == user_id and 
                        result.actual_outcome is None):
                        result.actual_outcome = actual_outcome
                        break
            
            logger.debug(f"Added outcome for user {user_id} in experiment {experiment_id}")
        
        except Exception as e:
            logger.error(f"Failed to add outcome: {e}")
    
    async def analyze_experiment(self, experiment_id: str) -> ExperimentSummary:
        """Analyze A/B test results"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            config = self.experiments[experiment_id]
            
            # Filter results for this experiment
            experiment_results = [r for r in self.results if r.experiment_id == experiment_id]
            
            if not experiment_results:
                raise ValueError(f"No results found for experiment {experiment_id}")
            
            # Separate control and treatment groups
            control_results = [r for r in experiment_results if r.model_version == config.model_a_version]
            treatment_results = [r for r in experiment_results if r.model_version == config.model_b_version]
            
            # Calculate basic statistics
            total_samples = len(experiment_results)
            control_samples = len(control_results)
            treatment_samples = len(treatment_results)
            
            if control_samples == 0 or treatment_samples == 0:
                raise ValueError("Insufficient data for both groups")
            
            # Calculate actual traffic split
            traffic_split_actual = treatment_samples / total_samples
            
            # Calculate duration
            start_time = config.start_time
            end_time = config.end_time or datetime.now()
            duration_hours = (end_time - start_time).total_seconds() / 3600
            
            # Run statistical tests
            statistical_tests = await self._run_statistical_tests(
                control_results, treatment_results, config.success_metrics, config.confidence_level
            )
            
            # Calculate conversion rates (assuming binary outcomes)
            control_outcomes = [r.actual_outcome for r in control_results if r.actual_outcome is not None]
            treatment_outcomes = [r.actual_outcome for r in treatment_results if r.actual_outcome is not None]
            
            conversion_rate_control = np.mean(control_outcomes) if control_outcomes else 0.0
            conversion_rate_treatment = np.mean(treatment_outcomes) if treatment_outcomes else 0.0
            
            # Generate recommendation
            recommendation, confidence_score = self._generate_recommendation(statistical_tests)
            
            # Determine status
            if config.end_time:
                status = ExperimentStatus.COMPLETED
            elif config.start_time:
                status = ExperimentStatus.ACTIVE
            else:
                status = ExperimentStatus.DRAFT
            
            summary = ExperimentSummary(
                experiment_id=experiment_id,
                experiment_name=config.name,
                status=status,
                start_time=start_time,
                end_time=config.end_time,
                duration_hours=duration_hours,
                total_samples=total_samples,
                control_samples=control_samples,
                treatment_samples=treatment_samples,
                traffic_split_actual=traffic_split_actual,
                conversion_rate_control=conversion_rate_control,
                conversion_rate_treatment=conversion_rate_treatment,
                statistical_tests=statistical_tests,
                recommendation=recommendation,
                confidence_score=confidence_score
            )
            
            return summary
        
        except Exception as e:
            logger.error(f"Failed to analyze experiment: {e}")
            raise
    
    async def _run_statistical_tests(self, 
                                   control_results: List[ExperimentResult],
                                   treatment_results: List[ExperimentResult],
                                   metrics: List[str],
                                   confidence_level: float) -> List[StatisticalTestResult]:
        """Run statistical tests for experiment analysis"""
        tests = []
        
        for metric in metrics:
            try:
                # Extract metric values
                control_values = self._extract_metric_values(control_results, metric)
                treatment_values = self._extract_metric_values(treatment_results, metric)
                
                if not control_values or not treatment_values:
                    continue
                
                # Run appropriate statistical test
                if metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    # Binary/classification metrics
                    test_result = self._run_proportion_test(
                        control_values, treatment_values, metric, confidence_level
                    )
                else:
                    # Continuous metrics
                    test_result = self._run_ttest(
                        control_values, treatment_values, metric, confidence_level
                    )
                
                tests.append(test_result)
            
            except Exception as e:
                logger.warning(f"Failed to run test for metric {metric}: {e}")
        
        return tests
    
    def _extract_metric_values(self, results: List[ExperimentResult], metric: str) -> List[float]:
        """Extract metric values from results"""
        values = []
        
        for result in results:
            if result.actual_outcome is not None:
                if metric == 'accuracy':
                    # Binary accuracy: prediction matches outcome
                    values.append(1.0 if round(result.prediction) == result.actual_outcome else 0.0)
                elif metric == 'conversion_rate':
                    # Use actual outcome directly
                    values.append(result.actual_outcome)
                elif metric == 'confidence':
                    # Use prediction confidence
                    values.append(result.confidence)
                elif metric == 'processing_time':
                    # Use processing time
                    values.append(result.processing_time_ms)
        
        return values
    
    def _run_proportion_test(self, 
                           control_values: List[float],
                           treatment_values: List[float],
                           metric_name: str,
                           confidence_level: float) -> StatisticalTestResult:
        """Run proportion test for binary metrics"""
        
        control_successes = sum(control_values)
        treatment_successes = sum(treatment_values)
        
        control_n = len(control_values)
        treatment_n = len(treatment_values)
        
        control_rate = control_successes / control_n
        treatment_rate = treatment_successes / treatment_n
        
        # Chi-square test
        observed = np.array([[control_successes, control_n - control_successes],
                            [treatment_successes, treatment_n - treatment_successes]])
        
        chi2, p_value, dof, expected = chi2_contingency(observed)
        
        # Effect size (difference in proportions)
        effect_size = treatment_rate - control_rate
        
        # Confidence interval for difference in proportions
        se = np.sqrt(control_rate * (1 - control_rate) / control_n + 
                    treatment_rate * (1 - treatment_rate) / treatment_n)
        
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        margin_error = z_score * se
        
        ci_lower = effect_size - margin_error
        ci_upper = effect_size + margin_error
        
        return StatisticalTestResult(
            metric_name=metric_name,
            control_mean=control_rate,
            treatment_mean=treatment_rate,
            control_std=np.sqrt(control_rate * (1 - control_rate)),
            treatment_std=np.sqrt(treatment_rate * (1 - treatment_rate)),
            effect_size=effect_size,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < (1 - confidence_level),
            test_statistic=chi2,
            test_type="chi_square",
            sample_size_control=control_n,
            sample_size_treatment=treatment_n
        )
    
    def _run_ttest(self, 
                  control_values: List[float],
                  treatment_values: List[float],
                  metric_name: str,
                  confidence_level: float) -> StatisticalTestResult:
        """Run t-test for continuous metrics"""
        
        control_array = np.array(control_values)
        treatment_array = np.array(treatment_values)
        
        # Welch's t-test (unequal variances)
        t_stat, p_value = ttest_ind(treatment_array, control_array, equal_var=False)
        
        control_mean = np.mean(control_array)
        treatment_mean = np.mean(treatment_array)
        control_std = np.std(control_array, ddof=1)
        treatment_std = np.std(treatment_array, ddof=1)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_array) - 1) * control_std**2 + 
                             (len(treatment_array) - 1) * treatment_std**2) / 
                            (len(control_array) + len(treatment_array) - 2))
        effect_size = (treatment_mean - control_mean) / pooled_std
        
        # Confidence interval for difference in means
        se = np.sqrt(control_std**2 / len(control_array) + treatment_std**2 / len(treatment_array))
        df = len(control_array) + len(treatment_array) - 2
        t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, df)
        margin_error = t_critical * se
        
        mean_diff = treatment_mean - control_mean
        ci_lower = mean_diff - margin_error
        ci_upper = mean_diff + margin_error
        
        return StatisticalTestResult(
            metric_name=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            control_std=control_std,
            treatment_std=treatment_std,
            effect_size=effect_size,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < (1 - confidence_level),
            test_statistic=t_stat,
            test_type="welch_ttest",
            sample_size_control=len(control_array),
            sample_size_treatment=len(treatment_array)
        )
    
    def _generate_recommendation(self, tests: List[StatisticalTestResult]) -> Tuple[str, float]:
        """Generate recommendation based on statistical tests"""
        
        if not tests:
            return "Insufficient data for recommendation", 0.0
        
        significant_tests = [t for t in tests if t.is_significant]
        
        if not significant_tests:
            return "No statistically significant differences detected. Continue with control model.", 0.5
        
        # Analyze effect sizes and directions
        positive_effects = [t for t in significant_tests if t.effect_size > 0]
        negative_effects = [t for t in significant_tests if t.effect_size < 0]
        
        if len(positive_effects) > len(negative_effects):
            avg_effect = np.mean([t.effect_size for t in positive_effects])
            confidence = min(0.95, 0.5 + len(positive_effects) * 0.1)
            
            if avg_effect > 0.2:  # Large effect
                return "Strong evidence to deploy treatment model. Significant improvement detected.", confidence
            else:
                return "Moderate evidence to deploy treatment model. Small but significant improvement.", confidence
        
        elif len(negative_effects) > len(positive_effects):
            confidence = min(0.95, 0.5 + len(negative_effects) * 0.1)
            return "Evidence suggests control model performs better. Do not deploy treatment model.", confidence
        
        else:
            return "Mixed results detected. Further investigation recommended.", 0.3
    
    async def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current experiment status"""
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            config = self.experiments[experiment_id]
            experiment_results = [r for r in self.results if r.experiment_id == experiment_id]
            
            # Basic statistics
            total_samples = len(experiment_results)
            control_samples = len([r for r in experiment_results if r.model_version == config.model_a_version])
            treatment_samples = len([r for r in experiment_results if r.model_version == config.model_b_version])
            
            # Calculate statistical power
            power_analysis = self._calculate_statistical_power(control_samples, treatment_samples)
            
            status = {
                "experiment_id": experiment_id,
                "name": config.name,
                "status": "active" if config.start_time and not config.end_time else "stopped",
                "start_time": config.start_time.isoformat() if config.start_time else None,
                "end_time": config.end_time.isoformat() if config.end_time else None,
                "total_samples": total_samples,
                "control_samples": control_samples,
                "treatment_samples": treatment_samples,
                "traffic_split_target": config.traffic_split,
                "traffic_split_actual": treatment_samples / max(total_samples, 1),
                "min_sample_size": config.min_sample_size,
                "sample_size_achieved": total_samples >= config.min_sample_size,
                "statistical_power": power_analysis,
                "models": {
                    "control": config.model_a_version,
                    "treatment": config.model_b_version
                }
            }
            
            return status
        
        except Exception as e:
            logger.error(f"Failed to get experiment status: {e}")
            raise
    
    def _calculate_statistical_power(self, control_n: int, treatment_n: int) -> Dict[str, Any]:
        """Calculate statistical power for current sample sizes"""
        
        # Simplified power calculation
        total_n = control_n + treatment_n
        
        if total_n < 100:
            power_level = "very_low"
            power_score = 0.2
        elif total_n < 500:
            power_level = "low"
            power_score = 0.5
        elif total_n < 1000:
            power_level = "moderate"
            power_score = 0.7
        elif total_n < 5000:
            power_level = "good"
            power_score = 0.8
        else:
            power_level = "excellent"
            power_score = 0.9
        
        return {
            "level": power_level,
            "score": power_score,
            "recommendation": self._get_power_recommendation(power_level, total_n)
        }
    
    def _get_power_recommendation(self, power_level: str, sample_size: int) -> str:
        """Get recommendation based on statistical power"""
        if power_level == "very_low":
            return f"Collect more data. Current sample size ({sample_size}) is too small for reliable results."
        elif power_level == "low":
            return f"Consider collecting more data for stronger statistical evidence."
        elif power_level == "moderate":
            return f"Sample size is adequate for detecting large effects."
        elif power_level == "good":
            return f"Good statistical power. Results should be reliable."
        else:
            return f"Excellent statistical power. Results are highly reliable."
    
    async def export_experiment_report(self, experiment_id: str, output_path: str = None) -> str:
        """Export detailed experiment report"""
        try:
            summary = await self.analyze_experiment(experiment_id)
            status = await self.get_experiment_status(experiment_id)
            
            report = {
                "experiment_summary": asdict(summary),
                "experiment_status": status,
                "statistical_tests": [asdict(test) for test in summary.statistical_tests],
                "export_timestamp": datetime.now().isoformat(),
                "recommendations": {
                    "primary": summary.recommendation,
                    "confidence": summary.confidence_score,
                    "next_steps": self._get_next_steps(summary)
                }
            }
            
            # Export to file
            if not output_path:
                output_path = self.storage_dir / f"experiment_{experiment_id}_report.json"
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Experiment report exported to {output_path}")
            
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Failed to export experiment report: {e}")
            raise
    
    def _get_next_steps(self, summary: ExperimentSummary) -> List[str]:
        """Get recommended next steps based on experiment results"""
        steps = []
        
        if summary.confidence_score < 0.7:
            steps.append("Collect more data to increase statistical confidence")
        
        if summary.total_samples < 1000:
            steps.append("Increase sample size for more robust results")
        
        if "deploy" in summary.recommendation.lower():
            steps.append("Plan gradual rollout of treatment model")
            steps.append("Monitor key metrics closely during deployment")
        
        if "investigate" in summary.recommendation.lower():
            steps.append("Analyze segment-level performance")
            steps.append("Review model differences and feature importance")
        
        if not steps:
            steps.append("Results are conclusive. Proceed with recommended action.")
        
        return steps

def main():
    """Main function for A/B testing demonstration"""
    
    async def run_ab_test_demo():
        # Initialize framework
        framework = ABTestFramework()
        await framework.initialize()
        
        # Create experiment configuration
        config = ExperimentConfig(
            experiment_id="demo_experiment_001",
            name="Model Version Comparison",
            description="Comparing baseline vs improved model performance",
            model_a_version="baseline_v1.0",
            model_b_version="improved_v2.0",
            traffic_split=0.5,
            split_strategy=TrafficSplitStrategy.HASH_BASED,
            min_sample_size=1000,
            confidence_level=0.95,
            success_metrics=["accuracy", "f1_score", "processing_time"]
        )
        
        # Create experiment
        experiment_id = await framework.create_experiment(config)
        
        # Start experiment
        await framework.start_experiment(experiment_id)
        
        print(f"Started A/B test experiment: {experiment_id}")
        
        # Simulate some predictions (this would be real traffic in production)
        for i in range(100):
            user_id = f"user_{i}"
            features = {
                "feature_1": np.random.normal(0, 1),
                "feature_2": np.random.uniform(0, 10),
                "feature_3": np.random.choice([0, 1])
            }
            
            try:
                result = await framework.get_prediction(
                    experiment_id, user_id, features
                )
                
                # Simulate actual outcome (in production, this comes later)
                actual_outcome = np.random.choice([0, 1], p=[0.7, 0.3])
                await framework.add_outcome(experiment_id, user_id, actual_outcome)
                
            except Exception as e:
                print(f"Prediction failed for user {user_id}: {e}")
        
        # Analyze results
        try:
            summary = await framework.analyze_experiment(experiment_id)
            print(f"\nExperiment Analysis:")
            print(f"Recommendation: {summary.recommendation}")
            print(f"Confidence: {summary.confidence_score:.2f}")
            print(f"Total samples: {summary.total_samples}")
            print(f"Control conversion: {summary.conversion_rate_control:.3f}")
            print(f"Treatment conversion: {summary.conversion_rate_treatment:.3f}")
            
            # Export report
            report_path = await framework.export_experiment_report(experiment_id)
            print(f"Report exported to: {report_path}")
            
        except Exception as e:
            print(f"Analysis failed: {e}")
        
        # Stop experiment
        await framework.stop_experiment(experiment_id)
        print("Experiment stopped")
    
    # Run demo
    asyncio.run(run_ab_test_demo())

if __name__ == "__main__":
    main()