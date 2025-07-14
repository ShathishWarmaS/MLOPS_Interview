"""
Advanced Hyperparameter Tuning for MLOps Pipeline
Automated hyperparameter optimization with multiple strategies
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json
import mlflow
import mlflow.sklearn
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# ML Libraries
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, f1_score
import optuna
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

logger = logging.getLogger(__name__)

@dataclass
class HyperparameterSpace:
    """Hyperparameter search space definition"""
    param_name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Union[Tuple[float, float], List[Any]]
    log_scale: bool = False
    description: str = ""

@dataclass
class TuningResult:
    """Hyperparameter tuning result"""
    study_name: str
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    optimization_time: float
    convergence_history: List[float]
    model_type: str
    optimization_method: str

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization with multiple strategies"""
    
    def __init__(self, 
                 experiment_name: str = "hyperparameter_tuning",
                 n_jobs: int = -1,
                 random_state: int = 42):
        self.experiment_name = experiment_name
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.random_state = random_state
        
        # Optimization methods
        self.optimizers = {
            'optuna': self._optimize_with_optuna,
            'hyperopt': self._optimize_with_hyperopt,
            'scikit_optimize': self._optimize_with_skopt,
            'grid_search': self._optimize_with_grid_search,
            'random_search': self._optimize_with_random_search
        }
        
        # Model configurations
        self.model_configs = self._get_model_configurations()
        
        # Results storage
        self.results = {}
        self.executor = ThreadPoolExecutor(max_workers=self.n_jobs)
    
    def _get_model_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations with hyperparameter spaces"""
        return {
            'random_forest': {
                'model_class': RandomForestClassifier,
                'param_space': {
                    'n_estimators': HyperparameterSpace(
                        'n_estimators', 'discrete', [50, 500], 
                        description="Number of trees in the forest"
                    ),
                    'max_depth': HyperparameterSpace(
                        'max_depth', 'discrete', [3, 20],
                        description="Maximum depth of trees"
                    ),
                    'min_samples_split': HyperparameterSpace(
                        'min_samples_split', 'discrete', [2, 20],
                        description="Minimum samples required to split"
                    ),
                    'min_samples_leaf': HyperparameterSpace(
                        'min_samples_leaf', 'discrete', [1, 10],
                        description="Minimum samples required at leaf"
                    ),
                    'max_features': HyperparameterSpace(
                        'max_features', 'categorical', ['sqrt', 'log2', 'auto'],
                        description="Number of features for best split"
                    )
                }
            },
            'gradient_boosting': {
                'model_class': GradientBoostingClassifier,
                'param_space': {
                    'n_estimators': HyperparameterSpace(
                        'n_estimators', 'discrete', [50, 300],
                        description="Number of boosting stages"
                    ),
                    'learning_rate': HyperparameterSpace(
                        'learning_rate', 'continuous', [0.01, 0.3], log_scale=True,
                        description="Learning rate shrinks contribution"
                    ),
                    'max_depth': HyperparameterSpace(
                        'max_depth', 'discrete', [3, 10],
                        description="Maximum depth of trees"
                    ),
                    'subsample': HyperparameterSpace(
                        'subsample', 'continuous', [0.6, 1.0],
                        description="Fraction of samples for fitting"
                    ),
                    'min_samples_split': HyperparameterSpace(
                        'min_samples_split', 'discrete', [2, 20],
                        description="Minimum samples required to split"
                    )
                }
            },
            'logistic_regression': {
                'model_class': LogisticRegression,
                'param_space': {
                    'C': HyperparameterSpace(
                        'C', 'continuous', [0.001, 100], log_scale=True,
                        description="Inverse regularization strength"
                    ),
                    'penalty': HyperparameterSpace(
                        'penalty', 'categorical', ['l1', 'l2', 'elasticnet'],
                        description="Regularization penalty"
                    ),
                    'solver': HyperparameterSpace(
                        'solver', 'categorical', ['liblinear', 'saga', 'lbfgs'],
                        description="Optimization algorithm"
                    ),
                    'max_iter': HyperparameterSpace(
                        'max_iter', 'discrete', [100, 1000],
                        description="Maximum iterations"
                    )
                }
            },
            'svm': {
                'model_class': SVC,
                'param_space': {
                    'C': HyperparameterSpace(
                        'C', 'continuous', [0.1, 100], log_scale=True,
                        description="Regularization parameter"
                    ),
                    'kernel': HyperparameterSpace(
                        'kernel', 'categorical', ['linear', 'rbf', 'poly'],
                        description="Kernel type"
                    ),
                    'gamma': HyperparameterSpace(
                        'gamma', 'categorical', ['scale', 'auto'],
                        description="Kernel coefficient"
                    ),
                    'degree': HyperparameterSpace(
                        'degree', 'discrete', [2, 5],
                        description="Degree for poly kernel"
                    )
                }
            },
            'neural_network': {
                'model_class': MLPClassifier,
                'param_space': {
                    'hidden_layer_sizes': HyperparameterSpace(
                        'hidden_layer_sizes', 'categorical', 
                        [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                        description="Hidden layer architecture"
                    ),
                    'activation': HyperparameterSpace(
                        'activation', 'categorical', ['relu', 'tanh', 'logistic'],
                        description="Activation function"
                    ),
                    'alpha': HyperparameterSpace(
                        'alpha', 'continuous', [0.0001, 0.1], log_scale=True,
                        description="L2 regularization term"
                    ),
                    'learning_rate': HyperparameterSpace(
                        'learning_rate', 'categorical', ['constant', 'adaptive'],
                        description="Learning rate schedule"
                    ),
                    'max_iter': HyperparameterSpace(
                        'max_iter', 'discrete', [200, 800],
                        description="Maximum iterations"
                    )
                }
            }
        }
    
    async def optimize_model(self, 
                           model_name: str,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray,
                           optimization_method: str = 'optuna',
                           n_trials: int = 100,
                           timeout: Optional[float] = None) -> TuningResult:
        """Optimize hyperparameters for a specific model"""
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        if optimization_method not in self.optimizers:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        logger.info(f"Starting hyperparameter optimization for {model_name} "
                   f"using {optimization_method}")
        
        start_time = time.time()
        
        try:
            # Run optimization
            optimizer_func = self.optimizers[optimization_method]
            result = await optimizer_func(
                model_name, X_train, y_train, X_val, y_val, n_trials, timeout
            )
            
            optimization_time = time.time() - start_time
            result.optimization_time = optimization_time
            result.optimization_method = optimization_method
            
            # Store result
            self.results[f"{model_name}_{optimization_method}"] = result
            
            # Log to MLflow
            await self._log_to_mlflow(result, X_train, y_train, X_val, y_val)
            
            logger.info(f"Optimization completed for {model_name}: "
                       f"best_score={result.best_score:.4f} in {optimization_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed for {model_name}: {e}")
            raise
    
    async def _optimize_with_optuna(self,
                                  model_name: str,
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_val: np.ndarray,
                                  y_val: np.ndarray,
                                  n_trials: int,
                                  timeout: Optional[float]) -> TuningResult:
        """Optimize using Optuna"""
        
        model_config = self.model_configs[model_name]
        param_space = model_config['param_space']
        
        # Create study
        study_name = f"{model_name}_optuna_{int(time.time())}"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        def objective(trial):
            try:
                # Sample hyperparameters
                params = {}
                for param_name, param_space_def in param_space.items():
                    if param_space_def.param_type == 'continuous':
                        if param_space_def.log_scale:
                            params[param_name] = trial.suggest_loguniform(
                                param_name, *param_space_def.bounds
                            )
                        else:
                            params[param_name] = trial.suggest_uniform(
                                param_name, *param_space_def.bounds
                            )
                    elif param_space_def.param_type == 'discrete':
                        params[param_name] = trial.suggest_int(
                            param_name, *param_space_def.bounds
                        )
                    elif param_space_def.param_type == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_space_def.bounds
                        )
                
                # Handle special parameter combinations
                params = self._validate_params(model_name, params)
                
                # Train and evaluate model
                model = model_config['model_class'](
                    random_state=self.random_state,
                    **params
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Extract results
        convergence_history = [trial.value for trial in study.trials if trial.value]
        
        return TuningResult(
            study_name=study_name,
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            optimization_time=0.0,  # Will be set by caller
            convergence_history=convergence_history,
            model_type=model_name,
            optimization_method='optuna'
        )
    
    async def _optimize_with_hyperopt(self,
                                    model_name: str,
                                    X_train: np.ndarray,
                                    y_train: np.ndarray,
                                    X_val: np.ndarray,
                                    y_val: np.ndarray,
                                    n_trials: int,
                                    timeout: Optional[float]) -> TuningResult:
        """Optimize using Hyperopt"""
        
        model_config = self.model_configs[model_name]
        param_space = model_config['param_space']
        
        # Define search space
        space = {}
        for param_name, param_space_def in param_space.items():
            if param_space_def.param_type == 'continuous':
                if param_space_def.log_scale:
                    space[param_name] = hp.loguniform(
                        param_name, 
                        np.log(param_space_def.bounds[0]),
                        np.log(param_space_def.bounds[1])
                    )
                else:
                    space[param_name] = hp.uniform(
                        param_name, *param_space_def.bounds
                    )
            elif param_space_def.param_type == 'discrete':
                space[param_name] = hp.randint(
                    param_name, 
                    param_space_def.bounds[0],
                    param_space_def.bounds[1] + 1
                )
            elif param_space_def.param_type == 'categorical':
                space[param_name] = hp.choice(param_name, param_space_def.bounds)
        
        def objective(params):
            try:
                # Handle discrete parameters
                for param_name, param_space_def in param_space.items():
                    if param_space_def.param_type == 'discrete':
                        params[param_name] = int(params[param_name])
                
                # Validate parameters
                params = self._validate_params(model_name, params)
                
                # Train and evaluate model
                model = model_config['model_class'](
                    random_state=self.random_state,
                    **params
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                
                return {'loss': -score, 'status': STATUS_OK}
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return {'loss': -0.0, 'status': STATUS_OK}
        
        # Run optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=n_trials,
            trials=trials,
            rstate=np.random.RandomState(self.random_state)
        )
        
        # Extract results
        convergence_history = [-trial['result']['loss'] for trial in trials.trials]
        best_score = max(convergence_history) if convergence_history else 0.0
        
        return TuningResult(
            study_name=f"{model_name}_hyperopt_{int(time.time())}",
            best_params=best,
            best_score=best_score,
            n_trials=len(trials.trials),
            optimization_time=0.0,
            convergence_history=convergence_history,
            model_type=model_name,
            optimization_method='hyperopt'
        )
    
    async def _optimize_with_skopt(self,
                                 model_name: str,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 X_val: np.ndarray,
                                 y_val: np.ndarray,
                                 n_trials: int,
                                 timeout: Optional[float]) -> TuningResult:
        """Optimize using scikit-optimize"""
        
        model_config = self.model_configs[model_name]
        param_space = model_config['param_space']
        
        # Define search space
        dimensions = []
        param_names = []
        
        for param_name, param_space_def in param_space.items():
            param_names.append(param_name)
            
            if param_space_def.param_type == 'continuous':
                if param_space_def.log_scale:
                    dimensions.append(Real(*param_space_def.bounds, prior='log-uniform'))
                else:
                    dimensions.append(Real(*param_space_def.bounds))
            elif param_space_def.param_type == 'discrete':
                dimensions.append(Integer(*param_space_def.bounds))
            elif param_space_def.param_type == 'categorical':
                dimensions.append(Categorical(param_space_def.bounds))
        
        def objective(param_values):
            try:
                # Create parameter dictionary
                params = dict(zip(param_names, param_values))
                
                # Validate parameters
                params = self._validate_params(model_name, params)
                
                # Train and evaluate model
                model = model_config['model_class'](
                    random_state=self.random_state,
                    **params
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                
                return -score  # Minimize negative score
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_trials,
            random_state=self.random_state,
            acq_func='EI'
        )
        
        # Extract results
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun
        convergence_history = [-y for y in result.func_vals]
        
        return TuningResult(
            study_name=f"{model_name}_skopt_{int(time.time())}",
            best_params=best_params,
            best_score=best_score,
            n_trials=len(result.func_vals),
            optimization_time=0.0,
            convergence_history=convergence_history,
            model_type=model_name,
            optimization_method='scikit_optimize'
        )
    
    async def _optimize_with_grid_search(self,
                                       model_name: str,
                                       X_train: np.ndarray,
                                       y_train: np.ndarray,
                                       X_val: np.ndarray,
                                       y_val: np.ndarray,
                                       n_trials: int,
                                       timeout: Optional[float]) -> TuningResult:
        """Optimize using Grid Search"""
        
        model_config = self.model_configs[model_name]
        param_space = model_config['param_space']
        
        # Create grid search space (simplified for grid search)
        param_grid = {}
        for param_name, param_space_def in param_space.items():
            if param_space_def.param_type == 'continuous':
                # Create discrete grid for continuous parameters
                low, high = param_space_def.bounds
                if param_space_def.log_scale:
                    values = np.logspace(np.log10(low), np.log10(high), num=5)
                else:
                    values = np.linspace(low, high, num=5)
                param_grid[param_name] = values.tolist()
            elif param_space_def.param_type == 'discrete':
                # Sample from discrete range
                low, high = param_space_def.bounds
                values = np.linspace(low, high, num=min(5, high - low + 1), dtype=int)
                param_grid[param_name] = values.tolist()
            elif param_space_def.param_type == 'categorical':
                param_grid[param_name] = param_space_def.bounds
        
        # Combine train and validation for cross-validation
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])
        
        # Create model
        model = model_config['model_class'](random_state=self.random_state)
        
        # Run grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring=make_scorer(f1_score, average='weighted'),
            n_jobs=min(self.n_jobs, 4),  # Limit for grid search
            verbose=1
        )
        
        grid_search.fit(X_combined, y_combined)
        
        # Extract results
        convergence_history = [result.mean_test_score for result in grid_search.cv_results_['mean_test_score']]
        
        return TuningResult(
            study_name=f"{model_name}_grid_search_{int(time.time())}",
            best_params=grid_search.best_params_,
            best_score=grid_search.best_score_,
            n_trials=len(grid_search.cv_results_['mean_test_score']),
            optimization_time=0.0,
            convergence_history=convergence_history,
            model_type=model_name,
            optimization_method='grid_search'
        )
    
    async def _optimize_with_random_search(self,
                                         model_name: str,
                                         X_train: np.ndarray,
                                         y_train: np.ndarray,
                                         X_val: np.ndarray,
                                         y_val: np.ndarray,
                                         n_trials: int,
                                         timeout: Optional[float]) -> TuningResult:
        """Optimize using Random Search"""
        
        model_config = self.model_configs[model_name]
        param_space = model_config['param_space']
        
        # Create random search distributions
        param_distributions = {}
        for param_name, param_space_def in param_space.items():
            if param_space_def.param_type == 'continuous':
                from scipy.stats import uniform, loguniform
                low, high = param_space_def.bounds
                if param_space_def.log_scale:
                    param_distributions[param_name] = loguniform(low, high)
                else:
                    param_distributions[param_name] = uniform(low, high - low)
            elif param_space_def.param_type == 'discrete':
                from scipy.stats import randint
                low, high = param_space_def.bounds
                param_distributions[param_name] = randint(low, high + 1)
            elif param_space_def.param_type == 'categorical':
                param_distributions[param_name] = param_space_def.bounds
        
        # Combine train and validation for cross-validation
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])
        
        # Create model
        model = model_config['model_class'](random_state=self.random_state)
        
        # Run random search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_trials,
            cv=3,
            scoring=make_scorer(f1_score, average='weighted'),
            n_jobs=min(self.n_jobs, 4),
            random_state=self.random_state,
            verbose=1
        )
        
        random_search.fit(X_combined, y_combined)
        
        # Extract results
        convergence_history = list(random_search.cv_results_['mean_test_score'])
        
        return TuningResult(
            study_name=f"{model_name}_random_search_{int(time.time())}",
            best_params=random_search.best_params_,
            best_score=random_search.best_score_,
            n_trials=len(random_search.cv_results_['mean_test_score']),
            optimization_time=0.0,
            convergence_history=convergence_history,
            model_type=model_name,
            optimization_method='random_search'
        )
    
    def _validate_params(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix parameter combinations"""
        
        if model_name == 'logistic_regression':
            # Handle solver-penalty compatibility
            if params.get('penalty') == 'elasticnet' and params.get('solver') != 'saga':
                params['solver'] = 'saga'
            elif params.get('penalty') == 'l1' and params.get('solver') not in ['liblinear', 'saga']:
                params['solver'] = 'liblinear'
        
        elif model_name == 'svm':
            # Handle kernel-specific parameters
            if params.get('kernel') != 'poly':
                params.pop('degree', None)
        
        elif model_name == 'neural_network':
            # Ensure reasonable max_iter
            if params.get('max_iter', 200) < 200:
                params['max_iter'] = 200
        
        return params
    
    async def _log_to_mlflow(self,
                           result: TuningResult,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray):
        """Log results to MLflow"""
        
        try:
            with mlflow.start_run(run_name=f"hyperopt_{result.study_name}"):
                # Log parameters
                mlflow.log_params({
                    "model_type": result.model_type,
                    "optimization_method": result.optimization_method,
                    "n_trials": result.n_trials,
                    "optimization_time": result.optimization_time
                })
                
                # Log best hyperparameters
                for param_name, param_value in result.best_params.items():
                    mlflow.log_param(f"best_{param_name}", param_value)
                
                # Log metrics
                mlflow.log_metric("best_score", result.best_score)
                mlflow.log_metric("trials_completed", result.n_trials)
                mlflow.log_metric("optimization_time_seconds", result.optimization_time)
                
                # Log convergence history
                for i, score in enumerate(result.convergence_history):
                    mlflow.log_metric("convergence_score", score, step=i)
                
                # Train and log best model
                model_config = self.model_configs[result.model_type]
                best_model = model_config['model_class'](
                    random_state=self.random_state,
                    **result.best_params
                )
                
                best_model.fit(X_train, y_train)
                
                # Log model
                mlflow.sklearn.log_model(
                    best_model,
                    f"best_{result.model_type}_model",
                    registered_model_name=f"hyperopt_{result.model_type}"
                )
                
                # Final evaluation
                y_train_pred = best_model.predict(X_train)
                y_val_pred = best_model.predict(X_val)
                
                train_score = f1_score(y_train, y_train_pred, average='weighted')
                val_score = f1_score(y_val, y_val_pred, average='weighted')
                
                mlflow.log_metric("final_train_score", train_score)
                mlflow.log_metric("final_val_score", val_score)
                
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")
    
    async def optimize_all_models(self,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_val: np.ndarray,
                                y_val: np.ndarray,
                                optimization_methods: List[str] = None,
                                n_trials_per_method: int = 50) -> Dict[str, TuningResult]:
        """Optimize all models with multiple methods"""
        
        if optimization_methods is None:
            optimization_methods = ['optuna', 'random_search']
        
        logger.info(f"Starting optimization for {len(self.model_configs)} models "
                   f"with {len(optimization_methods)} methods")
        
        all_results = {}
        
        # Run optimizations
        tasks = []
        for model_name in self.model_configs.keys():
            for method in optimization_methods:
                task = self.optimize_model(
                    model_name, X_train, y_train, X_val, y_val,
                    method, n_trials_per_method
                )
                tasks.append((f"{model_name}_{method}", task))
        
        # Execute all optimizations
        for task_name, task in tasks:
            try:
                result = await task
                all_results[task_name] = result
                logger.info(f"Completed {task_name}: score={result.best_score:.4f}")
            except Exception as e:
                logger.error(f"Failed {task_name}: {e}")
        
        # Find overall best
        best_result = max(all_results.values(), key=lambda r: r.best_score)
        logger.info(f"Overall best: {best_result.model_type} with "
                   f"{best_result.optimization_method} "
                   f"(score={best_result.best_score:.4f})")
        
        return all_results
    
    async def compare_optimization_methods(self,
                                         model_name: str,
                                         X_train: np.ndarray,
                                         y_train: np.ndarray,
                                         X_val: np.ndarray,
                                         y_val: np.ndarray,
                                         n_trials: int = 100) -> Dict[str, TuningResult]:
        """Compare different optimization methods for a single model"""
        
        logger.info(f"Comparing optimization methods for {model_name}")
        
        results = {}
        methods = ['optuna', 'hyperopt', 'random_search']
        
        for method in methods:
            try:
                result = await self.optimize_model(
                    model_name, X_train, y_train, X_val, y_val,
                    method, n_trials
                )
                results[method] = result
                
                logger.info(f"{method}: best_score={result.best_score:.4f}, "
                           f"time={result.optimization_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Method {method} failed: {e}")
        
        return results
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results"""
        
        if not self.results:
            return {"message": "No optimization results available"}
        
        summary = {
            "total_optimizations": len(self.results),
            "models_optimized": set(),
            "methods_used": set(),
            "best_overall": None,
            "results_by_model": {},
            "results_by_method": {}
        }
        
        best_score = -1
        best_result = None
        
        for result_name, result in self.results.items():
            summary["models_optimized"].add(result.model_type)
            summary["methods_used"].add(result.optimization_method)
            
            if result.best_score > best_score:
                best_score = result.best_score
                best_result = result
            
            # Group by model
            if result.model_type not in summary["results_by_model"]:
                summary["results_by_model"][result.model_type] = []
            summary["results_by_model"][result.model_type].append({
                "method": result.optimization_method,
                "score": result.best_score,
                "time": result.optimization_time,
                "trials": result.n_trials
            })
            
            # Group by method
            if result.optimization_method not in summary["results_by_method"]:
                summary["results_by_method"][result.optimization_method] = []
            summary["results_by_method"][result.optimization_method].append({
                "model": result.model_type,
                "score": result.best_score,
                "time": result.optimization_time,
                "trials": result.n_trials
            })
        
        if best_result:
            summary["best_overall"] = {
                "model": best_result.model_type,
                "method": best_result.optimization_method,
                "score": best_result.best_score,
                "params": best_result.best_params,
                "time": best_result.optimization_time
            }
        
        # Convert sets to lists for JSON serialization
        summary["models_optimized"] = list(summary["models_optimized"])
        summary["methods_used"] = list(summary["methods_used"])
        
        return summary

def main():
    """Main function for hyperparameter optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization')
    parser.add_argument('--model', type=str, choices=['random_forest', 'gradient_boosting', 'logistic_regression', 'svm', 'neural_network'],
                       help='Model to optimize')
    parser.add_argument('--method', type=str, choices=['optuna', 'hyperopt', 'scikit_optimize', 'grid_search', 'random_search'],
                       default='optuna', help='Optimization method')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--data-path', type=str, help='Path to training data')
    parser.add_argument('--target-column', type=str, default='target', help='Target column name')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    async def run_optimization():
        # Initialize optimizer
        optimizer = HyperparameterOptimizer()
        
        # Load data (mock data for demonstration)
        if args.data_path:
            data = pd.read_csv(args.data_path)
            X = data.drop(columns=[args.target_column])
            y = data[args.target_column]
        else:
            # Generate mock data
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=10000, n_features=20, 
                                     n_informative=15, n_redundant=5,
                                     random_state=42)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Run optimization
        if args.model:
            result = await optimizer.optimize_model(
                args.model, X_train, y_train, X_val, y_val,
                args.method, args.trials
            )
            print(f"Best score: {result.best_score:.4f}")
            print(f"Best params: {result.best_params}")
        else:
            results = await optimizer.optimize_all_models(
                X_train, y_train, X_val, y_val,
                [args.method], args.trials
            )
            
            # Print summary
            summary = optimizer.get_optimization_summary()
            print(json.dumps(summary, indent=2, default=str))
    
    # Run optimization
    asyncio.run(run_optimization())

if __name__ == "__main__":
    main()