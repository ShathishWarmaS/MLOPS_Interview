"""
Model Monitor for MLOps Pipeline
Real-time monitoring of model performance and data drift
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class PredictionLog:
    """Prediction log entry"""
    timestamp: float
    request_id: str
    features: Dict[str, Any]
    prediction: float
    confidence: float
    model_version: str
    latency_ms: float
    user_feedback: Optional[float] = None  # Actual label if available

@dataclass
class MetricsSnapshot:
    """Metrics snapshot at a point in time"""
    timestamp: float
    total_predictions: int
    error_rate: float
    average_latency: float
    p95_latency: float
    p99_latency: float
    prediction_distribution: Dict[str, int]
    feature_stats: Dict[str, Dict[str, float]]
    drift_scores: Dict[str, float]

class DataDriftDetector:
    """Statistical data drift detection"""
    
    def __init__(self, reference_window_size: int = 1000, detection_window_size: int = 100):
        self.reference_window_size = reference_window_size
        self.detection_window_size = detection_window_size
        self.reference_data = deque(maxlen=reference_window_size)
        self.current_window = deque(maxlen=detection_window_size)
        
    def add_reference_sample(self, features: Dict[str, Any]):
        """Add sample to reference distribution"""
        self.reference_data.append(features)
    
    def add_current_sample(self, features: Dict[str, Any]):
        """Add sample to current window"""
        self.current_window.append(features)
    
    def calculate_drift_scores(self) -> Dict[str, float]:
        """Calculate drift scores for each feature"""
        if len(self.reference_data) < 100 or len(self.current_window) < 20:
            return {}
        
        drift_scores = {}
        
        # Convert to DataFrames for easier processing
        ref_df = pd.DataFrame(list(self.reference_data))
        curr_df = pd.DataFrame(list(self.current_window))
        
        # Calculate drift for each numerical feature
        for column in ref_df.select_dtypes(include=[np.number]).columns:
            if column in curr_df.columns:
                drift_score = self._calculate_kolmogorov_smirnov(
                    ref_df[column].values,
                    curr_df[column].values
                )
                drift_scores[column] = drift_score
        
        # Calculate drift for categorical features
        for column in ref_df.select_dtypes(include=['object']).columns:
            if column in curr_df.columns:
                drift_score = self._calculate_categorical_drift(
                    ref_df[column].values,
                    curr_df[column].values
                )
                drift_scores[column] = drift_score
        
        return drift_scores
    
    def _calculate_kolmogorov_smirnov(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate KS test statistic"""
        try:
            from scipy import stats
            statistic, p_value = stats.ks_2samp(reference, current)
            return float(statistic)
        except ImportError:
            # Fallback implementation
            return self._simple_distribution_distance(reference, current)
    
    def _calculate_categorical_drift(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate categorical distribution drift"""
        ref_counts = pd.Series(reference).value_counts(normalize=True)
        curr_counts = pd.Series(current).value_counts(normalize=True)
        
        # Align categories
        all_categories = set(ref_counts.index) | set(curr_counts.index)
        ref_probs = [ref_counts.get(cat, 0) for cat in all_categories]
        curr_probs = [curr_counts.get(cat, 0) for cat in all_categories]
        
        # Calculate Jensen-Shannon divergence
        return self._jensen_shannon_divergence(ref_probs, curr_probs)
    
    def _simple_distribution_distance(self, ref: np.ndarray, curr: np.ndarray) -> float:
        """Simple distribution distance when scipy not available"""
        ref_mean, ref_std = np.mean(ref), np.std(ref)
        curr_mean, curr_std = np.mean(curr), np.std(curr)
        
        mean_diff = abs(ref_mean - curr_mean) / (ref_std + 1e-8)
        std_diff = abs(ref_std - curr_std) / (ref_std + 1e-8)
        
        return min(1.0, (mean_diff + std_diff) / 2)
    
    def _jensen_shannon_divergence(self, p: List[float], q: List[float]) -> float:
        """Calculate Jensen-Shannon divergence"""
        p = np.array(p) + 1e-8  # Add small constant to avoid log(0)
        q = np.array(q) + 1e-8
        m = (p + q) / 2
        
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        
        return float(np.sqrt((kl_pm + kl_qm) / 2))

class PerformanceMonitor:
    """Monitor model performance metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.start_time = time.time()
        
    def add_prediction(self, prediction_log: PredictionLog):
        """Add prediction to monitoring"""
        self.predictions.append(prediction_log)
        self.latencies.append(prediction_log.latency_ms)
        
        # Track errors (simple heuristic)
        is_error = prediction_log.latency_ms > 1000  # Consider high latency as error
        self.errors.append(1 if is_error else 0)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.predictions:
            return {}
        
        # Calculate latency statistics
        latencies = list(self.latencies)
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95) if len(latencies) > 20 else avg_latency
        p99_latency = np.percentile(latencies, 99) if len(latencies) > 100 else avg_latency
        
        # Calculate error rate
        error_rate = np.mean(self.errors) if self.errors else 0.0
        
        # Calculate throughput
        time_span = time.time() - self.start_time
        throughput = len(self.predictions) / max(time_span, 1)
        
        # Prediction distribution
        predictions_values = [p.prediction for p in self.predictions]
        prediction_mean = np.mean(predictions_values)
        prediction_std = np.std(predictions_values)
        
        return {
            'total_predictions': len(self.predictions),
            'predictions_per_second': throughput,
            'average_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'error_rate': error_rate,
            'prediction_mean': prediction_mean,
            'prediction_std': prediction_std,
            'uptime_seconds': time_span
        }

class ModelMonitor:
    """Comprehensive model monitoring system"""
    
    def __init__(self, 
                 db_path: str = "monitoring.db",
                 max_memory_logs: int = 10000,
                 drift_check_interval: int = 100):
        
        self.db_path = db_path
        self.max_memory_logs = max_memory_logs
        self.drift_check_interval = drift_check_interval
        
        # Components
        self.drift_detector = DataDriftDetector()
        self.performance_monitor = PerformanceMonitor()
        
        # Storage
        self.prediction_logs = deque(maxlen=max_memory_logs)
        self.metrics_history = deque(maxlen=1000)
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()
        
        # State
        self.is_initialized = False
        self.last_drift_check = 0
        self.alert_thresholds = {
            'error_rate': 0.05,
            'latency_p95': 1000,
            'drift_score': 0.3
        }
        
        # Counters
        self.prediction_count = 0
        self.alert_count = 0
    
    async def initialize(self):
        """Initialize monitoring system"""
        try:
            logger.info("Initializing model monitor...")
            
            # Setup database
            await self._setup_database()
            
            # Load reference data if available
            await self._load_reference_data()
            
            self.is_initialized = True
            logger.info("✅ Model monitor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize model monitor: {e}")
            raise
    
    async def _setup_database(self):
        """Setup SQLite database for persistent storage"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._create_database_tables
            )
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def _create_database_tables(self):
        """Create database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    request_id TEXT,
                    features TEXT,
                    prediction REAL,
                    confidence REAL,
                    model_version TEXT,
                    latency_ms REAL,
                    user_feedback REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    metrics TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    data TEXT
                )
            """)
    
    async def _load_reference_data(self):
        """Load reference data for drift detection"""
        try:
            # Load recent prediction logs as reference
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT features FROM prediction_logs 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                """)
                
                for row in cursor:
                    features = json.loads(row[0])
                    self.drift_detector.add_reference_sample(features)
            
            logger.info(f"Loaded {len(self.drift_detector.reference_data)} reference samples")
            
        except Exception as e:
            logger.warning(f"Could not load reference data: {e}")
    
    async def log_prediction(self, request, response, latency_ms: float):
        """Log a single prediction"""
        try:
            # Create prediction log
            pred_log = PredictionLog(
                timestamp=time.time(),
                request_id=getattr(response, 'request_id', 'unknown'),
                features=request.features if hasattr(request, 'features') else {},
                prediction=getattr(response, 'prediction', 0.0),
                confidence=getattr(response, 'confidence', 0.0),
                model_version=getattr(response, 'model_version', 'unknown'),
                latency_ms=latency_ms
            )
            
            # Add to memory storage
            with self.lock:
                self.prediction_logs.append(pred_log)
                self.prediction_count += 1
            
            # Add to monitors
            self.performance_monitor.add_prediction(pred_log)
            self.drift_detector.add_current_sample(pred_log.features)
            
            # Periodic drift check
            if self.prediction_count % self.drift_check_interval == 0:
                await self._check_drift()
            
            # Store in database (async)
            asyncio.create_task(self._store_prediction_log(pred_log))
            
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
    
    async def log_batch_prediction(self, request, response):
        """Log batch prediction"""
        try:
            # Log each prediction in the batch
            if hasattr(response, 'predictions'):
                for i, pred in enumerate(response.predictions):
                    sample_features = request.samples[i] if i < len(request.samples) else {}
                    
                    # Create mock request for individual prediction
                    class MockRequest:
                        def __init__(self, features):
                            self.features = features
                    
                    await self.log_prediction(
                        MockRequest(sample_features),
                        pred,
                        getattr(pred, 'latency_ms', 0.0)
                    )
            
        except Exception as e:
            logger.error(f"Failed to log batch prediction: {e}")
    
    async def _store_prediction_log(self, pred_log: PredictionLog):
        """Store prediction log in database"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._insert_prediction_log,
                pred_log
            )
        except Exception as e:
            logger.error(f"Failed to store prediction log: {e}")
    
    def _insert_prediction_log(self, pred_log: PredictionLog):
        """Insert prediction log into database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO prediction_logs 
                (timestamp, request_id, features, prediction, confidence, 
                 model_version, latency_ms, user_feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pred_log.timestamp,
                pred_log.request_id,
                json.dumps(pred_log.features),
                pred_log.prediction,
                pred_log.confidence,
                pred_log.model_version,
                pred_log.latency_ms,
                pred_log.user_feedback
            ))
    
    async def _check_drift(self):
        """Check for data drift"""
        try:
            if not self.drift_detector.reference_data:
                return
            
            # Calculate drift scores
            drift_scores = self.drift_detector.calculate_drift_scores()
            
            # Check for significant drift
            high_drift_features = []
            for feature, score in drift_scores.items():
                if score > self.alert_thresholds['drift_score']:
                    high_drift_features.append((feature, score))
            
            if high_drift_features:
                await self._create_alert(
                    alert_type="data_drift",
                    severity="warning",
                    message=f"High drift detected in features: {high_drift_features}",
                    data={"drift_scores": drift_scores}
                )
            
            logger.debug(f"Drift check completed. Scores: {drift_scores}")
            
        except Exception as e:
            logger.error(f"Drift check failed: {e}")
    
    async def _create_alert(self, alert_type: str, severity: str, message: str, data: Dict[str, Any]):
        """Create monitoring alert"""
        try:
            alert = {
                'timestamp': time.time(),
                'alert_type': alert_type,
                'severity': severity,
                'message': message,
                'data': data
            }
            
            # Log alert
            logger.warning(f"ALERT [{severity.upper()}] {alert_type}: {message}")
            
            # Store alert
            asyncio.create_task(self._store_alert(alert))
            
            with self.lock:
                self.alert_count += 1
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
    
    async def _store_alert(self, alert: Dict[str, Any]):
        """Store alert in database"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._insert_alert,
                alert
            )
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
    
    def _insert_alert(self, alert: Dict[str, Any]):
        """Insert alert into database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO alerts (timestamp, alert_type, severity, message, data)
                VALUES (?, ?, ?, ?, ?)
            """, (
                alert['timestamp'],
                alert['alert_type'],
                alert['severity'],
                alert['message'],
                json.dumps(alert['data'])
            ))
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics"""
        try:
            # Get performance metrics
            perf_metrics = self.performance_monitor.get_performance_metrics()
            
            # Get drift scores
            drift_scores = self.drift_detector.calculate_drift_scores()
            
            # Combine metrics
            metrics = {
                **perf_metrics,
                'drift_scores': drift_scores,
                'alert_count': self.alert_count,
                'monitoring_status': 'healthy' if self.is_initialized else 'initializing'
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return {}
    
    async def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history from database"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, metrics FROM metrics_snapshots 
                    WHERE timestamp > ? 
                    ORDER BY timestamp ASC
                """, (cutoff_time,))
                
                history = []
                for row in cursor:
                    metrics = json.loads(row[1])
                    metrics['timestamp'] = row[0]
                    history.append(metrics)
                
                return history
                
        except Exception as e:
            logger.error(f"Failed to get metrics history: {e}")
            return []
    
    async def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, alert_type, severity, message, data 
                    FROM alerts 
                    WHERE timestamp > ? 
                    ORDER BY timestamp DESC
                """, (cutoff_time,))
                
                alerts = []
                for row in cursor:
                    alert = {
                        'timestamp': row[0],
                        'alert_type': row[1],
                        'severity': row[2],
                        'message': row[3],
                        'data': json.loads(row[4])
                    }
                    alerts.append(alert)
                
                return alerts
                
        except Exception as e:
            logger.error(f"Failed to get recent alerts: {e}")
            return []
    
    async def add_user_feedback(self, request_id: str, actual_label: float):
        """Add user feedback for model improvement"""
        try:
            # Update prediction log with feedback
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE prediction_logs 
                    SET user_feedback = ? 
                    WHERE request_id = ?
                """, (actual_label, request_id))
            
            logger.info(f"Added user feedback for request {request_id}: {actual_label}")
            
        except Exception as e:
            logger.error(f"Failed to add user feedback: {e}")
    
    async def cleanup(self):
        """Cleanup monitoring resources"""
        logger.info("Cleaning up model monitor...")
        
        self.is_initialized = False
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("✅ Model monitor cleanup completed")