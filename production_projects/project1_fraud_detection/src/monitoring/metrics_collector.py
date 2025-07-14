#!/usr/bin/env python3
"""
Metrics Collector - Production Monitoring System
Real-time metrics collection and monitoring for fraud detection API
"""

import asyncio
import time
import logging
import json
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import statistics
from dataclasses import dataclass, asdict
import threading
from datetime import datetime, timedelta
import uuid

from api.models import TransactionRequest
from shared.config import config

logger = logging.getLogger(__name__)

@dataclass
class PredictionMetric:
    """Individual prediction metric"""
    timestamp: float
    request_id: str
    transaction_id: str
    fraud_probability: float
    risk_level: str
    confidence: float
    latency_ms: float
    model_version: str
    user_id: Optional[str] = None
    amount: Optional[float] = None
    merchant_category: Optional[str] = None

@dataclass
class ErrorMetric:
    """Error metric"""
    timestamp: float
    error_type: str
    error_message: str
    request_id: str
    endpoint: str
    status_code: int

@dataclass
class PerformanceMetric:
    """Performance metric"""
    timestamp: float
    metric_type: str
    value: float
    tags: Dict[str, str]

class MetricsCollector:
    """
    Production-grade metrics collector
    
    Features:
    - Real-time metric collection
    - Aggregation and statistics
    - Performance monitoring
    - Error tracking
    - Alerting thresholds
    - Memory-efficient storage
    """
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.start_time = time.time()
        
        # Metrics storage
        self.prediction_metrics = deque(maxlen=max_history_size)
        self.error_metrics = deque(maxlen=max_history_size)
        self.performance_metrics = deque(maxlen=max_history_size)
        
        # Aggregated metrics
        self.total_predictions = 0
        self.total_errors = 0
        self.fraud_predictions = 0
        self.high_risk_predictions = 0
        
        # Performance tracking
        self.latency_history = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.last_error = None
        
        # Alerting thresholds
        self.alert_thresholds = {
            'high_latency_ms': 1000,
            'high_error_rate': 0.05,
            'high_fraud_rate': 0.1,
            'low_confidence': 0.5
        }
        
        # Background tasks
        self.background_tasks = set()
        self.is_running = False
        self.lock = threading.Lock()
        
        logger.info("📊 Metrics collector initialized")
    
    async def initialize(self):
        """Initialize the metrics collector"""
        try:
            logger.info("🚀 Starting metrics collector...")
            
            self.is_running = True
            
            # Start background tasks
            task1 = asyncio.create_task(self._metrics_aggregation_loop())
            task2 = asyncio.create_task(self._performance_monitoring_loop())
            task3 = asyncio.create_task(self._alerting_loop())
            
            self.background_tasks.update([task1, task2, task3])
            
            logger.info("✅ Metrics collector started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize metrics collector: {e}")
            raise
    
    async def log_prediction(
        self,
        request: TransactionRequest,
        prediction: Dict[str, Any],
        latency: float,
        request_id: str
    ):
        """Log a prediction event"""
        try:
            metric = PredictionMetric(
                timestamp=time.time(),
                request_id=request_id,
                transaction_id=request.transaction_id,
                fraud_probability=prediction['fraud_probability'],
                risk_level=prediction['risk_level'],
                confidence=prediction['confidence'],
                latency_ms=latency,
                model_version=prediction['model_version'],
                user_id=getattr(request, 'user_id', None),
                amount=request.amount,
                merchant_category=request.merchant_category
            )
            
            # Store metric
            with self.lock:
                self.prediction_metrics.append(metric)
                self.total_predictions += 1
                
                # Update aggregated metrics
                if prediction['fraud_probability'] > 0.5:
                    self.fraud_predictions += 1
                
                if prediction['risk_level'] == 'high':
                    self.high_risk_predictions += 1
                
                # Update latency history
                self.latency_history.append(latency)
            
            # Log for debugging
            logger.debug(f"Prediction metric logged: {request_id}")
            
        except Exception as e:
            logger.error(f"Failed to log prediction metric: {e}")
    
    async def log_error(
        self,
        error_type: str,
        error_message: str,
        request_id: str,
        endpoint: str,
        status_code: int
    ):
        """Log an error event"""
        try:
            metric = ErrorMetric(
                timestamp=time.time(),
                error_type=error_type,
                error_message=error_message,
                request_id=request_id,
                endpoint=endpoint,
                status_code=status_code
            )
            
            # Store metric
            with self.lock:
                self.error_metrics.append(metric)
                self.total_errors += 1
                self.error_counts[error_type] += 1
                self.last_error = error_message
            
            logger.debug(f"Error metric logged: {error_type} - {error_message}")
            
        except Exception as e:
            logger.error(f"Failed to log error metric: {e}")
    
    async def log_performance(
        self,
        metric_type: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Log a performance metric"""
        try:
            metric = PerformanceMetric(
                timestamp=time.time(),
                metric_type=metric_type,
                value=value,
                tags=tags or {}
            )
            
            # Store metric
            with self.lock:
                self.performance_metrics.append(metric)
            
            logger.debug(f"Performance metric logged: {metric_type} = {value}")
            
        except Exception as e:
            logger.error(f"Failed to log performance metric: {e}")
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current aggregated metrics"""
        try:
            with self.lock:
                # Calculate derived metrics
                uptime_seconds = time.time() - self.start_time
                
                # Calculate rates
                predictions_per_second = self.total_predictions / uptime_seconds if uptime_seconds > 0 else 0
                error_rate = self.total_errors / max(self.total_predictions, 1)
                fraud_rate = self.fraud_predictions / max(self.total_predictions, 1)
                high_risk_rate = self.high_risk_predictions / max(self.total_predictions, 1)
                
                # Calculate latency statistics
                latency_stats = {}
                if self.latency_history:
                    latencies = list(self.latency_history)
                    latency_stats = {
                        'average_latency_ms': statistics.mean(latencies),
                        'median_latency_ms': statistics.median(latencies),
                        'p95_latency_ms': self._percentile(latencies, 0.95),
                        'p99_latency_ms': self._percentile(latencies, 0.99),
                        'max_latency_ms': max(latencies),
                        'min_latency_ms': min(latencies)
                    }
                else:
                    latency_stats = {
                        'average_latency_ms': 0,
                        'median_latency_ms': 0,
                        'p95_latency_ms': 0,
                        'p99_latency_ms': 0,
                        'max_latency_ms': 0,
                        'min_latency_ms': 0
                    }
                
                # Recent metrics (last 5 minutes)
                recent_threshold = time.time() - 300
                recent_predictions = [
                    m for m in self.prediction_metrics 
                    if m.timestamp > recent_threshold
                ]
                recent_errors = [
                    m for m in self.error_metrics 
                    if m.timestamp > recent_threshold
                ]
                
                # Calculate recent rates
                recent_prediction_rate = len(recent_predictions) / 300 if recent_predictions else 0
                recent_error_rate = len(recent_errors) / max(len(recent_predictions), 1)
                
                return {
                    # Core metrics
                    'total_predictions': self.total_predictions,
                    'total_errors': self.total_errors,
                    'fraud_predictions': self.fraud_predictions,
                    'high_risk_predictions': self.high_risk_predictions,
                    
                    # Rates
                    'predictions_per_second': predictions_per_second,
                    'error_rate': error_rate,
                    'fraud_rate': fraud_rate,
                    'high_risk_rate': high_risk_rate,
                    
                    # Recent metrics
                    'recent_prediction_rate': recent_prediction_rate,
                    'recent_error_rate': recent_error_rate,
                    
                    # Latency statistics
                    **latency_stats,
                    
                    # System metrics
                    'uptime_seconds': uptime_seconds,
                    'metrics_history_size': len(self.prediction_metrics),
                    'error_history_size': len(self.error_metrics),
                    
                    # Error breakdown
                    'error_types': dict(self.error_counts),
                    'last_error': self.last_error,
                    
                    # Timestamp
                    'timestamp': time.time()
                }
        
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return {'error': str(e)}
    
    async def get_detailed_metrics(self, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get detailed metrics for a specific time range"""
        try:
            threshold = time.time() - (time_range_minutes * 60)
            
            with self.lock:
                # Filter metrics by time range
                filtered_predictions = [
                    m for m in self.prediction_metrics 
                    if m.timestamp > threshold
                ]
                filtered_errors = [
                    m for m in self.error_metrics 
                    if m.timestamp > threshold
                ]
                
                # Group by time buckets (5-minute intervals)
                time_buckets = defaultdict(list)
                for metric in filtered_predictions:
                    bucket = int(metric.timestamp // 300) * 300
                    time_buckets[bucket].append(metric)
                
                # Calculate time-series data
                time_series = {}
                for bucket_time, metrics in time_buckets.items():
                    time_series[bucket_time] = {
                        'prediction_count': len(metrics),
                        'fraud_count': sum(1 for m in metrics if m.fraud_probability > 0.5),
                        'avg_latency': statistics.mean([m.latency_ms for m in metrics]),
                        'avg_confidence': statistics.mean([m.confidence for m in metrics])
                    }
                
                return {
                    'time_range_minutes': time_range_minutes,
                    'total_predictions': len(filtered_predictions),
                    'total_errors': len(filtered_errors),
                    'time_series': time_series,
                    'merchant_categories': self._group_by_field(filtered_predictions, 'merchant_category'),
                    'risk_levels': self._group_by_field(filtered_predictions, 'risk_level'),
                    'model_versions': self._group_by_field(filtered_predictions, 'model_version')
                }
        
        except Exception as e:
            logger.error(f"Failed to get detailed metrics: {e}")
            return {'error': str(e)}
    
    def _group_by_field(self, metrics: List[PredictionMetric], field: str) -> Dict[str, int]:
        """Group metrics by a specific field"""
        groups = defaultdict(int)
        for metric in metrics:
            value = getattr(metric, field, None)
            if value:
                groups[value] += 1
        return dict(groups)
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * percentile
        f = int(k)
        c = k - f
        
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    async def _metrics_aggregation_loop(self):
        """Background task for metrics aggregation"""
        while self.is_running:
            try:
                # Calculate throughput
                current_time = time.time()
                current_predictions = self.total_predictions
                
                # Store throughput sample
                self.throughput_history.append({
                    'timestamp': current_time,
                    'predictions': current_predictions
                })
                
                # Clean up old throughput data
                if len(self.throughput_history) > 1:
                    cutoff_time = current_time - 3600  # 1 hour
                    self.throughput_history = deque([
                        sample for sample in self.throughput_history 
                        if sample['timestamp'] > cutoff_time
                    ], maxlen=100)
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in metrics aggregation loop: {e}")
                await asyncio.sleep(60)
    
    async def _performance_monitoring_loop(self):
        """Background task for performance monitoring"""
        while self.is_running:
            try:
                # Monitor system performance
                await self._monitor_system_performance()
                
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _monitor_system_performance(self):
        """Monitor system performance metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.log_performance('cpu_usage_percent', cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            await self.log_performance('memory_usage_percent', memory.percent)
            await self.log_performance('memory_used_bytes', memory.used)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            await self.log_performance('disk_usage_percent', disk.percent)
            
        except ImportError:
            # psutil not available, skip system monitoring
            pass
        except Exception as e:
            logger.error(f"Error monitoring system performance: {e}")
    
    async def _alerting_loop(self):
        """Background task for alerting"""
        while self.is_running:
            try:
                await self._check_alerts()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in alerting loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_alerts(self):
        """Check for alert conditions"""
        try:
            current_metrics = await self.get_current_metrics()
            
            # Check latency alerts
            if current_metrics.get('average_latency_ms', 0) > self.alert_thresholds['high_latency_ms']:
                await self._send_alert(
                    'high_latency',
                    f"High latency detected: {current_metrics['average_latency_ms']:.2f}ms"
                )
            
            # Check error rate alerts
            if current_metrics.get('error_rate', 0) > self.alert_thresholds['high_error_rate']:
                await self._send_alert(
                    'high_error_rate',
                    f"High error rate detected: {current_metrics['error_rate']:.2%}"
                )
            
            # Check fraud rate alerts
            if current_metrics.get('fraud_rate', 0) > self.alert_thresholds['high_fraud_rate']:
                await self._send_alert(
                    'high_fraud_rate',
                    f"High fraud rate detected: {current_metrics['fraud_rate']:.2%}"
                )
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def _send_alert(self, alert_type: str, message: str):
        """Send alert (placeholder implementation)"""
        logger.warning(f"🚨 ALERT [{alert_type}]: {message}")
        
        # In production, integrate with:
        # - Slack notifications
        # - Email alerts
        # - PagerDuty
        # - Monitoring systems (Prometheus AlertManager)
    
    async def health_check(self):
        """Health check for metrics collector"""
        try:
            # Check if background tasks are running
            running_tasks = sum(1 for task in self.background_tasks if not task.done())
            
            if running_tasks < len(self.background_tasks):
                raise RuntimeError("Some background tasks are not running")
            
            # Check metrics storage
            if len(self.prediction_metrics) >= self.max_history_size:
                logger.warning("Metrics storage is at capacity")
            
            return {
                'status': 'healthy',
                'running_tasks': running_tasks,
                'metrics_count': len(self.prediction_metrics),
                'errors_count': len(self.error_metrics)
            }
        
        except Exception as e:
            logger.error(f"Metrics collector health check failed: {e}")
            raise
    
    async def close(self):
        """Close the metrics collector"""
        try:
            logger.info("🛑 Stopping metrics collector...")
            
            self.is_running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            logger.info("✅ Metrics collector stopped")
            
        except Exception as e:
            logger.error(f"Error closing metrics collector: {e}")
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in various formats"""
        try:
            if format == 'json':
                return json.dumps(asyncio.run(self.get_current_metrics()), indent=2)
            elif format == 'prometheus':
                return self._export_prometheus_format()
            else:
                raise ValueError(f"Unsupported export format: {format}")
        
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return f"Error exporting metrics: {e}"
    
    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        try:
            metrics = asyncio.run(self.get_current_metrics())
            
            lines = []
            lines.append("# HELP fraud_predictions_total Total number of fraud predictions")
            lines.append("# TYPE fraud_predictions_total counter")
            lines.append(f"fraud_predictions_total {metrics['total_predictions']}")
            
            lines.append("# HELP fraud_error_total Total number of errors")
            lines.append("# TYPE fraud_error_total counter")
            lines.append(f"fraud_error_total {metrics['total_errors']}")
            
            lines.append("# HELP fraud_latency_ms Average prediction latency in milliseconds")
            lines.append("# TYPE fraud_latency_ms gauge")
            lines.append(f"fraud_latency_ms {metrics['average_latency_ms']}")
            
            lines.append("# HELP fraud_error_rate Error rate")
            lines.append("# TYPE fraud_error_rate gauge")
            lines.append(f"fraud_error_rate {metrics['error_rate']}")
            
            return '\n'.join(lines)
        
        except Exception as e:
            logger.error(f"Failed to export Prometheus format: {e}")
            return ""