"""
Advanced Data Drift Detection for MLOps Pipeline
Statistical and ML-based drift detection with multiple algorithms
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import threading
from abc import ABC, abstractmethod

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Statistical libraries
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis"""
    timestamp: float
    feature_name: str
    drift_score: float
    drift_detected: bool
    test_statistic: float
    p_value: Optional[float]
    confidence_level: float
    method: str
    reference_stats: Dict[str, Any]
    current_stats: Dict[str, Any]
    interpretation: str

@dataclass
class PopulationStabilityResult:
    """Population Stability Index result"""
    feature_name: str
    psi_score: float
    stability_level: str  # 'stable', 'slightly_unstable', 'unstable'
    bin_contributions: List[Dict[str, Any]]
    interpretation: str

class DriftDetector(ABC):
    """Abstract base class for drift detectors"""
    
    @abstractmethod
    def detect_drift(self, 
                    reference_data: np.ndarray,
                    current_data: np.ndarray,
                    feature_name: str = "",
                    confidence_level: float = 0.95) -> DriftDetectionResult:
        """Detect drift between reference and current data"""
        pass

class KolmogorovSmirnovDetector(DriftDetector):
    """Kolmogorov-Smirnov test for continuous features"""
    
    def detect_drift(self, 
                    reference_data: np.ndarray,
                    current_data: np.ndarray,
                    feature_name: str = "",
                    confidence_level: float = 0.95) -> DriftDetectionResult:
        
        # Perform KS test
        ks_statistic, p_value = stats.ks_2samp(reference_data, current_data)
        
        # Determine if drift is detected
        alpha = 1 - confidence_level
        drift_detected = p_value < alpha
        
        # Calculate basic statistics
        ref_stats = {
            'mean': float(np.mean(reference_data)),
            'std': float(np.std(reference_data)),
            'median': float(np.median(reference_data)),
            'min': float(np.min(reference_data)),
            'max': float(np.max(reference_data))
        }
        
        curr_stats = {
            'mean': float(np.mean(current_data)),
            'std': float(np.std(current_data)),
            'median': float(np.median(current_data)),
            'min': float(np.min(current_data)),
            'max': float(np.max(current_data))
        }
        
        # Interpretation
        if drift_detected:
            interpretation = f"Significant distribution change detected (p={p_value:.4f} < {alpha})"
        else:
            interpretation = f"No significant distribution change (p={p_value:.4f} >= {alpha})"
        
        return DriftDetectionResult(
            timestamp=time.time(),
            feature_name=feature_name,
            drift_score=ks_statistic,
            drift_detected=drift_detected,
            test_statistic=ks_statistic,
            p_value=p_value,
            confidence_level=confidence_level,
            method="kolmogorov_smirnov",
            reference_stats=ref_stats,
            current_stats=curr_stats,
            interpretation=interpretation
        )

class ChiSquareDetector(DriftDetector):
    """Chi-square test for categorical features"""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
    
    def detect_drift(self, 
                    reference_data: np.ndarray,
                    current_data: np.ndarray,
                    feature_name: str = "",
                    confidence_level: float = 0.95) -> DriftDetectionResult:
        
        # For continuous data, create bins
        if len(np.unique(reference_data)) > self.n_bins:
            # Create bins based on reference data
            bins = np.histogram_bin_edges(reference_data, bins=self.n_bins)
            ref_counts, _ = np.histogram(reference_data, bins=bins)
            curr_counts, _ = np.histogram(current_data, bins=bins)
        else:
            # Categorical data
            all_categories = np.unique(np.concatenate([reference_data, current_data]))
            ref_counts = []
            curr_counts = []
            
            for category in all_categories:
                ref_counts.append(np.sum(reference_data == category))
                curr_counts.append(np.sum(current_data == category))
            
            ref_counts = np.array(ref_counts)
            curr_counts = np.array(curr_counts)
        
        # Add small constant to avoid zero counts
        ref_counts = ref_counts + 1e-6
        curr_counts = curr_counts + 1e-6
        
        # Normalize to probabilities
        ref_probs = ref_counts / np.sum(ref_counts)
        curr_probs = curr_counts / np.sum(curr_counts)
        
        # Chi-square test
        expected = ref_probs * np.sum(curr_counts)
        chi2_statistic = np.sum((curr_counts - expected) ** 2 / expected)
        
        # Degrees of freedom
        dof = len(ref_counts) - 1
        
        # P-value
        p_value = 1 - stats.chi2.cdf(chi2_statistic, dof)
        
        # Determine if drift is detected
        alpha = 1 - confidence_level
        drift_detected = p_value < alpha
        
        # Calculate statistics
        ref_stats = {
            'distribution': ref_probs.tolist(),
            'entropy': float(-np.sum(ref_probs * np.log(ref_probs + 1e-10)))
        }
        
        curr_stats = {
            'distribution': curr_probs.tolist(),
            'entropy': float(-np.sum(curr_probs * np.log(curr_probs + 1e-10)))
        }
        
        # Interpretation
        if drift_detected:
            interpretation = f"Significant categorical distribution change (χ²={chi2_statistic:.4f}, p={p_value:.4f})"
        else:
            interpretation = f"No significant categorical distribution change (χ²={chi2_statistic:.4f}, p={p_value:.4f})"
        
        return DriftDetectionResult(
            timestamp=time.time(),
            feature_name=feature_name,
            drift_score=chi2_statistic,
            drift_detected=drift_detected,
            test_statistic=chi2_statistic,
            p_value=p_value,
            confidence_level=confidence_level,
            method="chi_square",
            reference_stats=ref_stats,
            current_stats=curr_stats,
            interpretation=interpretation
        )

class JensenShannonDetector(DriftDetector):
    """Jensen-Shannon divergence for distribution comparison"""
    
    def __init__(self, n_bins: int = 50):
        self.n_bins = n_bins
        
    def detect_drift(self, 
                    reference_data: np.ndarray,
                    current_data: np.ndarray,
                    feature_name: str = "",
                    confidence_level: float = 0.95) -> DriftDetectionResult:
        
        # Create histograms
        min_val = min(np.min(reference_data), np.min(current_data))
        max_val = max(np.max(reference_data), np.max(current_data))
        bins = np.linspace(min_val, max_val, self.n_bins + 1)
        
        ref_hist, _ = np.histogram(reference_data, bins=bins, density=True)
        curr_hist, _ = np.histogram(current_data, bins=bins, density=True)
        
        # Normalize to probability distributions
        ref_hist = ref_hist / np.sum(ref_hist)
        curr_hist = curr_hist / np.sum(curr_hist)
        
        # Add small constant to avoid log(0)
        ref_hist = ref_hist + 1e-10
        curr_hist = curr_hist + 1e-10
        
        # Calculate Jensen-Shannon divergence
        js_distance = jensenshannon(ref_hist, curr_hist)
        
        # Empirical threshold (can be calibrated)
        threshold = 0.1  # Typical threshold for JS divergence
        drift_detected = js_distance > threshold
        
        # Calculate statistics
        ref_stats = {
            'mean': float(np.mean(reference_data)),
            'std': float(np.std(reference_data)),
            'histogram': ref_hist.tolist()
        }
        
        curr_stats = {
            'mean': float(np.mean(current_data)),
            'std': float(np.std(current_data)),
            'histogram': curr_hist.tolist()
        }
        
        # Interpretation
        if drift_detected:
            interpretation = f"Significant distribution divergence (JS distance={js_distance:.4f} > {threshold})"
        else:
            interpretation = f"No significant distribution divergence (JS distance={js_distance:.4f} <= {threshold})"
        
        return DriftDetectionResult(
            timestamp=time.time(),
            feature_name=feature_name,
            drift_score=js_distance,
            drift_detected=drift_detected,
            test_statistic=js_distance,
            p_value=None,  # JS divergence doesn't provide p-value
            confidence_level=confidence_level,
            method="jensen_shannon",
            reference_stats=ref_stats,
            current_stats=curr_stats,
            interpretation=interpretation
        )

class MMDDetector(DriftDetector):
    """Maximum Mean Discrepancy for high-dimensional drift detection"""
    
    def __init__(self, kernel: str = 'rbf', gamma: float = None):
        self.kernel = kernel
        self.gamma = gamma
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        """RBF kernel computation"""
        # Compute pairwise distances
        XX = np.sum(X**2, axis=1)[:, None]
        YY = np.sum(Y**2, axis=1)[None, :]
        XY = X @ Y.T
        
        distances = XX + YY - 2 * XY
        return np.exp(-gamma * distances)
    
    def detect_drift(self, 
                    reference_data: np.ndarray,
                    current_data: np.ndarray,
                    feature_name: str = "",
                    confidence_level: float = 0.95) -> DriftDetectionResult:
        
        # Ensure 2D arrays
        if reference_data.ndim == 1:
            reference_data = reference_data.reshape(-1, 1)
        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)
        
        # Set gamma if not provided
        if self.gamma is None:
            gamma = 1.0 / reference_data.shape[1]
        else:
            gamma = self.gamma
        
        n_ref = reference_data.shape[0]
        n_curr = current_data.shape[0]
        
        # Compute kernel matrices
        K_ref_ref = self._rbf_kernel(reference_data, reference_data, gamma)
        K_curr_curr = self._rbf_kernel(current_data, current_data, gamma)
        K_ref_curr = self._rbf_kernel(reference_data, current_data, gamma)
        
        # Compute MMD statistic
        mmd_squared = (np.sum(K_ref_ref) / (n_ref * n_ref) + 
                      np.sum(K_curr_curr) / (n_curr * n_curr) - 
                      2 * np.sum(K_ref_curr) / (n_ref * n_curr))
        
        mmd = np.sqrt(max(0, mmd_squared))
        
        # Empirical threshold (can be calibrated through permutation testing)
        threshold = 0.05
        drift_detected = mmd > threshold
        
        # Calculate basic statistics
        ref_stats = {
            'mean': np.mean(reference_data, axis=0).tolist(),
            'std': np.std(reference_data, axis=0).tolist(),
            'shape': reference_data.shape
        }
        
        curr_stats = {
            'mean': np.mean(current_data, axis=0).tolist(),
            'std': np.std(current_data, axis=0).tolist(),
            'shape': current_data.shape
        }
        
        # Interpretation
        if drift_detected:
            interpretation = f"Significant high-dimensional drift (MMD={mmd:.4f} > {threshold})"
        else:
            interpretation = f"No significant high-dimensional drift (MMD={mmd:.4f} <= {threshold})"
        
        return DriftDetectionResult(
            timestamp=time.time(),
            feature_name=feature_name,
            drift_score=mmd,
            drift_detected=drift_detected,
            test_statistic=mmd,
            p_value=None,
            confidence_level=confidence_level,
            method="mmd",
            reference_stats=ref_stats,
            current_stats=curr_stats,
            interpretation=interpretation
        )

class AdversarialDriftDetector(DriftDetector):
    """Adversarial drift detection using classifier approach"""
    
    def __init__(self, model_type: str = 'isolation_forest'):
        self.model_type = model_type
    
    def detect_drift(self, 
                    reference_data: np.ndarray,
                    current_data: np.ndarray,
                    feature_name: str = "",
                    confidence_level: float = 0.95) -> DriftDetectionResult:
        
        # Ensure 2D arrays
        if reference_data.ndim == 1:
            reference_data = reference_data.reshape(-1, 1)
        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)
        
        # Create combined dataset with labels
        X = np.vstack([reference_data, current_data])
        y = np.hstack([np.zeros(len(reference_data)), np.ones(len(current_data))])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        if self.model_type == 'isolation_forest':
            # Use isolation forest for anomaly detection
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(X_train[y_train == 0])  # Train only on reference data
            
            # Predict on test set
            ref_scores = model.decision_function(X_test[y_test == 0])
            curr_scores = model.decision_function(X_test[y_test == 1])
            
            # Use mean difference as drift score
            drift_score = float(np.mean(ref_scores) - np.mean(curr_scores))
            
        else:
            # Use binary classifier approach
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict probabilities
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate AUC as drift score
            try:
                drift_score = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                drift_score = 0.5  # No discrimination ability
        
        # Determine drift (AUC > 0.75 indicates good separation between distributions)
        threshold = 0.75 if self.model_type != 'isolation_forest' else 0.1
        
        if self.model_type == 'isolation_forest':
            drift_detected = abs(drift_score) > threshold
        else:
            drift_detected = drift_score > threshold
        
        # Calculate statistics
        ref_stats = {
            'mean': np.mean(reference_data, axis=0).tolist(),
            'std': np.std(reference_data, axis=0).tolist(),
            'shape': reference_data.shape
        }
        
        curr_stats = {
            'mean': np.mean(current_data, axis=0).tolist(),
            'std': np.std(current_data, axis=0).tolist(),
            'shape': current_data.shape
        }
        
        # Interpretation
        if drift_detected:
            if self.model_type == 'isolation_forest':
                interpretation = f"Anomaly-based drift detected (score difference={drift_score:.4f})"
            else:
                interpretation = f"Classifier-based drift detected (AUC={drift_score:.4f} > {threshold})"
        else:
            if self.model_type == 'isolation_forest':
                interpretation = f"No anomaly-based drift (score difference={drift_score:.4f})"
            else:
                interpretation = f"No classifier-based drift (AUC={drift_score:.4f} <= {threshold})"
        
        return DriftDetectionResult(
            timestamp=time.time(),
            feature_name=feature_name,
            drift_score=drift_score,
            drift_detected=drift_detected,
            test_statistic=drift_score,
            p_value=None,
            confidence_level=confidence_level,
            method=f"adversarial_{self.model_type}",
            reference_stats=ref_stats,
            current_stats=curr_stats,
            interpretation=interpretation
        )

class PopulationStabilityIndex:
    """Population Stability Index (PSI) calculator"""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
    
    def calculate_psi(self, 
                     reference_data: np.ndarray,
                     current_data: np.ndarray,
                     feature_name: str = "") -> PopulationStabilityResult:
        
        # Create bins based on reference data
        if len(np.unique(reference_data)) <= self.n_bins:
            # Categorical data
            categories = np.unique(reference_data)
            ref_counts = [(reference_data == cat).sum() for cat in categories]
            curr_counts = [(current_data == cat).sum() for cat in categories]
        else:
            # Continuous data - create quantile-based bins
            bins = np.histogram_bin_edges(reference_data, bins=self.n_bins)
            ref_counts, _ = np.histogram(reference_data, bins=bins)
            curr_counts, _ = np.histogram(current_data, bins=bins)
        
        # Convert to proportions
        ref_props = np.array(ref_counts) / len(reference_data)
        curr_props = np.array(curr_counts) / len(current_data)
        
        # Add small constant to avoid log(0)
        ref_props = np.where(ref_props == 0, 1e-6, ref_props)
        curr_props = np.where(curr_props == 0, 1e-6, curr_props)
        
        # Calculate PSI
        psi_values = (curr_props - ref_props) * np.log(curr_props / ref_props)
        psi_total = np.sum(psi_values)
        
        # Determine stability level
        if psi_total < 0.1:
            stability_level = "stable"
        elif psi_total < 0.25:
            stability_level = "slightly_unstable"
        else:
            stability_level = "unstable"
        
        # Create bin contributions
        bin_contributions = []
        for i, (ref_prop, curr_prop, psi_val) in enumerate(zip(ref_props, curr_props, psi_values)):
            bin_contributions.append({
                'bin_index': i,
                'reference_proportion': float(ref_prop),
                'current_proportion': float(curr_prop),
                'psi_contribution': float(psi_val)
            })
        
        # Interpretation
        interpretation = f"PSI = {psi_total:.4f} ({stability_level})"
        
        return PopulationStabilityResult(
            feature_name=feature_name,
            psi_score=float(psi_total),
            stability_level=stability_level,
            bin_contributions=bin_contributions,
            interpretation=interpretation
        )

class ComprehensiveDriftDetector:
    """Comprehensive drift detection system with multiple algorithms"""
    
    def __init__(self, 
                 reference_window_size: int = 1000,
                 detection_window_size: int = 100,
                 detection_interval: int = 50):
        
        self.reference_window_size = reference_window_size
        self.detection_window_size = detection_window_size
        self.detection_interval = detection_interval
        
        # Initialize detectors
        self.detectors = {
            'ks_test': KolmogorovSmirnovDetector(),
            'chi_square': ChiSquareDetector(),
            'jensen_shannon': JensenShannonDetector(),
            'mmd': MMDDetector(),
            'adversarial_rf': AdversarialDriftDetector('random_forest'),
            'adversarial_if': AdversarialDriftDetector('isolation_forest')
        }
        
        self.psi_calculator = PopulationStabilityIndex()
        
        # Data storage
        self.reference_data = {}
        self.current_data = {}
        
        # Results storage
        self.drift_results = []
        self.psi_results = []
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        
        # Configuration
        self.feature_types = {}  # 'continuous' or 'categorical'
        self.detection_enabled = True
    
    def set_reference_data(self, data: Dict[str, np.ndarray]):
        """Set reference data for all features"""
        with self.lock:
            self.reference_data = data.copy()
            
            # Auto-detect feature types
            for feature_name, feature_data in data.items():
                if len(np.unique(feature_data)) <= 20:
                    self.feature_types[feature_name] = 'categorical'
                else:
                    self.feature_types[feature_name] = 'continuous'
        
        logger.info(f"Reference data set for {len(data)} features")
    
    def add_current_data(self, data: Dict[str, Any]):
        """Add current data point"""
        with self.lock:
            for feature_name, value in data.items():
                if feature_name not in self.current_data:
                    self.current_data[feature_name] = []
                
                self.current_data[feature_name].append(value)
                
                # Keep only recent data
                if len(self.current_data[feature_name]) > self.detection_window_size:
                    self.current_data[feature_name] = \
                        self.current_data[feature_name][-self.detection_window_size:]
    
    async def detect_drift_all_features(self, confidence_level: float = 0.95) -> Dict[str, List[DriftDetectionResult]]:
        """Detect drift for all features using multiple methods"""
        
        if not self.reference_data or not self.current_data:
            logger.warning("Reference or current data not available")
            return {}
        
        results = {}
        
        # Process each feature
        for feature_name in self.reference_data.keys():
            if feature_name in self.current_data and len(self.current_data[feature_name]) >= 20:
                feature_results = await self._detect_drift_single_feature(
                    feature_name, confidence_level
                )
                results[feature_name] = feature_results
        
        return results
    
    async def _detect_drift_single_feature(self, 
                                         feature_name: str,
                                         confidence_level: float) -> List[DriftDetectionResult]:
        """Detect drift for a single feature using multiple methods"""
        
        ref_data = np.array(self.reference_data[feature_name])
        curr_data = np.array(self.current_data[feature_name])
        
        feature_type = self.feature_types.get(feature_name, 'continuous')
        results = []
        
        # Select appropriate detectors based on feature type
        if feature_type == 'continuous':
            detector_names = ['ks_test', 'jensen_shannon', 'mmd', 'adversarial_rf']
        else:
            detector_names = ['chi_square', 'jensen_shannon', 'adversarial_rf']
        
        # Run detectors in parallel
        tasks = []
        for detector_name in detector_names:
            if detector_name in self.detectors:
                task = asyncio.create_task(
                    self._run_detector_async(
                        self.detectors[detector_name],
                        ref_data, curr_data, feature_name, confidence_level
                    )
                )
                tasks.append(task)
        
        # Wait for all detectors to complete
        detector_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect valid results
        for result in detector_results:
            if isinstance(result, DriftDetectionResult):
                results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Detector failed for {feature_name}: {result}")
        
        # Store results
        with self.lock:
            self.drift_results.extend(results)
            
            # Keep only recent results
            cutoff_time = time.time() - 3600  # Keep last hour
            self.drift_results = [r for r in self.drift_results if r.timestamp > cutoff_time]
        
        return results
    
    async def _run_detector_async(self,
                                detector: DriftDetector,
                                ref_data: np.ndarray,
                                curr_data: np.ndarray,
                                feature_name: str,
                                confidence_level: float) -> DriftDetectionResult:
        """Run detector asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            detector.detect_drift,
            ref_data, curr_data, feature_name, confidence_level
        )
    
    async def calculate_psi_all_features(self) -> Dict[str, PopulationStabilityResult]:
        """Calculate PSI for all features"""
        
        if not self.reference_data or not self.current_data:
            logger.warning("Reference or current data not available")
            return {}
        
        results = {}
        
        for feature_name in self.reference_data.keys():
            if feature_name in self.current_data and len(self.current_data[feature_name]) >= 20:
                ref_data = np.array(self.reference_data[feature_name])
                curr_data = np.array(self.current_data[feature_name])
                
                psi_result = self.psi_calculator.calculate_psi(
                    ref_data, curr_data, feature_name
                )
                results[feature_name] = psi_result
        
        # Store results
        with self.lock:
            self.psi_results.extend(results.values())
            
            # Keep only recent results
            cutoff_time = time.time() - 3600
            self.psi_results = [r for r in self.psi_results 
                              if hasattr(r, 'timestamp') and r.timestamp > cutoff_time]
        
        return results
    
    def get_drift_summary(self, time_window_hours: int = 1) -> Dict[str, Any]:
        """Get summary of drift detection results"""
        
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        with self.lock:
            recent_results = [r for r in self.drift_results if r.timestamp > cutoff_time]
            recent_psi = [r for r in self.psi_results 
                         if hasattr(r, 'timestamp') and r.timestamp > cutoff_time]
        
        # Group by feature
        feature_summary = {}
        
        for result in recent_results:
            feature_name = result.feature_name
            
            if feature_name not in feature_summary:
                feature_summary[feature_name] = {
                    'drift_detected_count': 0,
                    'total_tests': 0,
                    'max_drift_score': 0.0,
                    'methods_used': set(),
                    'latest_result': None
                }
            
            feature_summary[feature_name]['total_tests'] += 1
            if result.drift_detected:
                feature_summary[feature_name]['drift_detected_count'] += 1
            
            feature_summary[feature_name]['max_drift_score'] = max(
                feature_summary[feature_name]['max_drift_score'],
                result.drift_score
            )
            
            feature_summary[feature_name]['methods_used'].add(result.method)
            
            if (feature_summary[feature_name]['latest_result'] is None or
                result.timestamp > feature_summary[feature_name]['latest_result'].timestamp):
                feature_summary[feature_name]['latest_result'] = result
        
        # Convert sets to lists for JSON serialization
        for feature_data in feature_summary.values():
            feature_data['methods_used'] = list(feature_data['methods_used'])
        
        # Overall summary
        total_features = len(feature_summary)
        features_with_drift = sum(1 for f in feature_summary.values() 
                                if f['drift_detected_count'] > 0)
        
        summary = {
            'time_window_hours': time_window_hours,
            'total_features_monitored': total_features,
            'features_with_drift': features_with_drift,
            'drift_rate': features_with_drift / max(total_features, 1),
            'total_drift_tests': len(recent_results),
            'total_psi_calculations': len(recent_psi),
            'feature_details': feature_summary,
            'alert_level': self._determine_alert_level(features_with_drift, total_features)
        }
        
        return summary
    
    def _determine_alert_level(self, features_with_drift: int, total_features: int) -> str:
        """Determine alert level based on drift detection results"""
        if total_features == 0:
            return 'unknown'
        
        drift_rate = features_with_drift / total_features
        
        if drift_rate >= 0.5:
            return 'high'
        elif drift_rate >= 0.2:
            return 'medium'
        elif drift_rate > 0:
            return 'low'
        else:
            return 'none'
    
    def export_drift_report(self, output_path: str = "drift_report.json"):
        """Export comprehensive drift report"""
        
        summary = self.get_drift_summary(24)  # Last 24 hours
        
        # Add detailed results
        with self.lock:
            recent_results = [asdict(r) for r in self.drift_results[-100:]]  # Last 100 results
            recent_psi = [asdict(r) for r in self.psi_results[-50:]]  # Last 50 PSI results
        
        report = {
            'generation_time': datetime.now().isoformat(),
            'summary': summary,
            'detailed_drift_results': recent_results,
            'psi_results': recent_psi,
            'configuration': {
                'reference_window_size': self.reference_window_size,
                'detection_window_size': self.detection_window_size,
                'detection_interval': self.detection_interval,
                'feature_types': self.feature_types
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Drift report exported to {output_path}")
        
        return report

def main():
    """Main function for drift detection testing"""
    
    async def test_drift_detection():
        # Create test data
        np.random.seed(42)
        
        # Reference data (normal distribution)
        ref_data = {
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.uniform(0, 10, 1000),
            'feature_3': np.random.choice(['A', 'B', 'C'], 1000)
        }
        
        # Current data (with drift)
        curr_data_drift = {
            'feature_1': np.random.normal(0.5, 1.2, 200),  # Mean and variance shift
            'feature_2': np.random.uniform(2, 12, 200),    # Range shift
            'feature_3': np.random.choice(['A', 'B', 'C', 'D'], 200, p=[0.1, 0.3, 0.3, 0.3])  # New category
        }
        
        # Initialize detector
        detector = ComprehensiveDriftDetector()
        detector.set_reference_data(ref_data)
        
        # Add current data
        for i in range(len(curr_data_drift['feature_1'])):
            data_point = {
                'feature_1': curr_data_drift['feature_1'][i],
                'feature_2': curr_data_drift['feature_2'][i],
                'feature_3': curr_data_drift['feature_3'][i]
            }
            detector.add_current_data(data_point)
        
        # Detect drift
        drift_results = await detector.detect_drift_all_features()
        
        # Calculate PSI
        psi_results = await detector.calculate_psi_all_features()
        
        # Print results
        print("Drift Detection Results:")
        print("=" * 50)
        
        for feature_name, results in drift_results.items():
            print(f"\nFeature: {feature_name}")
            for result in results:
                print(f"  {result.method}: {result.interpretation}")
        
        print("\nPSI Results:")
        print("=" * 20)
        for feature_name, psi_result in psi_results.items():
            print(f"{feature_name}: {psi_result.interpretation}")
        
        # Generate summary
        summary = detector.get_drift_summary()
        print(f"\nOverall Summary:")
        print(f"Alert Level: {summary['alert_level']}")
        print(f"Features with drift: {summary['features_with_drift']}/{summary['total_features_monitored']}")
        
        # Export report
        detector.export_drift_report("test_drift_report.json")
    
    # Run test
    asyncio.run(test_drift_detection())

if __name__ == "__main__":
    main()