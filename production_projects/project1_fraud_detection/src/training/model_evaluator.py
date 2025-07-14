"""
Model Evaluation Module for Fraud Detection
Comprehensive evaluation metrics and analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation for fraud detection"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Derived metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
        
        # False positive and negative rates
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Probability-based metrics (if probabilities provided)
        if y_prob is not None:
            metrics['auc_score'] = roc_auc_score(y_true, y_prob)
            metrics['average_precision'] = average_precision_score(y_true, y_prob)
            
            # Calculate metrics at different thresholds
            optimal_threshold = self.find_optimal_threshold(y_true, y_prob)
            metrics['optimal_threshold'] = optimal_threshold
            
            # Recalculate with optimal threshold
            y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
            metrics['precision_optimal'] = precision_score(y_true, y_pred_optimal, zero_division=0)
            metrics['recall_optimal'] = recall_score(y_true, y_pred_optimal, zero_division=0)
            metrics['f1_optimal'] = f1_score(y_true, y_pred_optimal, zero_division=0)
        
        # Business metrics for fraud detection
        metrics.update(self._calculate_business_metrics(y_true, y_pred, y_prob))
        
        return metrics
    
    def _calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_prob: np.ndarray = None) -> Dict[str, float]:
        """Calculate business-relevant metrics for fraud detection"""
        
        business_metrics = {}
        
        # Assume average transaction value and cost parameters
        avg_transaction_value = 100.0  # $100 average transaction
        investigation_cost = 50.0     # $50 cost to investigate a flagged transaction
        fraud_loss_rate = 0.8         # 80% of fraud amount is lost if not caught
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate financial impact
        # Savings from catching fraud
        fraud_prevented = tp * avg_transaction_value * fraud_loss_rate
        
        # Cost of investigations
        investigation_costs = (tp + fp) * investigation_cost
        
        # Loss from missed fraud
        fraud_losses = fn * avg_transaction_value * fraud_loss_rate
        
        # Net benefit
        net_benefit = fraud_prevented - investigation_costs - fraud_losses
        
        business_metrics['fraud_prevented_value'] = fraud_prevented
        business_metrics['investigation_costs'] = investigation_costs
        business_metrics['fraud_losses'] = fraud_losses
        business_metrics['net_benefit'] = net_benefit
        
        # Efficiency metrics
        total_transactions = len(y_true)
        business_metrics['investigation_rate'] = (tp + fp) / total_transactions
        business_metrics['fraud_detection_rate'] = tp / max(1, tp + fn)  # Recall
        business_metrics['precision_investigation'] = tp / max(1, tp + fp)  # Precision
        
        # Cost-benefit ratio
        business_metrics['benefit_cost_ratio'] = fraud_prevented / max(1, investigation_costs)
        
        return business_metrics
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray, 
                              metric: str = 'f1') -> float:
        """
        Find optimal threshold for binary classification
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
            
        Returns:
            Optimal threshold value
        """
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred_thresh = (y_prob >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred_thresh, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred_thresh, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred_thresh, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred_thresh)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def evaluate_threshold_impact(self, y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
        """
        Evaluate model performance across different thresholds
        
        Returns:
            DataFrame with metrics for each threshold
        """
        
        thresholds = np.arange(0.1, 0.95, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            
            # Business metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            investigation_rate = (tp + fp) / len(y_true)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'investigation_rate': investigation_rate,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn
            })
        
        return pd.DataFrame(results)
    
    def calculate_model_stability(self, y_true_list: List[np.ndarray], 
                                 y_prob_list: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate model stability across multiple evaluation sets
        
        Args:
            y_true_list: List of true label arrays
            y_prob_list: List of predicted probability arrays
            
        Returns:
            Stability metrics
        """
        
        auc_scores = []
        precision_scores = []
        recall_scores = []
        
        for y_true, y_prob in zip(y_true_list, y_prob_list):
            y_pred = (y_prob >= self.threshold).astype(int)
            
            auc_scores.append(roc_auc_score(y_true, y_prob))
            precision_scores.append(precision_score(y_true, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_true, y_pred, zero_division=0))
        
        return {
            'auc_mean': np.mean(auc_scores),
            'auc_std': np.std(auc_scores),
            'auc_cv': np.std(auc_scores) / np.mean(auc_scores) if np.mean(auc_scores) > 0 else 0,
            'precision_mean': np.mean(precision_scores),
            'precision_std': np.std(precision_scores),
            'recall_mean': np.mean(recall_scores),
            'recall_std': np.std(recall_scores),
            'n_evaluations': len(auc_scores)
        }
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_prob: np.ndarray = None, model_name: str = "Model") -> str:
        """
        Generate a comprehensive text evaluation report
        
        Returns:
            Formatted evaluation report string
        """
        
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        report = f"\n{'='*60}\n"
        report += f"FRAUD DETECTION MODEL EVALUATION REPORT\n"
        report += f"Model: {model_name}\n"
        report += f"{'='*60}\n\n"
        
        # Basic Metrics
        report += "CLASSIFICATION METRICS:\n"
        report += f"  Accuracy:     {metrics['accuracy']:.4f}\n"
        report += f"  Precision:    {metrics['precision']:.4f}\n"
        report += f"  Recall:       {metrics['recall']:.4f}\n"
        report += f"  F1-Score:     {metrics['f1_score']:.4f}\n"
        report += f"  Specificity:  {metrics['specificity']:.4f}\n\n"
        
        # Probability-based metrics
        if y_prob is not None:
            report += "PROBABILITY-BASED METRICS:\n"
            report += f"  AUC-ROC:      {metrics['auc_score']:.4f}\n"
            report += f"  AUC-PR:       {metrics['average_precision']:.4f}\n"
            report += f"  Optimal Threshold: {metrics['optimal_threshold']:.3f}\n\n"
        
        # Confusion Matrix
        report += "CONFUSION MATRIX:\n"
        report += f"  True Negatives:  {metrics['true_negatives']:>6}\n"
        report += f"  False Positives: {metrics['false_positives']:>6}\n"
        report += f"  False Negatives: {metrics['false_negatives']:>6}\n"
        report += f"  True Positives:  {metrics['true_positives']:>6}\n\n"
        
        # Business Impact
        report += "BUSINESS IMPACT:\n"
        report += f"  Investigation Rate:     {metrics['investigation_rate']:.4f}\n"
        report += f"  Fraud Detection Rate:   {metrics['fraud_detection_rate']:.4f}\n"
        report += f"  Net Benefit:           ${metrics['net_benefit']:,.2f}\n"
        report += f"  Benefit-Cost Ratio:     {metrics['benefit_cost_ratio']:.2f}\n\n"
        
        # Model Performance Summary
        if y_prob is not None and metrics['auc_score'] >= 0.9:
            performance = "EXCELLENT"
        elif y_prob is not None and metrics['auc_score'] >= 0.8:
            performance = "GOOD"
        elif y_prob is not None and metrics['auc_score'] >= 0.7:
            performance = "FAIR"
        else:
            performance = "NEEDS IMPROVEMENT"
        
        report += f"OVERALL PERFORMANCE: {performance}\n"
        report += f"{'='*60}\n"
        
        return report