"""
Data Processing Pipeline for Fraud Detection
Handles data generation, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List
import logging
import joblib
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class FraudDataProcessor:
    """Process transaction data for fraud detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_stats = {}
        self.is_fitted = False
        
    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate realistic synthetic transaction data
        
        This creates a dataset that mimics real-world transaction patterns
        with realistic correlations between features and fraud labels.
        """
        
        logger.info(f"Generating {n_samples} synthetic transaction samples")
        
        np.random.seed(42)  # For reproducibility
        
        # Generate base features with realistic distributions
        data = {
            # Transaction amount (log-normal distribution)
            'amount': np.random.lognormal(mean=4, sigma=1.5, size=n_samples),
            
            # Time features
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            
            # Merchant information
            'merchant_category': np.random.choice(
                ['grocery', 'gas', 'restaurant', 'retail', 'online'], 
                n_samples,
                p=[0.25, 0.15, 0.20, 0.25, 0.15]  # Realistic distribution
            ),
            
            # User demographics
            'user_age': np.random.normal(40, 15, n_samples).astype(int),
            'account_age_days': np.random.exponential(365, n_samples).astype(int),
        }
        
        df = pd.DataFrame(data)
        
        # Clean up unrealistic values
        df['user_age'] = np.clip(df['user_age'], 18, 80)
        df['account_age_days'] = np.clip(df['account_age_days'], 1, 3650)  # Max 10 years
        df['amount'] = np.clip(df['amount'], 1, 10000)  # Reasonable range
        
        # Generate realistic fraud labels based on feature patterns
        fraud_probability = self._calculate_fraud_probability(df)
        df['is_fraud'] = np.random.binomial(1, fraud_probability)
        
        logger.info(f"Generated data with {df['is_fraud'].sum()} fraud cases ({df['is_fraud'].mean():.3%} fraud rate)")
        
        return df
    
    def _calculate_fraud_probability(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate fraud probability based on realistic patterns
        
        This function encodes domain knowledge about fraud patterns:
        - Higher amounts are more likely to be fraud
        - Transactions at unusual hours are suspicious
        - Online transactions have higher fraud rates
        - New accounts are riskier
        """
        
        base_prob = 0.02  # 2% base fraud rate
        
        # Amount-based risk (higher amounts = higher risk)
        amount_factor = 1 + np.log1p(df['amount']) / 10
        
        # Time-based risk (night and early morning more risky)
        time_risk = np.where(
            (df['hour'] >= 1) & (df['hour'] <= 5),  # 1 AM to 5 AM
            3.0,  # 3x higher risk
            np.where(
                (df['hour'] >= 22) | (df['hour'] <= 1),  # 10 PM to 1 AM
                2.0,  # 2x higher risk
                1.0   # Normal risk
            )
        )
        
        # Weekend risk (slightly higher on weekends)
        weekend_factor = np.where(df['day_of_week'] >= 5, 1.2, 1.0)
        
        # Merchant category risk
        merchant_risk_map = {
            'online': 2.5,    # Highest risk
            'gas': 1.0,       # Low risk
            'grocery': 0.8,   # Lowest risk
            'restaurant': 1.2,
            'retail': 1.5
        }
        merchant_factor = df['merchant_category'].map(merchant_risk_map)
        
        # Account age risk (newer accounts are riskier)
        account_factor = np.where(
            df['account_age_days'] < 30,
            3.0,  # Very new accounts
            np.where(
                df['account_age_days'] < 90,
                2.0,  # New accounts
                1.0   # Established accounts
            )
        )
        
        # Age-based risk (very young and very old users might be riskier)
        age_factor = np.where(
            (df['user_age'] < 25) | (df['user_age'] > 65),
            1.3,
            1.0
        )
        
        # Combine all factors
        combined_probability = (base_prob * 
                              amount_factor * 
                              time_risk * 
                              weekend_factor * 
                              merchant_factor * 
                              account_factor * 
                              age_factor)
        
        # Cap at 30% to keep realistic
        return np.clip(combined_probability, 0, 0.30)
    
    def preprocess_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Preprocess features for model training/inference
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data (fits transformers) or inference data
            
        Returns:
            Preprocessed DataFrame ready for model
        """
        
        logger.info(f"Preprocessing features for {'training' if is_training else 'inference'}")
        
        df_processed = df.copy()
        
        # Feature engineering
        df_processed = self._engineer_features(df_processed)
        
        # Handle categorical features
        df_processed = self._encode_categorical_features(df_processed, is_training)
        
        # Scale numerical features
        df_processed = self._scale_numerical_features(df_processed, is_training)
        
        if is_training:
            self.is_fitted = True
            logger.info("Data processor fitted successfully")
        
        return df_processed
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        
        df_processed = df.copy()
        
        # Amount-based features
        df_processed['amount_log'] = np.log1p(df_processed['amount'])
        df_processed['amount_bin'] = pd.cut(
            df_processed['amount'], 
            bins=[0, 50, 200, 500, float('inf')], 
            labels=['small', 'medium', 'large', 'very_large']
        )
        
        # Time-based features
        df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
        df_processed['is_night'] = (
            (df_processed['hour'] >= 22) | (df_processed['hour'] <= 6)
        ).astype(int)
        df_processed['is_business_hours'] = (
            (df_processed['hour'] >= 9) & (df_processed['hour'] <= 17) & 
            (df_processed['day_of_week'] < 5)
        ).astype(int)
        
        # Account-based features
        df_processed['account_age_weeks'] = df_processed['account_age_days'] / 7
        df_processed['is_new_account'] = (df_processed['account_age_days'] < 30).astype(int)
        
        # Age-based features
        df_processed['age_group'] = pd.cut(
            df_processed['user_age'],
            bins=[0, 25, 35, 50, 65, 100],
            labels=['young', 'young_adult', 'middle_aged', 'older', 'senior']
        )
        
        # Risk score (composite feature)
        df_processed['risk_score'] = (
            df_processed['is_night'] * 2 +
            df_processed['is_new_account'] * 3 +
            (df_processed['amount'] > 1000).astype(int) * 2
        )
        
        return df_processed
    
    def _encode_categorical_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Encode categorical features"""
        
        categorical_features = ['merchant_category', 'amount_bin', 'age_group']
        
        for feature in categorical_features:
            if feature in df.columns:
                if is_training:
                    # Fit new encoder
                    le = LabelEncoder()
                    df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                    self.label_encoders[feature] = le
                    logger.debug(f"Fitted encoder for {feature}")
                else:
                    # Use existing encoder
                    if feature in self.label_encoders:
                        le = self.label_encoders[feature]
                        # Handle unseen categories
                        known_categories = set(le.classes_)
                        df[f'{feature}_safe'] = df[feature].astype(str).apply(
                            lambda x: x if x in known_categories else 'unknown'
                        )
                        
                        # Add 'unknown' to encoder if not present
                        if 'unknown' not in known_categories:
                            le.classes_ = np.append(le.classes_, 'unknown')
                        
                        df[f'{feature}_encoded'] = le.transform(df[f'{feature}_safe'])
                        df.drop(f'{feature}_safe', axis=1, inplace=True)
                    else:
                        logger.warning(f"Encoder for {feature} not found, using default encoding")
                        df[f'{feature}_encoded'] = 0  # Default value
        
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Scale numerical features"""
        
        numerical_features = [
            'amount_log', 'hour', 'day_of_week', 'user_age', 
            'account_age_days', 'account_age_weeks', 'risk_score'
        ]
        
        # Only scale features that exist in the dataframe
        features_to_scale = [f for f in numerical_features if f in df.columns]
        
        if is_training:
            df[features_to_scale] = self.scaler.fit_transform(df[features_to_scale])
            self.feature_stats = {
                'scaled_features': features_to_scale,
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist()
            }
            logger.debug(f"Fitted scaler for {len(features_to_scale)} features")
        else:
            if hasattr(self.scaler, 'mean_'):
                df[features_to_scale] = self.scaler.transform(df[features_to_scale])
            else:
                logger.warning("Scaler not fitted, skipping scaling")
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get the list of feature columns for model training"""
        return [
            'amount_log', 'hour', 'day_of_week', 'user_age', 'account_age_days',
            'merchant_category_encoded', 'is_weekend', 'is_night', 
            'is_business_hours', 'account_age_weeks', 'is_new_account',
            'amount_bin_encoded', 'age_group_encoded', 'risk_score'
        ]
    
    def save(self, filepath: str):
        """Save the data processor"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Data processor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load the data processor"""
        processor = joblib.load(filepath)
        logger.info(f"Data processor loaded from {filepath}")
        return processor
    
    def get_stats(self) -> Dict:
        """Get processor statistics"""
        return {
            'is_fitted': self.is_fitted,
            'feature_stats': self.feature_stats,
            'label_encoders': list(self.label_encoders.keys()),
            'feature_columns': self.get_feature_columns()
        }