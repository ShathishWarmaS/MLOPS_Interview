"""
Real-time Bid Optimizer for Video AdTech Platform
Advanced ML-driven bidding strategies for programmatic advertising
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from collections import defaultdict, deque
import math
import random

# ML libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared.config import Config
from shared.logging import setup_logger
from shared.metrics import MetricsCollector
from infrastructure.database import DatabaseClient
from infrastructure.redis_client import RedisClient

logger = setup_logger(__name__)

@dataclass
class BidRequest:
    """Bid request structure"""
    request_id: str
    timestamp: float
    user_id: str
    content_id: str
    ad_slot_id: str
    device_info: Dict[str, Any]
    user_profile: Dict[str, Any]
    content_context: Dict[str, Any]
    ad_slot_properties: Dict[str, Any]
    auction_timeout_ms: int
    floor_price: float
    currency: str = "USD"

@dataclass
class BidResponse:
    """Bid response structure"""
    request_id: str
    bid_price: float
    confidence: float
    strategy_used: str
    expected_ctr: float
    expected_cvr: float
    expected_revenue: float
    bid_factors: Dict[str, float]
    processing_time_ms: float

@dataclass
class AdPerformanceData:
    """Ad performance tracking data"""
    ad_id: str
    campaign_id: str
    user_id: str
    content_id: str
    bid_price: float
    won_auction: bool
    impression_delivered: bool
    click_occurred: bool
    conversion_occurred: bool
    revenue_generated: float
    timestamp: float

@dataclass
class BidStrategy:
    """Bidding strategy configuration"""
    strategy_name: str
    base_bid: float
    max_bid: float
    target_ctr: float
    target_cvr: float
    target_roas: float  # Return on Ad Spend
    budget_constraint: float
    pacing_factor: float
    risk_tolerance: float

class DeepQNetwork(nn.Module):
    """Deep Q-Network for reinforcement learning bidding"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DeepQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class CTRPredictor(nn.Module):
    """Click-through rate prediction model"""
    
    def __init__(self, feature_size: int):
        super(CTRPredictor, self).__init__()
        
        self.embedding_layers = nn.ModuleDict()
        self.feature_size = feature_size
        
        # Deep neural network for CTR prediction
        self.dnn = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.dnn(x)

class RealTimeBidOptimizer:
    """Advanced real-time bid optimizer"""
    
    def __init__(self, config: Config):
        self.config = config
        self.db_client = DatabaseClient(config)
        self.redis_client = RedisClient(config)
        self.metrics = MetricsCollector()
        
        # ML Models
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Bidding strategies
        self.strategies = {}
        
        # Performance tracking
        self.performance_history = deque(maxlen=10000)
        self.campaign_performance = defaultdict(lambda: {
            'impressions': 0,
            'clicks': 0,
            'conversions': 0,
            'spend': 0.0,
            'revenue': 0.0
        })
        
        # Feature engineering
        self.feature_scaler = StandardScaler()
        self.feature_columns = [
            'user_age', 'user_gender_encoded', 'device_type_encoded',
            'content_category_encoded', 'time_of_day', 'day_of_week',
            'user_engagement_score', 'content_popularity', 'ad_slot_position',
            'historical_ctr', 'historical_cvr', 'competition_level'
        ]
        
        # Initialize components
        asyncio.create_task(self._initialize_models())
        asyncio.create_task(self._load_strategies())
        
        logger.info("Real-time Bid Optimizer initialized")
    
    async def _initialize_models(self):
        """Initialize ML models for bidding"""
        try:
            logger.info("Loading bidding models...")
            
            # CTR prediction model
            self.models['ctr_predictor'] = CTRPredictor(len(self.feature_columns))
            self.models['ctr_predictor'].to(self.device)
            
            # CVR prediction model (similar to CTR)
            self.models['cvr_predictor'] = CTRPredictor(len(self.feature_columns))
            self.models['cvr_predictor'].to(self.device)
            
            # Deep Q-Network for reinforcement learning
            state_size = len(self.feature_columns) + 5  # Additional state features
            action_size = 20  # Discretized bid actions
            self.models['dqn'] = DeepQNetwork(state_size, action_size)
            self.models['dqn'].to(self.device)
            
            # XGBoost for ensemble predictions
            self.models['xgb_ctr'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            self.models['xgb_cvr'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            # Load pre-trained models if available
            await self._load_pretrained_models()
            
            logger.info("✅ Bidding models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def _load_strategies(self):
        """Load bidding strategies"""
        try:
            # Define default strategies
            self.strategies = {
                'conservative': BidStrategy(
                    strategy_name='conservative',
                    base_bid=0.50,
                    max_bid=2.00,
                    target_ctr=0.02,
                    target_cvr=0.05,
                    target_roas=3.0,
                    budget_constraint=1000.0,
                    pacing_factor=1.0,
                    risk_tolerance=0.3
                ),
                'aggressive': BidStrategy(
                    strategy_name='aggressive',
                    base_bid=1.00,
                    max_bid=5.00,
                    target_ctr=0.03,
                    target_cvr=0.08,
                    target_roas=2.5,
                    budget_constraint=2000.0,
                    pacing_factor=1.2,
                    risk_tolerance=0.7
                ),
                'balanced': BidStrategy(
                    strategy_name='balanced',
                    base_bid=0.75,
                    max_bid=3.00,
                    target_ctr=0.025,
                    target_cvr=0.06,
                    target_roas=2.8,
                    budget_constraint=1500.0,
                    pacing_factor=1.1,
                    risk_tolerance=0.5
                ),
                'ml_optimized': BidStrategy(
                    strategy_name='ml_optimized',
                    base_bid=0.80,
                    max_bid=4.00,
                    target_ctr=0.035,
                    target_cvr=0.07,
                    target_roas=3.2,
                    budget_constraint=2500.0,
                    pacing_factor=1.15,
                    risk_tolerance=0.6
                )
            }
            
            logger.info(f"Loaded {len(self.strategies)} bidding strategies")
            
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")
    
    async def optimize_bid(self, 
                          bid_request: BidRequest,
                          strategy_name: str = 'ml_optimized') -> BidResponse:
        """Optimize bid for given request"""
        try:
            start_time = time.time()
            
            # Extract features
            features = await self._extract_features(bid_request)
            
            # Get strategy
            strategy = self.strategies.get(strategy_name, self.strategies['balanced'])
            
            # Predict CTR and CVR
            ctr_prediction = await self._predict_ctr(features)
            cvr_prediction = await self._predict_cvr(features)
            
            # Calculate expected revenue
            expected_revenue = await self._calculate_expected_revenue(
                ctr_prediction, cvr_prediction, bid_request
            )
            
            # Apply bidding strategy
            bid_price = await self._calculate_optimal_bid(
                ctr_prediction, cvr_prediction, expected_revenue, 
                strategy, bid_request
            )
            
            # Apply constraints and adjustments
            bid_price = await self._apply_constraints(
                bid_price, strategy, bid_request
            )
            
            # Calculate confidence
            confidence = await self._calculate_bid_confidence(
                features, ctr_prediction, cvr_prediction, bid_price
            )
            
            # Track bid factors
            bid_factors = {
                'base_bid_factor': 1.0,
                'ctr_factor': float(ctr_prediction / strategy.target_ctr),
                'cvr_factor': float(cvr_prediction / strategy.target_cvr),
                'user_quality_factor': await self._get_user_quality_factor(bid_request.user_id),
                'content_quality_factor': await self._get_content_quality_factor(bid_request.content_id),
                'competition_factor': await self._get_competition_factor(bid_request),
                'time_factor': await self._get_time_factor(bid_request.timestamp),
                'budget_pacing_factor': await self._get_pacing_factor(strategy)
            }
            
            processing_time = (time.time() - start_time) * 1000
            
            response = BidResponse(
                request_id=bid_request.request_id,
                bid_price=float(bid_price),
                confidence=float(confidence),
                strategy_used=strategy_name,
                expected_ctr=float(ctr_prediction),
                expected_cvr=float(cvr_prediction),
                expected_revenue=float(expected_revenue),
                bid_factors=bid_factors,
                processing_time_ms=processing_time
            )
            
            # Track metrics
            self.metrics.record_histogram('bid_optimization_latency_ms', processing_time)
            self.metrics.increment_counter('bid_requests_processed')
            
            # Store bid decision
            await self._store_bid_decision(bid_request, response)
            
            logger.debug(f"Bid optimized: {bid_request.request_id} -> ${bid_price:.3f}")
            
            return response
            
        except Exception as e:
            logger.error(f"Bid optimization failed: {e}")
            self.metrics.increment_counter('bid_optimization_errors')
            
            # Return conservative fallback bid
            return BidResponse(
                request_id=bid_request.request_id,
                bid_price=self.strategies['conservative'].base_bid,
                confidence=0.5,
                strategy_used='fallback',
                expected_ctr=0.02,
                expected_cvr=0.05,
                expected_revenue=0.10,
                bid_factors={'fallback': 1.0},
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _extract_features(self, bid_request: BidRequest) -> np.ndarray:
        """Extract features for ML models"""
        try:
            features = {}
            
            # User features
            user_profile = bid_request.user_profile
            features['user_age'] = user_profile.get('age', 30)
            features['user_gender_encoded'] = self._encode_gender(user_profile.get('gender'))
            
            # Device features
            device_info = bid_request.device_info
            features['device_type_encoded'] = self._encode_device_type(device_info.get('type'))
            
            # Content features
            content_context = bid_request.content_context
            features['content_category_encoded'] = self._encode_content_category(
                content_context.get('category')
            )
            features['content_popularity'] = content_context.get('popularity_score', 0.5)
            
            # Temporal features
            dt = datetime.fromtimestamp(bid_request.timestamp)
            features['time_of_day'] = dt.hour
            features['day_of_week'] = dt.weekday()
            
            # Ad slot features
            ad_slot = bid_request.ad_slot_properties
            features['ad_slot_position'] = ad_slot.get('position', 1)
            
            # Historical features
            features['user_engagement_score'] = await self._get_user_engagement_score(
                bid_request.user_id
            )
            features['historical_ctr'] = await self._get_historical_ctr(
                bid_request.user_id, bid_request.content_id
            )
            features['historical_cvr'] = await self._get_historical_cvr(
                bid_request.user_id, bid_request.content_id
            )
            
            # Competition features
            features['competition_level'] = await self._estimate_competition_level(bid_request)
            
            # Convert to array
            feature_array = np.array([features[col] for col in self.feature_columns])
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return default features
            return np.zeros(len(self.feature_columns))
    
    async def _predict_ctr(self, features: np.ndarray) -> float:
        """Predict click-through rate"""
        try:
            # Normalize features
            features_normalized = self.feature_scaler.transform(features.reshape(1, -1))
            
            # Neural network prediction
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_normalized).to(self.device)
                nn_prediction = self.models['ctr_predictor'](features_tensor).item()
            
            # XGBoost prediction (if trained)
            xgb_prediction = 0.025  # Default
            try:
                xgb_prediction = self.models['xgb_ctr'].predict(features_normalized)[0]
            except Exception:
                pass
            
            # Ensemble prediction
            ctr = 0.7 * nn_prediction + 0.3 * xgb_prediction
            
            # Apply bounds
            ctr = np.clip(ctr, 0.001, 0.2)
            
            return float(ctr)
            
        except Exception as e:
            logger.error(f"CTR prediction failed: {e}")
            return 0.025  # Default CTR
    
    async def _predict_cvr(self, features: np.ndarray) -> float:
        """Predict conversion rate"""
        try:
            # Similar to CTR prediction
            features_normalized = self.feature_scaler.transform(features.reshape(1, -1))
            
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features_normalized).to(self.device)
                nn_prediction = self.models['cvr_predictor'](features_tensor).item()
            
            xgb_prediction = 0.05  # Default
            try:
                xgb_prediction = self.models['xgb_cvr'].predict(features_normalized)[0]
            except Exception:
                pass
            
            cvr = 0.7 * nn_prediction + 0.3 * xgb_prediction
            cvr = np.clip(cvr, 0.001, 0.3)
            
            return float(cvr)
            
        except Exception as e:
            logger.error(f"CVR prediction failed: {e}")
            return 0.05  # Default CVR
    
    async def _calculate_expected_revenue(self, 
                                        ctr: float, 
                                        cvr: float, 
                                        bid_request: BidRequest) -> float:
        """Calculate expected revenue from ad"""
        try:
            # Get average order value for this user/content combination
            aov = await self._get_average_order_value(
                bid_request.user_id, bid_request.content_id
            )
            
            # Commission rate (what we earn from conversion)
            commission_rate = 0.1  # 10% commission
            
            # Expected revenue = P(click) * P(conversion|click) * AOV * commission
            expected_revenue = ctr * cvr * aov * commission_rate
            
            return float(expected_revenue)
            
        except Exception as e:
            logger.error(f"Revenue calculation failed: {e}")
            return 0.10  # Default expected revenue
    
    async def _calculate_optimal_bid(self, 
                                   ctr: float, 
                                   cvr: float, 
                                   expected_revenue: float,
                                   strategy: BidStrategy,
                                   bid_request: BidRequest) -> float:
        """Calculate optimal bid price"""
        try:
            # Method 1: Revenue-based bidding
            revenue_bid = expected_revenue * 0.8  # Bid 80% of expected revenue
            
            # Method 2: Strategy-based bidding
            ctr_multiplier = ctr / strategy.target_ctr
            cvr_multiplier = cvr / strategy.target_cvr
            strategy_bid = strategy.base_bid * ctr_multiplier * cvr_multiplier
            
            # Method 3: Market-based bidding
            market_price = await self._estimate_market_price(bid_request)
            market_bid = market_price * strategy.risk_tolerance
            
            # Method 4: RL-based bidding
            rl_bid = await self._get_rl_bid(bid_request, ctr, cvr)
            
            # Ensemble of methods
            weights = [0.3, 0.3, 0.2, 0.2]
            bids = [revenue_bid, strategy_bid, market_bid, rl_bid]
            
            optimal_bid = sum(w * b for w, b in zip(weights, bids))
            
            # Apply strategy constraints
            optimal_bid = np.clip(optimal_bid, strategy.base_bid * 0.5, strategy.max_bid)
            
            return float(optimal_bid)
            
        except Exception as e:
            logger.error(f"Bid calculation failed: {e}")
            return strategy.base_bid
    
    async def _apply_constraints(self, 
                               bid_price: float, 
                               strategy: BidStrategy,
                               bid_request: BidRequest) -> float:
        """Apply constraints and adjustments to bid"""
        try:
            # Budget pacing
            pacing_factor = await self._get_pacing_factor(strategy)
            bid_price *= pacing_factor
            
            # Competition adjustment
            competition_factor = await self._get_competition_factor(bid_request)
            bid_price *= competition_factor
            
            # Quality adjustment
            quality_factor = await self._get_quality_adjustment(bid_request)
            bid_price *= quality_factor
            
            # Floor price constraint
            bid_price = max(bid_price, bid_request.floor_price)
            
            # Strategy max bid constraint
            bid_price = min(bid_price, strategy.max_bid)
            
            # Round to reasonable precision
            bid_price = round(bid_price, 3)
            
            return float(bid_price)
            
        except Exception as e:
            logger.error(f"Constraint application failed: {e}")
            return bid_price
    
    async def update_performance(self, performance_data: AdPerformanceData):
        """Update performance tracking with new data"""
        try:
            # Add to performance history
            self.performance_history.append(performance_data)
            
            # Update campaign performance
            campaign_perf = self.campaign_performance[performance_data.campaign_id]
            
            if performance_data.impression_delivered:
                campaign_perf['impressions'] += 1
                campaign_perf['spend'] += performance_data.bid_price
            
            if performance_data.click_occurred:
                campaign_perf['clicks'] += 1
            
            if performance_data.conversion_occurred:
                campaign_perf['conversions'] += 1
                campaign_perf['revenue'] += performance_data.revenue_generated
            
            # Update metrics
            self.metrics.increment_counter('ad_impressions_delivered')
            if performance_data.click_occurred:
                self.metrics.increment_counter('ad_clicks_occurred')
            if performance_data.conversion_occurred:
                self.metrics.increment_counter('ad_conversions_occurred')
            
            # Retrain models if needed
            if len(self.performance_history) % 1000 == 0:
                asyncio.create_task(self._retrain_models())
            
            logger.debug(f"Performance updated for ad {performance_data.ad_id}")
            
        except Exception as e:
            logger.error(f"Performance update failed: {e}")
    
    async def _retrain_models(self):
        """Retrain models with recent performance data"""
        try:
            logger.info("Starting model retraining...")
            
            # Prepare training data
            training_data = await self._prepare_training_data()
            
            if len(training_data) < 100:
                logger.warning("Insufficient data for retraining")
                return
            
            # Retrain XGBoost models
            await self._retrain_xgboost_models(training_data)
            
            # Update neural networks
            await self._update_neural_networks(training_data)
            
            logger.info("Model retraining completed")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    # Helper methods for feature engineering and calculations
    
    def _encode_gender(self, gender: str) -> float:
        """Encode gender to numeric value"""
        mapping = {'male': 0.0, 'female': 1.0, 'other': 0.5}
        return mapping.get(gender, 0.5)
    
    def _encode_device_type(self, device_type: str) -> float:
        """Encode device type to numeric value"""
        mapping = {'mobile': 0.0, 'tablet': 0.5, 'desktop': 1.0}
        return mapping.get(device_type, 0.0)
    
    def _encode_content_category(self, category: str) -> float:
        """Encode content category to numeric value"""
        categories = ['sports', 'news', 'entertainment', 'tech', 'lifestyle']
        try:
            return float(categories.index(category)) / len(categories)
        except ValueError:
            return 0.5
    
    async def _get_user_engagement_score(self, user_id: str) -> float:
        """Get user engagement score"""
        try:
            # Query from cache or database
            score = await self.redis_client.get(f"user_engagement:{user_id}")
            return float(score) if score else 0.5
        except Exception:
            return 0.5
    
    async def _get_historical_ctr(self, user_id: str, content_id: str) -> float:
        """Get historical CTR for user-content combination"""
        try:
            key = f"historical_ctr:{user_id}:{content_id}"
            ctr = await self.redis_client.get(key)
            return float(ctr) if ctr else 0.025
        except Exception:
            return 0.025
    
    async def _get_historical_cvr(self, user_id: str, content_id: str) -> float:
        """Get historical CVR for user-content combination"""
        try:
            key = f"historical_cvr:{user_id}:{content_id}"
            cvr = await self.redis_client.get(key)
            return float(cvr) if cvr else 0.05
        except Exception:
            return 0.05
    
    # Additional helper methods would continue here...

async def main():
    """Main function for testing"""
    config = Config()
    optimizer = RealTimeBidOptimizer(config)
    
    # Test bid request
    test_request = BidRequest(
        request_id="test_123",
        timestamp=time.time(),
        user_id="user_456",
        content_id="content_789",
        ad_slot_id="slot_001",
        device_info={"type": "mobile"},
        user_profile={"age": 25, "gender": "female"},
        content_context={"category": "entertainment", "popularity_score": 0.8},
        ad_slot_properties={"position": 1},
        auction_timeout_ms=100,
        floor_price=0.10
    )
    
    # Optimize bid
    response = await optimizer.optimize_bid(test_request)
    print(f"Optimized bid: ${response.bid_price:.3f}")
    print(f"Expected CTR: {response.expected_ctr:.4f}")
    print(f"Expected CVR: {response.expected_cvr:.4f}")

if __name__ == "__main__":
    asyncio.run(main())