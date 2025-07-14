#!/usr/bin/env python3
"""
Simple test script for the fraud detection API
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from api.models import TransactionRequest
from inference.predictor import FraudPredictor
from training.model_evaluator import ModelEvaluator
from training.data_processor import FraudDataProcessor

async def test_fraud_predictor():
    """Test the fraud predictor functionality"""
    print("🧪 Testing Fraud Detection System...")
    
    try:
        # Initialize predictor
        print("\n1. 🤖 Initializing FraudPredictor...")
        predictor = FraudPredictor()
        
        # Load model (will train a new one if none exists)
        print("\n2. 📥 Loading model...")
        await predictor.load_model(source='local')
        
        # Create test transaction
        print("\n3. 💳 Creating test transaction...")
        test_transaction = TransactionRequest(
            transaction_id="test_123",
            amount=150.75,
            merchant="Amazon.com",
            merchant_category="online",
            hour=14,
            day_of_week=2,
            user_age=35,
            account_age_days=456,
            user_id="user_abc123"
        )
        
        # Make prediction
        print("\n4. 🔮 Making fraud prediction...")
        result = await predictor.predict(test_transaction)
        
        # Display results
        print("\n🎯 Prediction Results:")
        print(f"   Transaction ID: {result['prediction_id']}")
        print(f"   Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Model Version: {result['model_version']}")
        print(f"   Latency: {result['latency_ms']:.2f}ms")
        
        # Test multiple predictions
        print("\n5. 🔄 Testing multiple predictions...")
        test_cases = [
            # Low risk transaction
            TransactionRequest(
                amount=25.50,
                merchant="Starbucks",
                merchant_category="restaurant",
                hour=9,
                day_of_week=1,
                user_age=28,
                account_age_days=800
            ),
            # High risk transaction
            TransactionRequest(
                amount=2500.00,
                merchant="Unknown Shop",
                merchant_category="online",
                hour=3,
                day_of_week=6,
                user_age=22,
                account_age_days=15
            ),
            # Medium risk transaction
            TransactionRequest(
                amount=500.00,
                merchant="Target",
                merchant_category="retail",
                hour=19,
                day_of_week=5,
                user_age=45,
                account_age_days=200
            )
        ]
        
        for i, transaction in enumerate(test_cases, 1):
            result = await predictor.predict(transaction)
            print(f"   Test {i}: ${transaction.amount:>8.2f} at {transaction.hour:02d}:00 -> "
                  f"Risk: {result['risk_level']:>6s} (P={result['fraud_probability']:.3f})")
        
        # Get model info
        print("\n6. 📊 Model Information:")
        model_info = predictor.get_model_info()
        if model_info['status'] == 'loaded':
            print(f"   Model Type: {model_info.get('model_type', 'Unknown')}")
            print(f"   Version: {model_info['version']}")
            print(f"   Features: {model_info['feature_count']}")
            print(f"   Predictions Made: {model_info['prediction_count']}")
            if model_info.get('auc_score'):
                print(f"   AUC Score: {model_info['auc_score']:.4f}")
        
        # Health check
        print("\n7. 🏥 Health Check:")
        health = await predictor.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Model Loaded: {health['model_loaded']}")
        print(f"   Uptime: {health['uptime_seconds']:.1f}s")
        
        # Cleanup
        print("\n8. 🧹 Cleanup...")
        await predictor.cleanup()
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_data_processor():
    """Test the data processor functionality"""
    print("\n📊 Testing Data Processor...")
    
    try:
        # Initialize processor
        processor = FraudDataProcessor()
        
        # Generate synthetic data
        print("   Generating synthetic data...")
        df = processor.generate_synthetic_data(n_samples=1000)
        print(f"   Generated {len(df)} samples with {df['is_fraud'].sum()} fraud cases")
        
        # Preprocess data
        print("   Preprocessing features...")
        df_processed = processor.preprocess_features(df, is_training=True)
        print(f"   Processed data shape: {df_processed.shape}")
        print(f"   Feature columns: {len(processor.get_feature_columns())}")
        
        print("   ✅ Data processor test completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Data processor test failed: {e}")
        return False

async def test_model_evaluator():
    """Test the model evaluator functionality"""
    print("\n🎯 Testing Model Evaluator...")
    
    try:
        # Create dummy predictions
        import numpy as np
        
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.4, 0.1, 0.8, 0.2, 0.1, 0.7, 0.9])
        
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_prob)
        
        print(f"   Accuracy: {metrics['accuracy']:.3f}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
        print(f"   F1-Score: {metrics['f1_score']:.3f}")
        print(f"   AUC: {metrics['auc_score']:.3f}")
        
        print("   ✅ Model evaluator test completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Model evaluator test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Fraud Detection System Test Suite")
    print("=" * 50)
    
    # Test individual components
    test_results = []
    
    test_results.append(await test_data_processor())
    test_results.append(await test_model_evaluator())
    test_results.append(await test_fraud_predictor())
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    total_tests = len(test_results)
    passed_tests = sum(test_results)
    
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! The system is ready for production.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)