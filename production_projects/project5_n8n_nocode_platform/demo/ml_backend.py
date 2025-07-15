#!/usr/bin/env python3
"""
ML Backend for n8n MLOps Platform Demo
Real ML processing with scikit-learn for fraud detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import sys
import os
from datetime import datetime

class FraudDetectionPipeline:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.metrics = {}
        
    def load_data(self, file_path='data/sample_data.csv'):
        """Load the fraud detection dataset"""
        try:
            if os.path.exists(file_path):
                self.data = pd.read_csv(file_path)
            else:
                # Create sample data if file doesn't exist
                self.data = self._create_sample_data()
            
            print(f"✅ Data loaded successfully: {len(self.data)} records")
            return {
                'status': 'success',
                'records': len(self.data),
                'columns': list(self.data.columns),
                'fraud_count': int(self.data['is_fraud'].sum()),
                'legitimate_count': int(len(self.data) - self.data['is_fraud'].sum())
            }
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _create_sample_data(self):
        """Create sample fraud detection data"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'user_id': range(1001, 1001 + n_samples),
            'age': np.random.normal(35, 12, n_samples).astype(int),
            'income': np.random.normal(65000, 20000, n_samples).astype(int),
            'credit_score': np.random.normal(700, 80, n_samples).astype(int),
            'transaction_amount': np.random.lognormal(4, 1.5, n_samples),
            'time_of_day': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(1, 8, n_samples),
            'location_risk': np.random.uniform(0, 1, n_samples)
        }
        
        # Create merchant categories
        categories = ['grocery', 'restaurant', 'gas_station', 'electronics', 'pharmacy', 'online', 'department_store']
        data['merchant_category'] = np.random.choice(categories, n_samples)
        
        # Create fraud labels (higher probability for certain conditions)
        fraud_prob = (
            (data['transaction_amount'] > 1000) * 0.3 +
            (data['location_risk'] > 0.7) * 0.4 +
            (data['time_of_day'] > 22) * 0.2 +
            np.random.uniform(0, 0.1, n_samples)
        )
        data['is_fraud'] = (fraud_prob > 0.5).astype(int)
        
        return pd.DataFrame(data)
    
    def validate_data(self):
        """Validate data quality and schema"""
        try:
            validation_results = {
                'status': 'success',
                'missing_values': int(self.data.isnull().sum().sum()),
                'duplicate_rows': int(self.data.duplicated().sum()),
                'data_types': dict(self.data.dtypes.astype(str)),
                'summary_stats': {}
            }
            
            # Generate summary statistics
            for col in self.data.select_dtypes(include=[np.number]).columns:
                validation_results['summary_stats'][col] = {
                    'mean': float(self.data[col].mean()),
                    'std': float(self.data[col].std()),
                    'min': float(self.data[col].min()),
                    'max': float(self.data[col].max())
                }
            
            print("✅ Data validation completed")
            return validation_results
            
        except Exception as e:
            print(f"❌ Error validating data: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def engineer_features(self):
        """Engineer features for the model"""
        try:
            # Separate features and target
            feature_columns = [col for col in self.data.columns if col not in ['user_id', 'is_fraud']]
            X = self.data[feature_columns].copy()
            y = self.data['is_fraud'].copy()
            
            # Encode categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            
            # Scale numerical features
            numerical_columns = X.select_dtypes(include=[np.number]).columns
            X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
            
            self.X = X
            self.y = y
            self.feature_names = feature_columns
            
            print("✅ Feature engineering completed")
            return {
                'status': 'success',
                'features_engineered': len(feature_columns),
                'categorical_encoded': len(categorical_columns),
                'numerical_scaled': len(numerical_columns),
                'feature_names': feature_columns
            }
            
        except Exception as e:
            print(f"❌ Error in feature engineering: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train the fraud detection model"""
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
            )
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                class_weight='balanced'
            )
            
            print("🧠 Training Random Forest model...")
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            self.metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred)),
                'recall': float(recall_score(y_test, y_pred)),
                'f1_score': float(f1_score(y_test, y_pred)),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(
                    self.feature_names,
                    self.model.feature_importances_.tolist()
                ))
            
            # Store test data for later use
            self.X_test = X_test
            self.y_test = y_test
            self.y_pred = y_pred
            
            print("✅ Model training completed")
            return {
                'status': 'success',
                'model_type': 'Random Forest',
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'metrics': self.metrics,
                'feature_importance': self.feature_importance
            }
            
        except Exception as e:
            print(f"❌ Error training model: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        try:
            if self.model is None:
                return {'status': 'error', 'message': 'Model not trained yet'}
            
            # Additional evaluation metrics
            evaluation_results = {
                'status': 'success',
                'metrics': self.metrics,
                'feature_importance': self.feature_importance,
                'model_info': {
                    'algorithm': 'Random Forest',
                    'n_estimators': self.model.n_estimators,
                    'max_depth': self.model.max_depth,
                    'n_features': self.model.n_features_in_
                }
            }
            
            # Top 5 most important features
            if self.feature_importance:
                sorted_features = sorted(
                    self.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                evaluation_results['top_features'] = sorted_features[:5]
            
            print("✅ Model evaluation completed")
            return evaluation_results
            
        except Exception as e:
            print(f"❌ Error evaluating model: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, transaction_data):
        """Make fraud prediction for new transaction"""
        try:
            if self.model is None:
                return {'status': 'error', 'message': 'Model not trained yet'}
            
            # Convert input to DataFrame
            if isinstance(transaction_data, dict):
                df = pd.DataFrame([transaction_data])
            else:
                df = pd.DataFrame(transaction_data)
            
            # Apply same preprocessing
            for col in self.label_encoders:
                if col in df.columns:
                    df[col] = self.label_encoders[col].transform(df[col])
            
            # Scale numerical features
            numerical_columns = df.select_dtypes(include=[np.number]).columns
            df[numerical_columns] = self.scaler.transform(df[numerical_columns])
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0, 1]
            
            return {
                'status': 'success',
                'prediction': int(prediction),
                'fraud_probability': float(probability),
                'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
            }
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def run_full_pipeline(self):
        """Run the complete ML pipeline"""
        print("🚀 Starting full ML pipeline...")
        
        results = {}
        
        # Step 1: Load data
        print("\n📊 Step 1: Loading data...")
        results['load_data'] = self.load_data()
        if results['load_data']['status'] == 'error':
            return results
        
        # Step 2: Validate data
        print("\n✅ Step 2: Validating data...")
        results['validate_data'] = self.validate_data()
        if results['validate_data']['status'] == 'error':
            return results
        
        # Step 3: Engineer features
        print("\n🔧 Step 3: Engineering features...")
        results['engineer_features'] = self.engineer_features()
        if results['engineer_features']['status'] == 'error':
            return results
        
        # Step 4: Train model
        print("\n🧠 Step 4: Training model...")
        results['train_model'] = self.train_model()
        if results['train_model']['status'] == 'error':
            return results
        
        # Step 5: Evaluate model
        print("\n📈 Step 5: Evaluating model...")
        results['evaluate_model'] = self.evaluate_model()
        
        print("\n🎉 Full pipeline completed successfully!")
        return results

def main():
    """Main function for CLI usage"""
    pipeline = FraudDetectionPipeline()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'run':
            results = pipeline.run_full_pipeline()
            print("\n" + "="*50)
            print("FINAL RESULTS:")
            print("="*50)
            
            if 'train_model' in results and 'metrics' in results['train_model']:
                metrics = results['train_model']['metrics']
                print(f"Accuracy:  {metrics['accuracy']:.3f}")
                print(f"Precision: {metrics['precision']:.3f}")
                print(f"Recall:    {metrics['recall']:.3f}")
                print(f"F1-Score:  {metrics['f1_score']:.3f}")
            
            if 'evaluate_model' in results and 'top_features' in results['evaluate_model']:
                print("\nTop Features:")
                for feature, importance in results['evaluate_model']['top_features']:
                    print(f"  {feature}: {importance:.3f}")
        
        elif command == 'predict':
            # Example prediction
            sample_transaction = {
                'age': 35,
                'income': 75000,
                'credit_score': 720,
                'transaction_amount': 2500.0,
                'merchant_category': 'online',
                'time_of_day': 2,
                'day_of_week': 7,
                'location_risk': 0.8
            }
            
            # First train the model
            pipeline.run_full_pipeline()
            
            # Then make prediction
            prediction = pipeline.predict(sample_transaction)
            print(f"Prediction: {prediction}")
        
        else:
            print("Usage: python ml_backend.py [run|predict]")
    
    else:
        # Interactive mode
        print("🔄 n8n MLOps Platform - ML Backend")
        print("Starting interactive fraud detection pipeline...")
        results = pipeline.run_full_pipeline()
        
        # Export results to JSON for web interface
        with open('pipeline_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n💾 Results saved to pipeline_results.json")

if __name__ == "__main__":
    main()