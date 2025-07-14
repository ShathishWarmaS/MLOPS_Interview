"""
ML Security Framework
Advanced security measures for MLOps pipelines including
data privacy, model security, and adversarial protection
"""

import asyncio
import logging
import time
import json
import hashlib
import hmac
import secrets
import base64
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
import uuid
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
from functools import wraps

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# ML security libraries
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatType(Enum):
    """Types of security threats"""
    ADVERSARIAL_ATTACK = "adversarial_attack"
    DATA_POISONING = "data_poisoning"
    MODEL_INVERSION = "model_inversion"
    MEMBERSHIP_INFERENCE = "membership_inference"
    DATA_LEAKAGE = "data_leakage"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALICIOUS_INPUT = "malicious_input"

@dataclass
class SecurityEvent:
    """Security event record"""
    event_id: str
    timestamp: float
    threat_type: ThreatType
    severity: SecurityLevel
    user_id: Optional[str]
    ip_address: Optional[str]
    request_data: Dict[str, Any]
    detection_method: str
    confidence_score: float
    action_taken: str
    details: Dict[str, Any]

@dataclass
class AuthenticationResult:
    """Authentication result"""
    success: bool
    user_id: Optional[str]
    permissions: List[str]
    token: Optional[str]
    expires_at: Optional[datetime]
    error_message: Optional[str]

@dataclass
class AdversarialDetectionResult:
    """Adversarial attack detection result"""
    is_adversarial: bool
    confidence: float
    detection_method: str
    anomaly_score: float
    feature_deviations: Dict[str, float]
    recommended_action: str

class DataPrivacyProtector:
    """Data privacy and anonymization utilities"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.anonymization_mappings = {}
        self.lock = threading.Lock()
    
    def encrypt_sensitive_data(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Encrypt sensitive fields in data"""
        protected_data = data.copy()
        
        for field in sensitive_fields:
            if field in protected_data:
                value = str(protected_data[field])
                encrypted_value = self.cipher.encrypt(value.encode()).decode()
                protected_data[field] = encrypted_value
        
        return protected_data
    
    def decrypt_sensitive_data(self, data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Decrypt sensitive fields in data"""
        decrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in decrypted_data:
                try:
                    encrypted_value = decrypted_data[field].encode()
                    decrypted_value = self.cipher.decrypt(encrypted_value).decode()
                    decrypted_data[field] = decrypted_value
                except Exception as e:
                    logger.warning(f"Failed to decrypt field {field}: {e}")
        
        return decrypted_data
    
    def anonymize_data(self, data: pd.DataFrame, 
                      identifier_columns: List[str],
                      strategy: str = "hash") -> pd.DataFrame:
        """Anonymize personally identifiable information"""
        anonymized_data = data.copy()
        
        for column in identifier_columns:
            if column in anonymized_data.columns:
                if strategy == "hash":
                    anonymized_data[column] = anonymized_data[column].apply(
                        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
                    )
                elif strategy == "mask":
                    anonymized_data[column] = anonymized_data[column].apply(
                        lambda x: "*" * len(str(x))
                    )
                elif strategy == "generalize":
                    anonymized_data[column] = anonymized_data[column].apply(
                        lambda x: self._generalize_value(x, column)
                    )
        
        return anonymized_data
    
    def _generalize_value(self, value: Any, column: str) -> str:
        """Generalize values for k-anonymity"""
        # Simple generalization strategies
        if isinstance(value, (int, float)):
            # Generalize numbers to ranges
            if value < 0:
                return "negative"
            elif value < 10:
                return "0-10"
            elif value < 100:
                return "10-100"
            else:
                return "100+"
        else:
            # Generalize strings by keeping only first character
            return str(value)[0] + "*" * (len(str(value)) - 1)
    
    def apply_differential_privacy(self, data: np.ndarray, 
                                  epsilon: float = 1.0,
                                  sensitivity: float = 1.0) -> np.ndarray:
        """Apply differential privacy using Laplace mechanism"""
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        # Calculate noise scale
        scale = sensitivity / epsilon
        
        # Add Laplace noise
        noise = np.random.laplace(0, scale, data.shape)
        noisy_data = data + noise
        
        return noisy_data

class AuthenticationManager:
    """Authentication and authorization manager"""
    
    def __init__(self, secret_key: str, token_expiry_hours: int = 24):
        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours
        self.user_permissions = {}
        self.active_tokens = set()
        self.failed_attempts = {}
        self.lock = threading.Lock()
        
        # Rate limiting
        self.max_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str = None) -> AuthenticationResult:
        """Authenticate user credentials"""
        try:
            # Check for rate limiting
            if self._is_rate_limited(username, ip_address):
                return AuthenticationResult(
                    success=False,
                    user_id=None,
                    permissions=[],
                    token=None,
                    expires_at=None,
                    error_message="Too many failed attempts. Account temporarily locked."
                )
            
            # Simulate user authentication (in production, use proper authentication)
            is_valid = self._validate_credentials(username, password)
            
            if is_valid:
                # Clear failed attempts
                with self.lock:
                    self.failed_attempts.pop(username, None)
                
                # Generate JWT token
                token = self._generate_jwt_token(username)
                expires_at = datetime.utcnow() + timedelta(hours=self.token_expiry_hours)
                
                # Get user permissions
                permissions = self.user_permissions.get(username, ["read"])
                
                with self.lock:
                    self.active_tokens.add(token)
                
                return AuthenticationResult(
                    success=True,
                    user_id=username,
                    permissions=permissions,
                    token=token,
                    expires_at=expires_at,
                    error_message=None
                )
            else:
                # Record failed attempt
                self._record_failed_attempt(username, ip_address)
                
                return AuthenticationResult(
                    success=False,
                    user_id=None,
                    permissions=[],
                    token=None,
                    expires_at=None,
                    error_message="Invalid credentials"
                )
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                user_id=None,
                permissions=[],
                token=None,
                expires_at=None,
                error_message="Authentication service error"
            )
    
    def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate user credentials (mock implementation)"""
        # In production, this would check against a secure database
        # with properly hashed passwords
        valid_users = {
            "admin": "admin_secure_password",
            "data_scientist": "ds_password",
            "ml_engineer": "mle_password",
            "viewer": "viewer_password"
        }
        
        return valid_users.get(username) == password
    
    def _is_rate_limited(self, username: str, ip_address: str) -> bool:
        """Check if user/IP is rate limited"""
        with self.lock:
            now = datetime.utcnow()
            
            # Check username-based limiting
            if username in self.failed_attempts:
                attempts, last_attempt = self.failed_attempts[username]
                if attempts >= self.max_attempts:
                    if now - last_attempt < self.lockout_duration:
                        return True
                    else:
                        # Reset after lockout period
                        del self.failed_attempts[username]
            
            return False
    
    def _record_failed_attempt(self, username: str, ip_address: str):
        """Record failed authentication attempt"""
        with self.lock:
            now = datetime.utcnow()
            
            if username in self.failed_attempts:
                attempts, _ = self.failed_attempts[username]
                self.failed_attempts[username] = (attempts + 1, now)
            else:
                self.failed_attempts[username] = (1, now)
    
    def _generate_jwt_token(self, username: str) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": username,
            "permissions": self.user_permissions.get(username, ["read"]),
            "issued_at": datetime.utcnow().timestamp(),
            "expires_at": (datetime.utcnow() + timedelta(hours=self.token_expiry_hours)).timestamp()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            with self.lock:
                if token not in self.active_tokens:
                    return None
            
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Check expiration
            if datetime.utcnow().timestamp() > payload["expires_at"]:
                with self.lock:
                    self.active_tokens.discard(token)
                return None
            
            return payload
            
        except jwt.InvalidTokenError:
            return None
    
    def revoke_token(self, token: str):
        """Revoke token"""
        with self.lock:
            self.active_tokens.discard(token)

class AdversarialDetector:
    """Adversarial attack detection system"""
    
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        
        # Statistical baselines
        self.feature_stats = {}
        self._calculate_reference_statistics()
    
    def train(self):
        """Train anomaly detection models"""
        try:
            # Fit scaler and isolation forest
            scaled_data = self.scaler.fit_transform(self.reference_data)
            self.isolation_forest.fit(scaled_data)
            self.is_trained = True
            
            logger.info("Adversarial detector trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train adversarial detector: {e}")
            raise
    
    def _calculate_reference_statistics(self):
        """Calculate reference statistics for features"""
        for column in self.reference_data.columns:
            if self.reference_data[column].dtype in [np.float64, np.int64]:
                self.feature_stats[column] = {
                    "mean": float(self.reference_data[column].mean()),
                    "std": float(self.reference_data[column].std()),
                    "min": float(self.reference_data[column].min()),
                    "max": float(self.reference_data[column].max()),
                    "q25": float(self.reference_data[column].quantile(0.25)),
                    "q75": float(self.reference_data[column].quantile(0.75))
                }
    
    def detect_adversarial_input(self, input_data: Dict[str, Any]) -> AdversarialDetectionResult:
        """Detect adversarial inputs"""
        try:
            if not self.is_trained:
                self.train()
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Statistical anomaly detection
            stat_anomalies = self._detect_statistical_anomalies(input_df)
            
            # Isolation forest detection
            scaled_input = self.scaler.transform(input_df)
            isolation_score = self.isolation_forest.decision_function(scaled_input)[0]
            is_outlier = self.isolation_forest.predict(scaled_input)[0] == -1
            
            # Feature deviation analysis
            feature_deviations = self._calculate_feature_deviations(input_data)
            
            # Combined score
            stat_score = np.mean(list(stat_anomalies.values()))
            combined_score = 0.7 * stat_score + 0.3 * (1 - (isolation_score + 1) / 2)
            
            # Determine if adversarial
            is_adversarial = (combined_score > 0.7) or is_outlier or any(
                score > 3.0 for score in stat_anomalies.values()
            )
            
            # Recommendation
            if is_adversarial:
                if combined_score > 0.9:
                    recommendation = "REJECT - High confidence adversarial input"
                else:
                    recommendation = "FLAG - Potential adversarial input for review"
            else:
                recommendation = "ACCEPT - Input appears benign"
            
            return AdversarialDetectionResult(
                is_adversarial=is_adversarial,
                confidence=float(combined_score),
                detection_method="statistical_isolation_hybrid",
                anomaly_score=float(isolation_score),
                feature_deviations=feature_deviations,
                recommended_action=recommendation
            )
            
        except Exception as e:
            logger.error(f"Adversarial detection failed: {e}")
            return AdversarialDetectionResult(
                is_adversarial=False,
                confidence=0.0,
                detection_method="error",
                anomaly_score=0.0,
                feature_deviations={},
                recommended_action="ERROR - Could not analyze input"
            )
    
    def _detect_statistical_anomalies(self, input_df: pd.DataFrame) -> Dict[str, float]:
        """Detect statistical anomalies in features"""
        anomaly_scores = {}
        
        for column in input_df.columns:
            if column in self.feature_stats:
                value = input_df[column].iloc[0]
                stats = self.feature_stats[column]
                
                # Z-score based anomaly detection
                z_score = abs(value - stats["mean"]) / max(stats["std"], 1e-6)
                
                # Range-based detection
                in_range = stats["min"] <= value <= stats["max"]
                
                # IQR-based detection
                iqr = stats["q75"] - stats["q25"]
                lower_bound = stats["q25"] - 1.5 * iqr
                upper_bound = stats["q75"] + 1.5 * iqr
                in_iqr_range = lower_bound <= value <= upper_bound
                
                # Combined anomaly score
                range_penalty = 0 if in_range else 5.0
                iqr_penalty = 0 if in_iqr_range else 2.0
                
                anomaly_scores[column] = z_score + range_penalty + iqr_penalty
            else:
                anomaly_scores[column] = 0.0
        
        return anomaly_scores
    
    def _calculate_feature_deviations(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate feature deviations from expected ranges"""
        deviations = {}
        
        for feature, value in input_data.items():
            if feature in self.feature_stats:
                stats = self.feature_stats[feature]
                
                # Deviation as percentage of standard deviation
                deviation = abs(value - stats["mean"]) / max(stats["std"], 1e-6)
                deviations[feature] = float(deviation)
            else:
                deviations[feature] = 0.0
        
        return deviations

class MLSecurityFramework:
    """Comprehensive ML security framework"""
    
    def __init__(self, 
                 secret_key: str = None,
                 encryption_key: bytes = None,
                 reference_data: pd.DataFrame = None):
        
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        
        # Components
        self.auth_manager = AuthenticationManager(self.secret_key)
        self.privacy_protector = DataPrivacyProtector(encryption_key)
        self.adversarial_detector = AdversarialDetector(reference_data) if reference_data is not None else None
        
        # Security event logging
        self.security_events = []
        self.lock = threading.Lock()
        
        # Configuration
        self.security_policies = {
            "require_authentication": True,
            "enable_adversarial_detection": True,
            "enable_audit_logging": True,
            "max_request_size": 1024 * 1024,  # 1MB
            "rate_limit_requests_per_minute": 100
        }
        
        # Request tracking
        self.request_history = {}
        
        # Setup user permissions
        self._setup_default_permissions()
    
    def _setup_default_permissions(self):
        """Setup default user permissions"""
        self.auth_manager.user_permissions = {
            "admin": ["read", "write", "admin", "model_deploy"],
            "data_scientist": ["read", "write", "model_train"],
            "ml_engineer": ["read", "model_deploy", "monitoring"],
            "viewer": ["read"]
        }
    
    async def secure_predict(self, 
                           features: Dict[str, Any],
                           token: str = None,
                           user_id: str = None,
                           ip_address: str = None) -> Dict[str, Any]:
        """Secure prediction endpoint with security checks"""
        try:
            start_time = time.time()
            
            # Authentication check
            if self.security_policies["require_authentication"]:
                auth_result = self._verify_authentication(token)
                if not auth_result["success"]:
                    await self._log_security_event(
                        ThreatType.UNAUTHORIZED_ACCESS,
                        SecurityLevel.HIGH,
                        user_id, ip_address, features,
                        "Authentication failed",
                        0.9, "Request rejected"
                    )
                    return {"error": "Authentication required", "code": 401}
                
                user_id = auth_result["user_id"]
                permissions = auth_result["permissions"]
                
                # Check permissions
                if "read" not in permissions:
                    await self._log_security_event(
                        ThreatType.UNAUTHORIZED_ACCESS,
                        SecurityLevel.HIGH,
                        user_id, ip_address, features,
                        "Insufficient permissions",
                        0.9, "Request rejected"
                    )
                    return {"error": "Insufficient permissions", "code": 403}
            
            # Rate limiting
            if not self._check_rate_limit(user_id or ip_address):
                await self._log_security_event(
                    ThreatType.MALICIOUS_INPUT,
                    SecurityLevel.MEDIUM,
                    user_id, ip_address, features,
                    "Rate limit exceeded",
                    0.7, "Request throttled"
                )
                return {"error": "Rate limit exceeded", "code": 429}
            
            # Input validation
            validation_result = self._validate_input(features)
            if not validation_result["valid"]:
                await self._log_security_event(
                    ThreatType.MALICIOUS_INPUT,
                    SecurityLevel.MEDIUM,
                    user_id, ip_address, features,
                    f"Input validation failed: {validation_result['error']}",
                    0.8, "Request rejected"
                )
                return {"error": validation_result["error"], "code": 400}
            
            # Adversarial detection
            if (self.security_policies["enable_adversarial_detection"] and 
                self.adversarial_detector is not None):
                
                detection_result = self.adversarial_detector.detect_adversarial_input(features)
                
                if detection_result.is_adversarial:
                    await self._log_security_event(
                        ThreatType.ADVERSARIAL_ATTACK,
                        SecurityLevel.CRITICAL if detection_result.confidence > 0.9 else SecurityLevel.HIGH,
                        user_id, ip_address, features,
                        f"Adversarial input detected: {detection_result.recommended_action}",
                        detection_result.confidence,
                        "Request blocked" if detection_result.confidence > 0.9 else "Request flagged"
                    )
                    
                    if detection_result.confidence > 0.9:
                        return {
                            "error": "Input rejected by security system",
                            "code": 400,
                            "security_code": "ADVERSARIAL_DETECTED"
                        }
            
            # Data privacy protection
            protected_features = self.privacy_protector.encrypt_sensitive_data(
                features, ["user_id", "email", "phone", "ssn"]
            )
            
            # Simulate prediction (in production, call actual predictor)
            prediction_result = {
                "prediction": 0.75,
                "confidence": 0.85,
                "model_version": "secure_v1.0",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "security_checks_passed": True
            }
            
            # Audit logging
            if self.security_policies["enable_audit_logging"]:
                await self._log_security_event(
                    ThreatType.UNAUTHORIZED_ACCESS,  # Using as general access log
                    SecurityLevel.LOW,
                    user_id, ip_address, {"feature_count": len(features)},
                    "Successful prediction request",
                    1.0, "Request processed"
                )
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Secure prediction failed: {e}")
            await self._log_security_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                SecurityLevel.MEDIUM,
                user_id, ip_address, features,
                f"Prediction service error: {str(e)}",
                0.5, "Request failed"
            )
            return {"error": "Service unavailable", "code": 500}
    
    def _verify_authentication(self, token: str) -> Dict[str, Any]:
        """Verify authentication token"""
        if not token:
            return {"success": False, "error": "No token provided"}
        
        payload = self.auth_manager.verify_token(token)
        if payload is None:
            return {"success": False, "error": "Invalid or expired token"}
        
        return {
            "success": True,
            "user_id": payload["user_id"],
            "permissions": payload["permissions"]
        }
    
    def _check_rate_limit(self, identifier: str) -> bool:
        """Check rate limiting"""
        if not identifier:
            return True
        
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)
        
        with self.lock:
            if identifier not in self.request_history:
                self.request_history[identifier] = []
            
            # Clean old requests
            self.request_history[identifier] = [
                req_time for req_time in self.request_history[identifier]
                if req_time > window_start
            ]
            
            # Check limit
            if len(self.request_history[identifier]) >= self.security_policies["rate_limit_requests_per_minute"]:
                return False
            
            # Add current request
            self.request_history[identifier].append(now)
            return True
    
    def _validate_input(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input features"""
        # Check input size
        input_size = len(json.dumps(features).encode())
        if input_size > self.security_policies["max_request_size"]:
            return {
                "valid": False,
                "error": f"Input size ({input_size} bytes) exceeds limit"
            }
        
        # Check for required fields
        if not features:
            return {"valid": False, "error": "Empty feature set"}
        
        # Check for suspicious patterns
        for key, value in features.items():
            # Check for SQL injection patterns
            if isinstance(value, str) and any(
                pattern in value.lower() 
                for pattern in ["select ", "drop ", "insert ", "delete ", "update "]
            ):
                return {"valid": False, "error": "Suspicious SQL pattern detected"}
            
            # Check for script injection
            if isinstance(value, str) and any(
                pattern in value.lower()
                for pattern in ["<script", "javascript:", "onload=", "onerror="]
            ):
                return {"valid": False, "error": "Suspicious script pattern detected"}
            
            # Check for extremely large values
            if isinstance(value, (int, float)) and abs(value) > 1e10:
                return {"valid": False, "error": f"Feature {key} has extreme value"}
        
        return {"valid": True}
    
    async def _log_security_event(self, 
                                threat_type: ThreatType,
                                severity: SecurityLevel,
                                user_id: Optional[str],
                                ip_address: Optional[str],
                                request_data: Dict[str, Any],
                                detection_method: str,
                                confidence_score: float,
                                action_taken: str):
        """Log security event"""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            threat_type=threat_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            request_data=request_data,
            detection_method=detection_method,
            confidence_score=confidence_score,
            action_taken=action_taken,
            details={}
        )
        
        with self.lock:
            self.security_events.append(event)
        
        # Log to standard logger based on severity
        if severity == SecurityLevel.CRITICAL:
            logger.critical(f"SECURITY ALERT: {threat_type.value} - {action_taken}")
        elif severity == SecurityLevel.HIGH:
            logger.error(f"SECURITY WARNING: {threat_type.value} - {action_taken}")
        else:
            logger.info(f"SECURITY INFO: {threat_type.value} - {action_taken}")
    
    async def get_security_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get security events summary"""
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        with self.lock:
            recent_events = [e for e in self.security_events if e.timestamp > cutoff_time]
        
        # Analyze events
        threat_counts = {}
        severity_counts = {}
        
        for event in recent_events:
            threat_counts[event.threat_type.value] = threat_counts.get(event.threat_type.value, 0) + 1
            severity_counts[event.severity.value] = severity_counts.get(event.severity.value, 0) + 1
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(recent_events)
        
        summary = {
            "time_window_hours": time_window_hours,
            "total_events": len(recent_events),
            "threat_breakdown": threat_counts,
            "severity_breakdown": severity_counts,
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "recommendations": self._get_security_recommendations(recent_events),
            "active_tokens": len(self.auth_manager.active_tokens),
            "failed_login_attempts": len(self.auth_manager.failed_attempts)
        }
        
        return summary
    
    def _calculate_risk_score(self, events: List[SecurityEvent]) -> float:
        """Calculate overall risk score"""
        if not events:
            return 0.0
        
        score = 0.0
        for event in events:
            # Weight by severity
            severity_weights = {
                SecurityLevel.LOW: 0.1,
                SecurityLevel.MEDIUM: 0.3,
                SecurityLevel.HIGH: 0.7,
                SecurityLevel.CRITICAL: 1.0
            }
            
            # Weight by threat type
            threat_weights = {
                ThreatType.ADVERSARIAL_ATTACK: 1.0,
                ThreatType.DATA_POISONING: 0.9,
                ThreatType.UNAUTHORIZED_ACCESS: 0.7,
                ThreatType.MALICIOUS_INPUT: 0.5,
                ThreatType.DATA_LEAKAGE: 0.8
            }
            
            event_score = (
                severity_weights.get(event.severity, 0.5) *
                threat_weights.get(event.threat_type, 0.5) *
                event.confidence_score
            )
            
            score += event_score
        
        # Normalize by time and number of events
        normalized_score = min(score / max(len(events), 1), 1.0)
        return normalized_score
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Get risk level description"""
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_security_recommendations(self, events: List[SecurityEvent]) -> List[str]:
        """Get security recommendations based on events"""
        recommendations = []
        
        # Count event types
        adversarial_count = sum(1 for e in events if e.threat_type == ThreatType.ADVERSARIAL_ATTACK)
        auth_failures = sum(1 for e in events if e.threat_type == ThreatType.UNAUTHORIZED_ACCESS)
        
        if adversarial_count > 5:
            recommendations.append("Consider implementing additional adversarial defenses")
            recommendations.append("Review and update anomaly detection thresholds")
        
        if auth_failures > 10:
            recommendations.append("Investigate potential brute force attacks")
            recommendations.append("Consider implementing stronger authentication measures")
        
        if len(events) > 50:
            recommendations.append("High volume of security events detected")
            recommendations.append("Consider implementing stricter rate limiting")
        
        if not recommendations:
            recommendations.append("Security posture appears healthy")
        
        return recommendations
    
    async def export_security_report(self, output_path: str = None) -> str:
        """Export comprehensive security report"""
        try:
            summary = await self.get_security_summary(24)
            
            report = {
                "report_timestamp": datetime.utcnow().isoformat(),
                "security_summary": summary,
                "recent_events": [asdict(event) for event in self.security_events[-100:]],
                "security_policies": self.security_policies,
                "system_status": {
                    "adversarial_detector_active": self.adversarial_detector is not None,
                    "encryption_enabled": True,
                    "authentication_required": self.security_policies["require_authentication"]
                }
            }
            
            if not output_path:
                output_path = f"security_report_{int(time.time())}.json"
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Security report exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export security report: {e}")
            raise

def main():
    """Main function for security framework demonstration"""
    
    async def run_security_demo():
        # Create reference data for adversarial detection
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.uniform(0, 10, 1000),
            'feature_3': np.random.choice([0, 1], 1000),
            'feature_4': np.random.exponential(1, 1000)
        })
        
        # Initialize security framework
        security = MLSecurityFramework(reference_data=reference_data)
        
        print("🔒 ML Security Framework Demo")
        print("=" * 40)
        
        # Test authentication
        print("\n1. Testing Authentication...")
        auth_result = security.auth_manager.authenticate_user("data_scientist", "ds_password")
        
        if auth_result.success:
            print(f"✅ Authentication successful for {auth_result.user_id}")
            print(f"   Permissions: {auth_result.permissions}")
            print(f"   Token expires: {auth_result.expires_at}")
            
            # Test secure prediction
            print("\n2. Testing Secure Prediction...")
            normal_features = {
                'feature_1': 0.5,
                'feature_2': 5.0,
                'feature_3': 1,
                'feature_4': 1.2
            }
            
            result = await security.secure_predict(
                normal_features,
                token=auth_result.token,
                user_id=auth_result.user_id,
                ip_address="192.168.1.1"
            )
            
            if "error" not in result:
                print(f"✅ Normal prediction successful: {result['prediction']:.3f}")
            else:
                print(f"❌ Prediction failed: {result['error']}")
            
            # Test adversarial detection
            print("\n3. Testing Adversarial Detection...")
            adversarial_features = {
                'feature_1': 10.0,  # Extreme value
                'feature_2': -5.0,  # Out of normal range
                'feature_3': 1,
                'feature_4': 100.0  # Very large value
            }
            
            result = await security.secure_predict(
                adversarial_features,
                token=auth_result.token,
                user_id=auth_result.user_id,
                ip_address="192.168.1.1"
            )
            
            if "security_code" in result:
                print(f"🛡️  Adversarial input detected and blocked")
            else:
                print(f"⚠️  Adversarial input processed: {result}")
        
        else:
            print(f"❌ Authentication failed: {auth_result.error_message}")
        
        # Test rate limiting
        print("\n4. Testing Rate Limiting...")
        for i in range(5):
            result = await security.secure_predict(
                normal_features,
                token=auth_result.token if auth_result.success else None,
                user_id="test_user",
                ip_address="192.168.1.100"
            )
            
            if result.get("code") == 429:
                print(f"🚫 Rate limit triggered after {i} requests")
                break
        
        # Test data privacy
        print("\n5. Testing Data Privacy...")
        sensitive_data = {
            "user_id": "user123",
            "email": "user@example.com",
            "feature_1": 1.5,
            "feature_2": 3.0
        }
        
        encrypted_data = security.privacy_protector.encrypt_sensitive_data(
            sensitive_data, ["user_id", "email"]
        )
        
        print(f"Original: {sensitive_data}")
        print(f"Encrypted: {encrypted_data}")
        
        decrypted_data = security.privacy_protector.decrypt_sensitive_data(
            encrypted_data, ["user_id", "email"]
        )
        print(f"Decrypted: {decrypted_data}")
        
        # Get security summary
        print("\n6. Security Summary...")
        summary = await security.get_security_summary()
        
        print(f"Total security events: {summary['total_events']}")
        print(f"Risk level: {summary['risk_level']}")
        print(f"Risk score: {summary['risk_score']:.3f}")
        
        if summary['recommendations']:
            print("Recommendations:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        # Export security report
        report_path = await security.export_security_report()
        print(f"\n📊 Security report exported to: {report_path}")
    
    # Run demo
    asyncio.run(run_security_demo())

if __name__ == "__main__":
    main()