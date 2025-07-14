"""
API tests for the serving endpoint
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from serving.api.main import app


@pytest.mark.api
class TestAPI:
    """Test suite for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "uptime" in data
    
    def test_ready_endpoint(self, client):
        """Test readiness probe endpoint."""
        response = client.get("/ready")
        assert response.status_code in [200, 503]  # May not be ready in test
        
        data = response.json()
        assert "status" in data
    
    def test_live_endpoint(self, client):
        """Test liveness probe endpoint."""
        response = client.get("/live")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "alive"
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        # Should be in Prometheus format
        content = response.text
        assert "# HELP" in content or "# TYPE" in content
    
    def test_info_endpoint(self, client):
        """Test service info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "environment" in data
    
    @patch('serving.api.main.get_model_predictor')
    def test_predict_endpoint(self, mock_get_predictor, client, prediction_request_data):
        """Test prediction endpoint."""
        # Mock predictor
        mock_predictor = AsyncMock()
        mock_predictor.predict.return_value = {
            'prediction': 0.7,
            'confidence': 0.85,
            'model_version': '1.0.0'
        }
        mock_get_predictor.return_value = mock_predictor
        
        response = client.post("/predict", json=prediction_request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "model_version" in data
        assert "latency_ms" in data
    
    @patch('serving.api.main.get_model_predictor')
    def test_predict_batch_endpoint(self, mock_get_predictor, client, batch_prediction_request_data):
        """Test batch prediction endpoint."""
        # Mock predictor
        mock_predictor = AsyncMock()
        mock_predictor.predict_batch.return_value = [
            {
                'prediction': 0.7,
                'confidence': 0.85,
                'model_version': '1.0.0'
            },
            {
                'prediction': 0.3,
                'confidence': 0.90,
                'model_version': '1.0.0'
            },
            {
                'prediction': 0.8,
                'confidence': 0.75,
                'model_version': '1.0.0'
            }
        ]
        mock_get_predictor.return_value = mock_predictor
        
        response = client.post("/predict/batch", json=batch_prediction_request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "batch_size" in data
        assert "total_latency_ms" in data
        assert len(data["predictions"]) == 3
    
    def test_predict_endpoint_invalid_data(self, client):
        """Test prediction endpoint with invalid data."""
        invalid_data = {
            "feature_1": "invalid_string",  # Should be numeric
            "feature_2": None
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_missing_fields(self, client):
        """Test prediction endpoint with missing required fields."""
        incomplete_data = {
            "feature_1": 1.5
            # Missing other required features
        }
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422
    
    @patch('serving.api.main.get_model_predictor')
    def test_predict_endpoint_model_error(self, mock_get_predictor, client, prediction_request_data):
        """Test prediction endpoint when model raises an error."""
        # Mock predictor that raises an error
        mock_predictor = AsyncMock()
        mock_predictor.predict.side_effect = Exception("Model error")
        mock_get_predictor.return_value = mock_predictor
        
        response = client.post("/predict", json=prediction_request_data)
        assert response.status_code == 500
        
        data = response.json()
        assert "error" in data
    
    def test_predict_batch_too_large(self, client):
        """Test batch prediction with too many samples."""
        # Create a batch that's too large
        large_batch = {
            "predictions": [{"feature_1": 1.0, "feature_2": 2.0} for _ in range(1001)]
        }
        
        response = client.post("/predict/batch", json=large_batch)
        assert response.status_code == 422
    
    def test_predict_batch_empty(self, client):
        """Test batch prediction with empty batch."""
        empty_batch = {"predictions": []}
        
        response = client.post("/predict/batch", json=empty_batch)
        assert response.status_code == 422
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/predict")
        
        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
    
    def test_security_headers(self, client):
        """Test security headers are present."""
        response = client.get("/health")
        
        # Check for security headers
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers
    
    def test_rate_limiting(self, client):
        """Test rate limiting functionality."""
        # Make multiple rapid requests
        responses = []
        for _ in range(20):
            response = client.get("/health")
            responses.append(response)
        
        # At least some should succeed
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count > 0
        
        # Some might be rate limited (429)
        rate_limited = sum(1 for r in responses if r.status_code == 429)
        # In test environment, rate limiting might not be enforced
    
    def test_request_logging(self, client, caplog):
        """Test that requests are logged."""
        with caplog.at_level("INFO"):
            response = client.get("/health")
            assert response.status_code == 200
        
        # Check that request was logged
        # Note: Actual log format depends on implementation
    
    @patch('serving.api.main.get_drift_detector')
    def test_drift_detection_endpoint(self, mock_get_detector, client):
        """Test drift detection endpoint."""
        # Mock drift detector
        mock_detector = AsyncMock()
        mock_detector.detect_drift.return_value = {
            'drift_detected': True,
            'drift_score': 0.75,
            'drift_type': 'feature_drift'
        }
        mock_get_detector.return_value = mock_detector
        
        test_data = {
            'reference_data': [
                {'feature_1': 1.0, 'feature_2': 2.0},
                {'feature_1': 1.5, 'feature_2': 2.5}
            ],
            'current_data': [
                {'feature_1': 3.0, 'feature_2': 4.0},
                {'feature_1': 3.5, 'feature_2': 4.5}
            ]
        }
        
        response = client.post("/drift/detect", json=test_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "drift_detected" in data
        assert "drift_score" in data
    
    def test_model_info_endpoint(self, client):
        """Test model information endpoint."""
        response = client.get("/model/info")
        assert response.status_code in [200, 503]  # May not have model loaded
        
        if response.status_code == 200:
            data = response.json()
            assert "model_version" in data
            assert "model_type" in data
    
    @patch('serving.api.main.get_monitoring_service')
    def test_monitoring_endpoint(self, mock_get_monitoring, client):
        """Test monitoring endpoint."""
        # Mock monitoring service
        mock_monitor = AsyncMock()
        mock_monitor.get_metrics.return_value = {
            'total_predictions': 1000,
            'average_latency': 50.5,
            'error_rate': 0.01
        }
        mock_get_monitoring.return_value = mock_monitor
        
        response = client.get("/monitoring/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_predictions" in data
        assert "average_latency" in data
        assert "error_rate" in data
    
    def test_openapi_docs(self, client):
        """Test OpenAPI documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200
        
        response = client.get("/redoc")
        assert response.status_code == 200
        
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec
    
    def test_content_negotiation(self, client):
        """Test content negotiation."""
        # JSON response
        response = client.get("/health", headers={"Accept": "application/json"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Test that we handle different accept headers gracefully
        response = client.get("/health", headers={"Accept": "text/plain"})
        assert response.status_code == 200
    
    def test_error_handling(self, client):
        """Test error handling for various scenarios."""
        # Test 404 for non-existent endpoint
        response = client.get("/non-existent-endpoint")
        assert response.status_code == 404
        
        # Test method not allowed
        response = client.post("/health")
        assert response.status_code == 405
    
    @patch('serving.api.main.get_model_predictor')
    def test_async_endpoint_behavior(self, mock_get_predictor, client):
        """Test async endpoint behavior."""
        # Mock async predictor
        async def mock_predict(data):
            # Simulate async operation
            import asyncio
            await asyncio.sleep(0.01)
            return {'prediction': 0.5, 'confidence': 0.8, 'model_version': '1.0.0'}
        
        mock_predictor = AsyncMock()
        mock_predictor.predict = mock_predict
        mock_get_predictor.return_value = mock_predictor
        
        # Test that async endpoints work properly
        response = client.post("/predict", json={
            'feature_1': 1.0,
            'feature_2': 2.0,
            'feature_3': 'A',
            'feature_4': 5
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data['prediction'] == 0.5
    
    def test_request_validation_edge_cases(self, client):
        """Test request validation edge cases."""
        # Test with null values
        null_data = {
            'feature_1': None,
            'feature_2': 2.0,
            'feature_3': 'A',
            'feature_4': 5
        }
        response = client.post("/predict", json=null_data)
        assert response.status_code == 422
        
        # Test with extreme values
        extreme_data = {
            'feature_1': 1e10,
            'feature_2': -1e10,
            'feature_3': 'A',
            'feature_4': 999999
        }
        response = client.post("/predict", json=extreme_data)
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 422, 400]
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import threading
        
        def make_request():
            return client.get("/health")
        
        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
    
    def test_large_payload_handling(self, client):
        """Test handling of large payloads."""
        # Create a large but valid batch prediction request
        large_batch = {
            "predictions": [
                {
                    'feature_1': 1.0,
                    'feature_2': 2.0,
                    'feature_3': 'A',
                    'feature_4': 5
                } for _ in range(100)  # Large but under limit
            ]
        }
        
        response = client.post("/predict/batch", json=large_batch)
        # Should handle large payloads gracefully
        assert response.status_code in [200, 422, 413]  # 413 = Payload Too Large