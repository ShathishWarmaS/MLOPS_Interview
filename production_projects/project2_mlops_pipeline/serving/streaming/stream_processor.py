"""
Real-time Stream Processing for MLOps Pipeline
Apache Kafka integration for streaming ML inference
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import redis
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from serving.api.predictor import ModelPredictor
from monitoring.model_monitor import ModelMonitor

logger = logging.getLogger(__name__)

@dataclass
class StreamEvent:
    """Stream event data structure"""
    event_id: str
    timestamp: float
    event_type: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PredictionEvent:
    """Prediction event for streaming"""
    event_id: str
    timestamp: float
    features: Dict[str, Any]
    prediction: float
    confidence: float
    model_version: str
    processing_time_ms: float
    source_topic: str

class StreamProcessor:
    """Real-time stream processor for ML predictions"""
    
    def __init__(self, 
                 kafka_servers: List[str] = None,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 batch_size: int = 100,
                 batch_timeout: float = 1.0):
        
        self.kafka_servers = kafka_servers or ["localhost:9092"]
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Components
        self.predictor: Optional[ModelPredictor] = None
        self.monitor: Optional[ModelMonitor] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Kafka components
        self.producer: Optional[KafkaProducer] = None
        self.consumer: Optional[KafkaConsumer] = None
        
        # Processing state
        self.is_running = False
        self.event_buffer = []
        self.buffer_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Metrics
        self.processed_events = 0
        self.failed_events = 0
        self.start_time = time.time()
        
        # Configuration
        self.input_topics = ["ml-features", "prediction-requests"]
        self.output_topics = {
            "predictions": "ml-predictions",
            "alerts": "ml-alerts",
            "metrics": "ml-metrics"
        }
    
    async def initialize(self):
        """Initialize stream processor"""
        try:
            logger.info("Initializing stream processor...")
            
            # Initialize model predictor
            self.predictor = ModelPredictor()
            await self.predictor.load_model()
            
            # Initialize monitor
            self.monitor = ModelMonitor()
            await self.monitor.initialize()
            
            # Initialize Redis
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
            
            # Test Redis connection
            self.redis_client.ping()
            
            # Initialize Kafka
            await self._initialize_kafka()
            
            logger.info("✅ Stream processor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize stream processor: {e}")
            raise
    
    async def _initialize_kafka(self):
        """Initialize Kafka producer and consumer"""
        try:
            # Create producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retries=3,
                acks='all',
                compression_type='gzip'
            )
            
            # Create consumer
            self.consumer = KafkaConsumer(
                *self.input_topics,
                bootstrap_servers=self.kafka_servers,
                group_id='mlops-stream-processor',
                auto_offset_reset='latest',
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                max_poll_records=self.batch_size
            )
            
            logger.info(f"Kafka initialized. Topics: {self.input_topics}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {e}")
            raise
    
    async def start_processing(self):
        """Start stream processing"""
        try:
            logger.info("Starting stream processing...")
            self.is_running = True
            
            # Start consumer loop
            consumer_task = asyncio.create_task(self._consumer_loop())
            
            # Start batch processor
            processor_task = asyncio.create_task(self._batch_processor())
            
            # Start metrics reporter
            metrics_task = asyncio.create_task(self._metrics_reporter())
            
            # Wait for all tasks
            await asyncio.gather(consumer_task, processor_task, metrics_task)
            
        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            raise
        finally:
            self.is_running = False
    
    async def _consumer_loop(self):
        """Main consumer loop"""
        logger.info("Starting Kafka consumer loop...")
        
        while self.is_running:
            try:
                # Poll for messages
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if not message_batch:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process messages
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        await self._handle_message(message)
                
            except Exception as e:
                logger.error(f"Consumer loop error: {e}")
                await asyncio.sleep(1)
    
    async def _handle_message(self, message):
        """Handle individual Kafka message"""
        try:
            # Parse message
            event = StreamEvent(
                event_id=message.key or f"event_{int(time.time() * 1000)}",
                timestamp=message.timestamp / 1000 if message.timestamp else time.time(),
                event_type=message.topic,
                data=message.value,
                metadata={
                    'offset': message.offset,
                    'partition': message.partition,
                    'topic': message.topic
                }
            )
            
            # Add to buffer
            with self.buffer_lock:
                self.event_buffer.append(event)
            
            # Cache in Redis for deduplication
            await self._cache_event(event)
            
        except Exception as e:
            logger.error(f"Failed to handle message: {e}")
            self.failed_events += 1
    
    async def _cache_event(self, event: StreamEvent):
        """Cache event in Redis for deduplication"""
        try:
            key = f"stream_event:{event.event_id}"
            value = json.dumps(asdict(event))
            
            # Set with expiration (1 hour)
            self.redis_client.setex(key, 3600, value)
            
        except Exception as e:
            logger.warning(f"Failed to cache event: {e}")
    
    async def _batch_processor(self):
        """Process events in batches"""
        logger.info("Starting batch processor...")
        
        while self.is_running:
            try:
                # Get batch from buffer
                batch = []
                with self.buffer_lock:
                    if len(self.event_buffer) >= self.batch_size:
                        batch = self.event_buffer[:self.batch_size]
                        self.event_buffer = self.event_buffer[self.batch_size:]
                    elif self.event_buffer:
                        # Process partial batch after timeout
                        batch = self.event_buffer.copy()
                        self.event_buffer.clear()
                
                if batch:
                    await self._process_batch(batch)
                else:
                    await asyncio.sleep(self.batch_timeout)
                
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch(self, events: List[StreamEvent]):
        """Process batch of events"""
        try:
            logger.debug(f"Processing batch of {len(events)} events")
            
            # Group events by type
            prediction_events = []
            monitoring_events = []
            
            for event in events:
                if event.event_type in ["ml-features", "prediction-requests"]:
                    prediction_events.append(event)
                else:
                    monitoring_events.append(event)
            
            # Process predictions
            if prediction_events:
                await self._process_prediction_batch(prediction_events)
            
            # Process monitoring events
            if monitoring_events:
                await self._process_monitoring_batch(monitoring_events)
            
            self.processed_events += len(events)
            
        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            self.failed_events += len(events)
    
    async def _process_prediction_batch(self, events: List[StreamEvent]):
        """Process batch of prediction events"""
        try:
            # Prepare batch prediction request
            samples = []
            event_mapping = {}
            
            for event in events:
                if 'features' in event.data:
                    features = event.data['features']
                    samples.append(features)
                    event_mapping[len(samples) - 1] = event
            
            if not samples:
                return
            
            # Make batch predictions
            start_time = time.time()
            predictions = await self.predictor.predict_batch(samples)
            processing_time = (time.time() - start_time) * 1000
            
            # Create prediction events
            prediction_events = []
            for i, prediction in enumerate(predictions):
                if i in event_mapping:
                    original_event = event_mapping[i]
                    
                    pred_event = PredictionEvent(
                        event_id=f"pred_{original_event.event_id}",
                        timestamp=time.time(),
                        features=samples[i],
                        prediction=prediction.prediction,
                        confidence=prediction.confidence,
                        model_version=prediction.model_version,
                        processing_time_ms=processing_time / len(samples),
                        source_topic=original_event.event_type
                    )
                    
                    prediction_events.append(pred_event)
            
            # Publish predictions
            await self._publish_predictions(prediction_events)
            
            # Log to monitor
            for event in events:
                await self.monitor.log_prediction(
                    request=event.data,
                    response=prediction_events[0] if prediction_events else None,
                    latency_ms=processing_time / len(events)
                )
            
        except Exception as e:
            logger.error(f"Failed to process prediction batch: {e}")
            raise
    
    async def _process_monitoring_batch(self, events: List[StreamEvent]):
        """Process batch of monitoring events"""
        try:
            for event in events:
                # Process monitoring events (feedback, metrics, etc.)
                if event.data.get('type') == 'feedback':
                    await self._handle_feedback_event(event)
                elif event.data.get('type') == 'metric':
                    await self._handle_metric_event(event)
            
        except Exception as e:
            logger.error(f"Failed to process monitoring batch: {e}")
    
    async def _handle_feedback_event(self, event: StreamEvent):
        """Handle user feedback event"""
        try:
            feedback_data = event.data
            request_id = feedback_data.get('request_id')
            actual_label = feedback_data.get('actual_label')
            
            if request_id and actual_label is not None:
                await self.monitor.add_user_feedback(request_id, actual_label)
            
        except Exception as e:
            logger.error(f"Failed to handle feedback event: {e}")
    
    async def _handle_metric_event(self, event: StreamEvent):
        """Handle metric event"""
        try:
            # Store custom metrics
            metric_data = event.data
            key = f"custom_metric:{metric_data.get('name')}"
            value = json.dumps(metric_data)
            
            self.redis_client.setex(key, 300, value)  # 5 minute expiration
            
        except Exception as e:
            logger.error(f"Failed to handle metric event: {e}")
    
    async def _publish_predictions(self, prediction_events: List[PredictionEvent]):
        """Publish predictions to output topic"""
        try:
            output_topic = self.output_topics["predictions"]
            
            for pred_event in prediction_events:
                # Convert to dict
                event_data = asdict(pred_event)
                
                # Publish to Kafka
                future = self.producer.send(
                    output_topic,
                    key=pred_event.event_id,
                    value=event_data
                )
                
                # Don't wait for acknowledgment in streaming mode
                # future.get(timeout=1)  # Uncomment for guaranteed delivery
            
            logger.debug(f"Published {len(prediction_events)} predictions")
            
        except Exception as e:
            logger.error(f"Failed to publish predictions: {e}")
            raise
    
    async def _metrics_reporter(self):
        """Report processing metrics"""
        logger.info("Starting metrics reporter...")
        
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Report every 30 seconds
                
                # Calculate metrics
                uptime = time.time() - self.start_time
                processing_rate = self.processed_events / max(uptime, 1)
                error_rate = self.failed_events / max(self.processed_events + self.failed_events, 1)
                
                # Buffer status
                buffer_size = len(self.event_buffer)
                
                metrics = {
                    "timestamp": time.time(),
                    "uptime_seconds": uptime,
                    "processed_events": self.processed_events,
                    "failed_events": self.failed_events,
                    "processing_rate": processing_rate,
                    "error_rate": error_rate,
                    "buffer_size": buffer_size,
                    "predictor_ready": self.predictor.is_ready() if self.predictor else False
                }
                
                # Publish metrics
                await self._publish_metrics(metrics)
                
                # Log metrics
                logger.info(f"Stream metrics: {processing_rate:.2f} events/sec, "
                          f"error rate: {error_rate:.3f}, buffer: {buffer_size}")
                
            except Exception as e:
                logger.error(f"Metrics reporter error: {e}")
    
    async def _publish_metrics(self, metrics: Dict[str, Any]):
        """Publish metrics to Kafka"""
        try:
            output_topic = self.output_topics["metrics"]
            
            self.producer.send(
                output_topic,
                key="stream_processor_metrics",
                value=metrics
            )
            
        except Exception as e:
            logger.error(f"Failed to publish metrics: {e}")
    
    async def stop_processing(self):
        """Stop stream processing"""
        logger.info("Stopping stream processing...")
        
        self.is_running = False
        
        # Close Kafka connections
        if self.producer:
            self.producer.close()
        
        if self.consumer:
            self.consumer.close()
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("✅ Stream processing stopped")
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        uptime = time.time() - self.start_time
        processing_rate = self.processed_events / max(uptime, 1)
        error_rate = self.failed_events / max(self.processed_events + self.failed_events, 1)
        
        return {
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "processed_events": self.processed_events,
            "failed_events": self.failed_events,
            "processing_rate": processing_rate,
            "error_rate": error_rate,
            "buffer_size": len(self.event_buffer),
            "predictor_ready": self.predictor.is_ready() if self.predictor else False,
            "kafka_connected": self.producer is not None and self.consumer is not None,
            "redis_connected": self.redis_client is not None
        }

class StreamingMLService:
    """High-level streaming ML service orchestrator"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.processor: Optional[StreamProcessor] = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load streaming configuration"""
        default_config = {
            "kafka": {
                "servers": ["localhost:9092"],
                "topics": {
                    "input": ["ml-features", "prediction-requests"],
                    "output": {
                        "predictions": "ml-predictions",
                        "alerts": "ml-alerts",
                        "metrics": "ml-metrics"
                    }
                }
            },
            "redis": {
                "host": "localhost",
                "port": 6379
            },
            "processing": {
                "batch_size": 100,
                "batch_timeout": 1.0
            }
        }
        
        if config_path and Path(config_path).exists():
            import yaml
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    async def start(self):
        """Start streaming service"""
        try:
            logger.info("Starting streaming ML service...")
            
            # Initialize processor
            self.processor = StreamProcessor(
                kafka_servers=self.config["kafka"]["servers"],
                redis_host=self.config["redis"]["host"],
                redis_port=self.config["redis"]["port"],
                batch_size=self.config["processing"]["batch_size"],
                batch_timeout=self.config["processing"]["batch_timeout"]
            )
            
            # Initialize and start processing
            await self.processor.initialize()
            await self.processor.start_processing()
            
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Streaming service failed: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop streaming service"""
        if self.processor:
            await self.processor.stop_processing()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        if self.processor:
            return await self.processor.get_processing_stats()
        else:
            return {"status": "not_running"}

def main():
    """Main function for running streaming service"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MLOps Streaming Service')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start service
    service = StreamingMLService(args.config)
    
    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        logger.info("Streaming service stopped")

if __name__ == "__main__":
    main()