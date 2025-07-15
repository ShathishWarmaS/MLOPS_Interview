"""
LLM Server for Text Generation
Production-grade serving infrastructure for Large Language Models
"""

import asyncio
import logging
import time
import torch
import numpy as np
from typing import Dict, List, Any, Optional, AsyncIterator, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import queue
import threading
from collections import defaultdict, deque
import gc

# Transformers and model libraries
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    GenerationConfig, StoppingCriteria, StoppingCriteriaList,
    BitsAndBytesConfig
)
import torch.nn.functional as F
from accelerate import Accelerator
import bitsandbytes as bnb

# FastAPI for serving
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.config import Config
from shared.logging import setup_logger
from shared.metrics import MetricsCollector
from infrastructure.gpu_manager import GPUManager
from infrastructure.cache_manager import CacheManager
from model_management.safety.content_filter import ContentFilter

logger = setup_logger(__name__)

@dataclass
class GenerationRequest:
    """Text generation request"""
    request_id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 1
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    safety_check: bool = True
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class GenerationRequestModel(BaseModel):
    """Pydantic model for API requests"""
    prompt: str = Field(..., description="Input prompt for text generation")
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    top_k: int = Field(50, ge=1, le=200, description="Top-k sampling")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    do_sample: bool = Field(True, description="Whether to use sampling")
    num_return_sequences: int = Field(1, ge=1, le=5, description="Number of sequences to return")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Whether to stream the response")
    safety_check: bool = Field(True, description="Whether to perform safety checks")

@dataclass
class GenerationResponse:
    """Text generation response"""
    request_id: str
    generated_text: Union[str, List[str]]
    prompt: str
    finish_reason: str
    usage: Dict[str, int]
    safety_score: Optional[float] = None
    processing_time_ms: float = 0.0
    model_name: str = ""
    timestamp: str = ""

class CustomStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for generation"""
    
    def __init__(self, stop_sequences: List[str], tokenizer):
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        self.stop_token_ids = []
        
        for stop_seq in stop_sequences:
            tokens = tokenizer.encode(stop_seq, add_special_tokens=False)
            self.stop_token_ids.extend(tokens)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if any stop sequence is generated
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1].item() == stop_id:
                return True
        return False

class BatchProcessor:
    """Batch processing for efficient inference"""
    
    def __init__(self, max_batch_size: int = 8, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = queue.Queue()
        self.response_futures = {}
        self.processing = False
    
    async def add_request(self, request: GenerationRequest) -> GenerationResponse:
        """Add request to batch processing queue"""
        future = asyncio.Future()
        self.response_futures[request.request_id] = future
        self.pending_requests.put(request)
        
        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        """Process batched requests"""
        self.processing = True
        
        try:
            # Wait for requests or timeout
            await asyncio.sleep(self.max_wait_time)
            
            # Collect batch
            batch = []
            while not self.pending_requests.empty() and len(batch) < self.max_batch_size:
                batch.append(self.pending_requests.get())
            
            if batch:
                # Process batch (this would be implemented in the server)
                for request in batch:
                    # For now, create a placeholder response
                    response = GenerationResponse(
                        request_id=request.request_id,
                        generated_text="Batched response placeholder",
                        prompt=request.prompt,
                        finish_reason="completed",
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    )
                    
                    future = self.response_futures.pop(request.request_id)
                    future.set_result(response)
        
        finally:
            self.processing = False

class LLMServer:
    """Production-grade LLM serving server"""
    
    def __init__(self, config: Config):
        self.config = config
        self.gpu_manager = GPUManager(config)
        self.cache_manager = CacheManager(config)
        self.content_filter = ContentFilter(config)
        self.metrics = MetricsCollector()
        
        # Model configuration
        self.model_name = config.get('model_name', 'microsoft/DialoGPT-medium')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_memory_per_gpu = config.get('max_memory_per_gpu', '8GB')
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.accelerator = None
        
        # Batch processing
        self.batch_processor = BatchProcessor(
            max_batch_size=config.get('max_batch_size', 8),
            max_wait_time=config.get('max_wait_time', 0.1)
        )
        
        # Performance tracking
        self.request_queue = deque(maxlen=1000)
        self.active_requests = {}
        self.model_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_generated': 0,
            'avg_latency_ms': 0.0
        }
        
        # Initialize server
        asyncio.create_task(self._initialize_model())
        
        logger.info(f"LLM Server initialized for model: {self.model_name}")
    
    async def _initialize_model(self):
        """Initialize the language model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Setup accelerator for distributed/mixed precision
            self.accelerator = Accelerator(
                mixed_precision='fp16' if torch.cuda.is_available() else 'no'
            )
            
            # Configure quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            ) if self.config.get('use_quantization', True) else None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
                'device_map': 'auto' if torch.cuda.is_available() else None,
            }
            
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Prepare model with accelerator
            self.model = self.accelerator.prepare(self.model)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Warm up the model
            await self._warmup_model()
            
            logger.info("✅ Model loaded and ready for inference")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    async def _warmup_model(self):
        """Warm up the model with a test generation"""
        try:
            warmup_prompt = "Hello, this is a test."
            warmup_request = GenerationRequest(
                request_id="warmup",
                prompt=warmup_prompt,
                max_tokens=10,
                temperature=0.7
            )
            
            await self._generate_text(warmup_request)
            logger.info("Model warmup completed")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from prompt"""
        try:
            start_time = time.time()
            self.model_stats['total_requests'] += 1
            self.active_requests[request.request_id] = {
                'start_time': start_time,
                'prompt': request.prompt[:100]  # First 100 chars for logging
            }
            
            # Safety check
            if request.safety_check:
                safety_result = await self.content_filter.check_content(request.prompt)
                if not safety_result.is_safe:
                    self.model_stats['failed_requests'] += 1
                    raise HTTPException(
                        status_code=400,
                        detail=f"Content safety check failed: {safety_result.reason}"
                    )
            
            # Check cache
            cache_key = self._generate_cache_key(request)
            cached_response = await self.cache_manager.get(cache_key)
            if cached_response:
                logger.info(f"Cache hit for request {request.request_id}")
                self.metrics.increment_counter('cache_hits')
                return cached_response
            
            # Generate text
            if request.stream:
                return await self._generate_streaming(request)
            else:
                response = await self._generate_text(request)
            
            # Cache the response
            if not request.stream:
                await self.cache_manager.set(cache_key, response, ttl=3600)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            response.processing_time_ms = processing_time
            
            self.model_stats['successful_requests'] += 1
            self.model_stats['avg_latency_ms'] = (
                (self.model_stats['avg_latency_ms'] * (self.model_stats['successful_requests'] - 1) + processing_time) 
                / self.model_stats['successful_requests']
            )
            
            self.metrics.record_histogram('generation_latency_ms', processing_time)
            self.metrics.increment_counter('successful_generations')
            
            # Cleanup
            self.active_requests.pop(request.request_id, None)
            
            return response
            
        except Exception as e:
            self.model_stats['failed_requests'] += 1
            self.metrics.increment_counter('failed_generations')
            self.active_requests.pop(request.request_id, None)
            
            logger.error(f"Generation failed for request {request.request_id}: {e}")
            raise
    
    async def _generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using the model"""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                request.prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Setup generation config
            generation_config = GenerationConfig(
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
                num_return_sequences=request.num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Setup stopping criteria
            stopping_criteria = StoppingCriteriaList()
            if request.stop_sequences:
                stopping_criteria.append(
                    CustomStoppingCriteria(request.stop_sequences, self.tokenizer)
                )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    generation_config=generation_config,
                    stopping_criteria=stopping_criteria,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode outputs
            generated_sequences = outputs.sequences
            input_length = inputs.shape[1]
            
            generated_texts = []
            for sequence in generated_sequences:
                # Remove input tokens from output
                generated_tokens = sequence[input_length:]
                
                # Decode to text
                generated_text = self.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                generated_texts.append(generated_text)
            
            # Calculate usage
            prompt_tokens = inputs.shape[1]
            completion_tokens = sum(len(seq) - input_length for seq in generated_sequences)
            total_tokens = prompt_tokens + completion_tokens
            
            # Update token stats
            self.model_stats['total_tokens_generated'] += completion_tokens
            
            # Determine finish reason
            finish_reason = "length" if completion_tokens >= request.max_tokens else "stop"
            
            # Create response
            response = GenerationResponse(
                request_id=request.request_id,
                generated_text=generated_texts[0] if len(generated_texts) == 1 else generated_texts,
                prompt=request.prompt,
                finish_reason=finish_reason,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                },
                model_name=self.model_name,
                timestamp=datetime.utcnow().isoformat()
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    async def _generate_streaming(self, request: GenerationRequest) -> AsyncIterator[str]:
        """Generate text with streaming response"""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                request.prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            input_length = inputs.shape[1]
            
            # Initialize generation
            past_key_values = None
            generated_tokens = []
            
            for _ in range(request.max_tokens):
                with torch.no_grad():
                    if past_key_values is None:
                        # First iteration
                        outputs = self.model(inputs, use_cache=True)
                    else:
                        # Subsequent iterations
                        outputs = self.model(
                            inputs[:, -1:], 
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                    
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
                
                # Apply temperature
                if request.temperature > 0:
                    logits = logits / request.temperature
                
                # Apply top-k filtering
                if request.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, request.top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) sampling
                if request.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > request.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = float('-inf')
                
                # Sample next token
                if request.do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Check for stop tokens
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Check for custom stop sequences
                if request.stop_sequences:
                    current_text = self.tokenizer.decode(generated_tokens + [next_token.item()])
                    if any(stop_seq in current_text for stop_seq in request.stop_sequences):
                        break
                
                generated_tokens.append(next_token.item())
                
                # Decode and yield the new token
                new_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                
                yield f"data: {json.dumps({'token': new_text, 'done': False})}\\n\\n"
                
                # Update inputs for next iteration
                inputs = torch.cat([inputs, next_token], dim=1)
            
            # Send completion signal
            yield f"data: {json.dumps({'token': '', 'done': True})}\\n\\n"
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\\n\\n"
    
    def _generate_cache_key(self, request: GenerationRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            'prompt': request.prompt,
            'max_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'top_k': request.top_k,
            'model': self.model_name
        }
        return f"llm_cache:{hash(str(key_data))}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'active_requests': len(self.active_requests),
            'model_stats': self.model_stats,
            'gpu_memory_usage': self._get_gpu_memory_usage(),
            'cache_stats': self.cache_manager.get_stats() if self.cache_manager else {}
        }
    
    def _get_gpu_memory_usage(self) -> Dict[str, Any]:
        """Get GPU memory usage statistics"""
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
            }
        return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        try:
            # Quick inference test
            test_prompt = "Health check test"
            test_request = GenerationRequest(
                request_id="health_check",
                prompt=test_prompt,
                max_tokens=5,
                temperature=0.1
            )
            
            start_time = time.time()
            response = await self._generate_text(test_request)
            health_check_latency = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'health_check_latency_ms': health_check_latency,
                'gpu_available': torch.cuda.is_available(),
                'memory_usage': self._get_gpu_memory_usage()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model_loaded': self.model is not None
            }

# FastAPI app for serving
def create_app(config: Config) -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="LLM Text Generation Server",
        description="Production-grade LLM serving API",
        version="1.0.0"
    )
    
    # Initialize server
    llm_server = LLMServer(config)
    
    @app.post("/generate", response_model=Dict[str, Any])
    async def generate_text(request: GenerationRequestModel):
        """Generate text from prompt"""
        generation_request = GenerationRequest(
            request_id=f"req_{int(time.time() * 1000)}",
            **request.dict()
        )
        
        if request.stream:
            return StreamingResponse(
                llm_server._generate_streaming(generation_request),
                media_type="text/plain"
            )
        else:
            response = await llm_server.generate(generation_request)
            return asdict(response)
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return await llm_server.health_check()
    
    @app.get("/stats")
    async def get_stats():
        """Get server statistics"""
        return llm_server.get_stats()
    
    return app

# CLI entry point
async def main():
    """Main entry point"""
    config = Config()
    app = create_app(config)
    
    # Run server
    uvicorn.run(
        app,
        host=config.get('host', '0.0.0.0'),
        port=config.get('port', 8000),
        workers=1  # Single worker for GPU models
    )

if __name__ == "__main__":
    asyncio.run(main())