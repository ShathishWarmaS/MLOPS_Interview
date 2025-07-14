"""
MLOps Cost Optimization and Resource Management
Intelligent resource allocation and cost monitoring for ML infrastructure
"""

import asyncio
import logging
import time
import json
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
import statistics
from collections import defaultdict, deque

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Cloud provider libraries (mock for demonstration)
# In production, use actual cloud provider SDKs
import boto3  # For AWS
from kubernetes import client, config  # For Kubernetes

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Types of computational resources"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"

class OptimizationStrategy(Enum):
    """Cost optimization strategies"""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_LATENCY = "minimize_latency"
    BALANCE_COST_PERFORMANCE = "balance_cost_performance"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"

class InstanceType(Enum):
    """Cloud instance types"""
    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"
    GPU_SMALL = "gpu_small"
    GPU_LARGE = "gpu_large"

@dataclass
class ResourceUsage:
    """Resource usage metrics"""
    timestamp: float
    resource_type: ResourceType
    usage_percent: float
    allocated_amount: float
    used_amount: float
    cost_per_hour: float
    instance_id: str
    region: str

@dataclass
class CostAnalysis:
    """Cost analysis result"""
    period_start: datetime
    period_end: datetime
    total_cost: float
    cost_breakdown: Dict[str, float]
    resource_utilization: Dict[str, float]
    waste_analysis: Dict[str, Any]
    optimization_opportunities: List[Dict[str, Any]]
    projected_savings: float

@dataclass
class ResourceRecommendation:
    """Resource optimization recommendation"""
    resource_id: str
    current_config: Dict[str, Any]
    recommended_config: Dict[str, Any]
    expected_cost_savings: float
    expected_performance_impact: str
    confidence_score: float
    reasoning: str
    implementation_complexity: str

@dataclass
class AutoScalingConfig:
    """Auto-scaling configuration"""
    min_instances: int
    max_instances: int
    target_cpu_utilization: float
    target_memory_utilization: float
    scale_up_threshold: float
    scale_down_threshold: float
    cooldown_period_minutes: int
    enable_predictive_scaling: bool

class CloudCostTracker:
    """Cloud cost tracking and analysis"""
    
    def __init__(self, cloud_provider: str = "aws"):
        self.cloud_provider = cloud_provider
        self.cost_data = []
        self.instance_pricing = self._load_instance_pricing()
        self.lock = threading.Lock()
    
    def _load_instance_pricing(self) -> Dict[str, Dict[str, float]]:
        """Load instance pricing data (mock implementation)"""
        # In production, fetch real-time pricing from cloud provider APIs
        return {
            "us-east-1": {
                "t3.micro": 0.0104,
                "t3.small": 0.0208,
                "t3.medium": 0.0416,
                "t3.large": 0.0832,
                "t3.xlarge": 0.1664,
                "m5.large": 0.096,
                "m5.xlarge": 0.192,
                "c5.large": 0.085,
                "c5.xlarge": 0.17,
                "p3.2xlarge": 3.06,  # GPU instance
                "p3.8xlarge": 12.24,
                "r5.large": 0.126,   # Memory optimized
                "r5.xlarge": 0.252
            },
            "us-west-2": {
                "t3.micro": 0.0104,
                "t3.small": 0.0208,
                "t3.medium": 0.0416,
                "t3.large": 0.0832,
                "t3.xlarge": 0.1664,
                "m5.large": 0.096,
                "m5.xlarge": 0.192,
                "c5.large": 0.085,
                "c5.xlarge": 0.17,
                "p3.2xlarge": 3.06,
                "p3.8xlarge": 12.24,
                "r5.large": 0.126,
                "r5.xlarge": 0.252
            }
        }
    
    def record_usage(self, usage: ResourceUsage):
        """Record resource usage"""
        with self.lock:
            self.cost_data.append(usage)
    
    def calculate_cost(self, 
                      instance_type: str, 
                      region: str, 
                      hours: float) -> float:
        """Calculate cost for instance usage"""
        hourly_rate = self.instance_pricing.get(region, {}).get(instance_type, 0.0)
        return hourly_rate * hours
    
    def analyze_costs(self, 
                     start_date: datetime, 
                     end_date: datetime) -> CostAnalysis:
        """Analyze costs for a given period"""
        try:
            # Filter data for the period
            period_data = [
                usage for usage in self.cost_data
                if start_date.timestamp() <= usage.timestamp <= end_date.timestamp()
            ]
            
            if not period_data:
                return CostAnalysis(
                    period_start=start_date,
                    period_end=end_date,
                    total_cost=0.0,
                    cost_breakdown={},
                    resource_utilization={},
                    waste_analysis={},
                    optimization_opportunities=[],
                    projected_savings=0.0
                )
            
            # Calculate total cost
            total_cost = sum(usage.cost_per_hour for usage in period_data)
            
            # Cost breakdown by resource type
            cost_breakdown = defaultdict(float)
            utilization_by_type = defaultdict(list)
            
            for usage in period_data:
                cost_breakdown[usage.resource_type.value] += usage.cost_per_hour
                utilization_by_type[usage.resource_type.value].append(usage.usage_percent)
            
            # Average utilization
            resource_utilization = {
                resource_type: statistics.mean(utilizations)
                for resource_type, utilizations in utilization_by_type.items()
            }
            
            # Waste analysis
            waste_analysis = self._analyze_waste(period_data, resource_utilization)
            
            # Optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(
                period_data, resource_utilization
            )
            
            # Calculate projected savings
            projected_savings = sum(
                opp["potential_savings"] for opp in optimization_opportunities
            )
            
            return CostAnalysis(
                period_start=start_date,
                period_end=end_date,
                total_cost=total_cost,
                cost_breakdown=dict(cost_breakdown),
                resource_utilization=resource_utilization,
                waste_analysis=waste_analysis,
                optimization_opportunities=optimization_opportunities,
                projected_savings=projected_savings
            )
            
        except Exception as e:
            logger.error(f"Cost analysis failed: {e}")
            raise
    
    def _analyze_waste(self, 
                      period_data: List[ResourceUsage], 
                      utilization: Dict[str, float]) -> Dict[str, Any]:
        """Analyze resource waste"""
        waste_analysis = {
            "underutilized_resources": [],
            "idle_resources": [],
            "oversized_instances": [],
            "total_waste_cost": 0.0
        }
        
        # Group by instance
        instance_data = defaultdict(list)
        for usage in period_data:
            instance_data[usage.instance_id].append(usage)
        
        for instance_id, usages in instance_data.items():
            avg_cpu_util = statistics.mean([u.usage_percent for u in usages if u.resource_type == ResourceType.CPU])
            avg_memory_util = statistics.mean([u.usage_percent for u in usages if u.resource_type == ResourceType.MEMORY])
            
            total_cost = sum(u.cost_per_hour for u in usages)
            
            # Identify underutilized resources (< 20% utilization)
            if avg_cpu_util < 20 or avg_memory_util < 20:
                waste_analysis["underutilized_resources"].append({
                    "instance_id": instance_id,
                    "avg_cpu_utilization": avg_cpu_util,
                    "avg_memory_utilization": avg_memory_util,
                    "cost": total_cost,
                    "waste_percentage": 100 - max(avg_cpu_util, avg_memory_util)
                })
            
            # Identify idle resources (< 5% utilization)
            if avg_cpu_util < 5 and avg_memory_util < 5:
                waste_analysis["idle_resources"].append({
                    "instance_id": instance_id,
                    "cost": total_cost,
                    "reason": "Nearly idle - very low CPU and memory usage"
                })
                waste_analysis["total_waste_cost"] += total_cost * 0.9  # 90% waste
        
        return waste_analysis
    
    def _identify_optimization_opportunities(self, 
                                           period_data: List[ResourceUsage],
                                           utilization: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify cost optimization opportunities"""
        opportunities = []
        
        # Instance right-sizing opportunities
        instance_data = defaultdict(list)
        for usage in period_data:
            instance_data[usage.instance_id].append(usage)
        
        for instance_id, usages in instance_data.items():
            cpu_usages = [u.usage_percent for u in usages if u.resource_type == ResourceType.CPU]
            memory_usages = [u.usage_percent for u in usages if u.resource_type == ResourceType.MEMORY]
            
            if cpu_usages and memory_usages:
                max_cpu = max(cpu_usages)
                max_memory = max(memory_usages)
                avg_cpu = statistics.mean(cpu_usages)
                avg_memory = statistics.mean(memory_usages)
                
                current_cost = sum(u.cost_per_hour for u in usages)
                
                # Right-sizing opportunity
                if max_cpu < 60 and max_memory < 60:
                    potential_savings = current_cost * 0.3  # 30% savings estimate
                    opportunities.append({
                        "type": "instance_rightsizing",
                        "instance_id": instance_id,
                        "description": "Instance appears oversized based on usage patterns",
                        "current_max_cpu": max_cpu,
                        "current_max_memory": max_memory,
                        "potential_savings": potential_savings,
                        "confidence": 0.8
                    })
                
                # Spot instance opportunity
                if avg_cpu < 80 and avg_memory < 80:  # Stable workload
                    spot_savings = current_cost * 0.6  # 60% savings with spot instances
                    opportunities.append({
                        "type": "spot_instance",
                        "instance_id": instance_id,
                        "description": "Workload suitable for spot instances",
                        "potential_savings": spot_savings,
                        "confidence": 0.7
                    })
        
        # Storage optimization
        storage_cost = sum(
            u.cost_per_hour for u in period_data 
            if u.resource_type == ResourceType.STORAGE
        )
        
        if storage_cost > 0:
            storage_savings = storage_cost * 0.2  # 20% savings with optimization
            opportunities.append({
                "type": "storage_optimization",
                "description": "Optimize storage tiers and cleanup unused volumes",
                "potential_savings": storage_savings,
                "confidence": 0.6
            })
        
        return opportunities

class ResourceOptimizer:
    """ML resource optimization engine"""
    
    def __init__(self, cost_tracker: CloudCostTracker):
        self.cost_tracker = cost_tracker
        self.optimization_history = []
        self.performance_metrics = deque(maxlen=1000)
        self.lock = threading.Lock()
        
        # ML workload characteristics
        self.workload_patterns = {
            "training": {
                "cpu_intensive": True,
                "memory_intensive": True,
                "gpu_required": True,
                "duration": "hours",
                "predictable": False
            },
            "inference": {
                "cpu_intensive": False,
                "memory_intensive": False,
                "gpu_required": False,
                "duration": "continuous",
                "predictable": True
            },
            "data_processing": {
                "cpu_intensive": True,
                "memory_intensive": True,
                "gpu_required": False,
                "duration": "minutes",
                "predictable": True
            }
        }
    
    async def optimize_resources(self, 
                               workload_type: str,
                               performance_requirements: Dict[str, Any],
                               cost_constraints: Dict[str, Any]) -> List[ResourceRecommendation]:
        """Generate resource optimization recommendations"""
        try:
            logger.info(f"Optimizing resources for {workload_type} workload")
            
            # Analyze current usage patterns
            current_usage = await self._analyze_current_usage()
            
            # Get workload characteristics
            workload_config = self.workload_patterns.get(workload_type, {})
            
            # Generate recommendations
            recommendations = []
            
            # CPU optimization
            cpu_rec = await self._optimize_cpu_resources(
                current_usage, workload_config, performance_requirements, cost_constraints
            )
            if cpu_rec:
                recommendations.append(cpu_rec)
            
            # Memory optimization
            memory_rec = await self._optimize_memory_resources(
                current_usage, workload_config, performance_requirements, cost_constraints
            )
            if memory_rec:
                recommendations.append(memory_rec)
            
            # GPU optimization
            if workload_config.get("gpu_required"):
                gpu_rec = await self._optimize_gpu_resources(
                    current_usage, workload_config, performance_requirements, cost_constraints
                )
                if gpu_rec:
                    recommendations.append(gpu_rec)
            
            # Storage optimization
            storage_rec = await self._optimize_storage_resources(
                current_usage, workload_config, performance_requirements, cost_constraints
            )
            if storage_rec:
                recommendations.append(storage_rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")
            raise
    
    async def _analyze_current_usage(self) -> Dict[str, Any]:
        """Analyze current resource usage patterns"""
        with self.lock:
            recent_data = [
                usage for usage in self.cost_tracker.cost_data
                if usage.timestamp > time.time() - 3600  # Last hour
            ]
        
        if not recent_data:
            return {}
        
        # Aggregate by resource type
        usage_by_type = defaultdict(list)
        for usage in recent_data:
            usage_by_type[usage.resource_type.value].append(usage.usage_percent)
        
        analysis = {}
        for resource_type, usages in usage_by_type.items():
            analysis[resource_type] = {
                "current_utilization": statistics.mean(usages),
                "peak_utilization": max(usages),
                "variability": statistics.stdev(usages) if len(usages) > 1 else 0,
                "trend": self._calculate_trend(usages)
            }
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for usage values"""
        if len(values) < 3:
            return "stable"
        
        # Simple linear trend
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 5:
            return "increasing"
        elif slope < -5:
            return "decreasing"
        else:
            return "stable"
    
    async def _optimize_cpu_resources(self, 
                                    current_usage: Dict[str, Any],
                                    workload_config: Dict[str, Any],
                                    performance_req: Dict[str, Any],
                                    cost_constraints: Dict[str, Any]) -> Optional[ResourceRecommendation]:
        """Optimize CPU resource allocation"""
        
        cpu_usage = current_usage.get("cpu", {})
        if not cpu_usage:
            return None
        
        current_util = cpu_usage.get("current_utilization", 50)
        peak_util = cpu_usage.get("peak_utilization", 80)
        
        # Determine optimal CPU configuration
        target_utilization = performance_req.get("target_cpu_utilization", 70)
        
        if current_util < 30:
            # Over-provisioned
            recommended_reduction = 0.3  # Reduce by 30%
            recommendation = ResourceRecommendation(
                resource_id="cpu_optimization",
                current_config={"utilization": current_util, "type": "current"},
                recommended_config={"reduction": recommended_reduction, "type": "smaller_instance"},
                expected_cost_savings=100.0,  # Estimate
                expected_performance_impact="Minimal impact expected",
                confidence_score=0.8,
                reasoning=f"CPU utilization is low ({current_util:.1f}%), indicating over-provisioning",
                implementation_complexity="Low"
            )
            return recommendation
        
        elif peak_util > 90:
            # Under-provisioned
            recommended_increase = 0.2  # Increase by 20%
            recommendation = ResourceRecommendation(
                resource_id="cpu_scaling",
                current_config={"utilization": current_util, "peak": peak_util},
                recommended_config={"increase": recommended_increase, "type": "larger_instance"},
                expected_cost_savings=-50.0,  # Negative savings (cost increase)
                expected_performance_impact="Improved performance and reliability",
                confidence_score=0.9,
                reasoning=f"Peak CPU utilization is high ({peak_util:.1f}%), risking performance issues",
                implementation_complexity="Medium"
            )
            return recommendation
        
        return None
    
    async def _optimize_memory_resources(self, 
                                       current_usage: Dict[str, Any],
                                       workload_config: Dict[str, Any],
                                       performance_req: Dict[str, Any],
                                       cost_constraints: Dict[str, Any]) -> Optional[ResourceRecommendation]:
        """Optimize memory resource allocation"""
        
        memory_usage = current_usage.get("memory", {})
        if not memory_usage:
            return None
        
        current_util = memory_usage.get("current_utilization", 50)
        peak_util = memory_usage.get("peak_utilization", 80)
        
        if current_util < 40 and peak_util < 60:
            # Memory over-provisioned
            recommendation = ResourceRecommendation(
                resource_id="memory_optimization",
                current_config={"utilization": current_util, "peak": peak_util},
                recommended_config={"type": "memory_optimized_smaller"},
                expected_cost_savings=80.0,
                expected_performance_impact="No performance impact expected",
                confidence_score=0.75,
                reasoning=f"Memory utilization is consistently low (avg: {current_util:.1f}%, peak: {peak_util:.1f}%)",
                implementation_complexity="Low"
            )
            return recommendation
        
        elif peak_util > 85:
            # Memory under-provisioned
            recommendation = ResourceRecommendation(
                resource_id="memory_scaling",
                current_config={"utilization": current_util, "peak": peak_util},
                recommended_config={"type": "memory_optimized_larger"},
                expected_cost_savings=-75.0,
                expected_performance_impact="Prevent out-of-memory errors",
                confidence_score=0.9,
                reasoning=f"High memory utilization ({peak_util:.1f}%) risks out-of-memory errors",
                implementation_complexity="Medium"
            )
            return recommendation
        
        return None
    
    async def _optimize_gpu_resources(self, 
                                    current_usage: Dict[str, Any],
                                    workload_config: Dict[str, Any],
                                    performance_req: Dict[str, Any],
                                    cost_constraints: Dict[str, Any]) -> Optional[ResourceRecommendation]:
        """Optimize GPU resource allocation"""
        
        gpu_usage = current_usage.get("gpu", {})
        if not gpu_usage:
            return None
        
        current_util = gpu_usage.get("current_utilization", 50)
        
        # GPU optimization strategies
        if workload_config.get("duration") == "hours":
            # Training workload - consider spot instances
            recommendation = ResourceRecommendation(
                resource_id="gpu_spot_optimization",
                current_config={"type": "on_demand_gpu"},
                recommended_config={"type": "spot_gpu", "savings_strategy": "spot_instances"},
                expected_cost_savings=500.0,
                expected_performance_impact="Potential for interruption but significant cost savings",
                confidence_score=0.7,
                reasoning="Training workloads can benefit from spot GPU instances with 60-70% cost savings",
                implementation_complexity="Medium"
            )
            return recommendation
        
        elif current_util < 50:
            # GPU underutilized
            recommendation = ResourceRecommendation(
                resource_id="gpu_sharing",
                current_config={"utilization": current_util},
                recommended_config={"type": "shared_gpu", "sharing_strategy": "multi_tenancy"},
                expected_cost_savings=200.0,
                expected_performance_impact="Slight latency increase but better resource utilization",
                confidence_score=0.6,
                reasoning=f"GPU utilization is low ({current_util:.1f}%), consider GPU sharing",
                implementation_complexity="High"
            )
            return recommendation
        
        return None
    
    async def _optimize_storage_resources(self, 
                                        current_usage: Dict[str, Any],
                                        workload_config: Dict[str, Any],
                                        performance_req: Dict[str, Any],
                                        cost_constraints: Dict[str, Any]) -> Optional[ResourceRecommendation]:
        """Optimize storage resource allocation"""
        
        recommendation = ResourceRecommendation(
            resource_id="storage_tiering",
            current_config={"type": "standard_ssd"},
            recommended_config={
                "type": "tiered_storage",
                "hot_tier": "ssd",
                "warm_tier": "standard",
                "cold_tier": "archive"
            },
            expected_cost_savings=150.0,
            expected_performance_impact="Optimized for access patterns",
            confidence_score=0.8,
            reasoning="Implement storage tiering based on data access patterns",
            implementation_complexity="Medium"
        )
        
        return recommendation

class AutoScaler:
    """Intelligent auto-scaling system for ML workloads"""
    
    def __init__(self, config: AutoScalingConfig):
        self.config = config
        self.current_instances = config.min_instances
        self.scaling_history = []
        self.metrics_buffer = deque(maxlen=100)
        self.lock = threading.Lock()
        self.last_scaling_action = 0
    
    async def evaluate_scaling(self, current_metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Evaluate if scaling action is needed"""
        try:
            with self.lock:
                self.metrics_buffer.append({
                    "timestamp": time.time(),
                    "cpu_utilization": current_metrics.get("cpu_utilization", 0),
                    "memory_utilization": current_metrics.get("memory_utilization", 0),
                    "queue_length": current_metrics.get("queue_length", 0),
                    "response_time": current_metrics.get("response_time", 0)
                })
            
            # Check cooldown period
            if time.time() - self.last_scaling_action < self.config.cooldown_period_minutes * 60:
                return None
            
            # Calculate average metrics over recent period
            recent_metrics = list(self.metrics_buffer)[-10:]  # Last 10 data points
            
            if not recent_metrics:
                return None
            
            avg_cpu = statistics.mean([m["cpu_utilization"] for m in recent_metrics])
            avg_memory = statistics.mean([m["memory_utilization"] for m in recent_metrics])
            avg_queue = statistics.mean([m["queue_length"] for m in recent_metrics])
            
            # Scaling decision logic
            scaling_action = None
            
            # Scale up conditions
            if (avg_cpu > self.config.scale_up_threshold or 
                avg_memory > self.config.scale_up_threshold or
                avg_queue > 10):  # Queue threshold
                
                if self.current_instances < self.config.max_instances:
                    scaling_action = {
                        "action": "scale_up",
                        "from_instances": self.current_instances,
                        "to_instances": min(self.current_instances + 1, self.config.max_instances),
                        "reason": f"High utilization - CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%, Queue: {avg_queue:.1f}",
                        "confidence": self._calculate_scaling_confidence(recent_metrics, "up")
                    }
            
            # Scale down conditions
            elif (avg_cpu < self.config.scale_down_threshold and 
                  avg_memory < self.config.scale_down_threshold and
                  avg_queue < 2):
                
                if self.current_instances > self.config.min_instances:
                    scaling_action = {
                        "action": "scale_down",
                        "from_instances": self.current_instances,
                        "to_instances": max(self.current_instances - 1, self.config.min_instances),
                        "reason": f"Low utilization - CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%, Queue: {avg_queue:.1f}",
                        "confidence": self._calculate_scaling_confidence(recent_metrics, "down")
                    }
            
            if scaling_action:
                # Execute scaling action
                await self._execute_scaling_action(scaling_action)
                
                with self.lock:
                    self.scaling_history.append({
                        "timestamp": time.time(),
                        "action": scaling_action
                    })
                    self.last_scaling_action = time.time()
                
                return scaling_action
            
            return None
            
        except Exception as e:
            logger.error(f"Auto-scaling evaluation failed: {e}")
            return None
    
    def _calculate_scaling_confidence(self, recent_metrics: List[Dict[str, Any]], direction: str) -> float:
        """Calculate confidence score for scaling decision"""
        if len(recent_metrics) < 3:
            return 0.5
        
        # Check consistency of metrics trend
        cpu_values = [m["cpu_utilization"] for m in recent_metrics]
        memory_values = [m["memory_utilization"] for m in recent_metrics]
        
        cpu_trend = self._calculate_trend(cpu_values)
        memory_trend = self._calculate_trend(memory_values)
        
        if direction == "up":
            if cpu_trend == "increasing" and memory_trend == "increasing":
                return 0.9
            elif cpu_trend == "increasing" or memory_trend == "increasing":
                return 0.7
            else:
                return 0.5
        else:  # scale down
            if cpu_trend == "decreasing" and memory_trend == "decreasing":
                return 0.9
            elif cpu_trend == "stable" and memory_trend == "stable":
                return 0.7
            else:
                return 0.5
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 3:
            return "stable"
        
        increases = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        decreases = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
        
        if increases > decreases + 1:
            return "increasing"
        elif decreases > increases + 1:
            return "decreasing"
        else:
            return "stable"
    
    async def _execute_scaling_action(self, action: Dict[str, Any]):
        """Execute the scaling action"""
        try:
            # Update current instance count
            self.current_instances = action["to_instances"]
            
            # In production, this would call cloud provider APIs
            # For example, updating Kubernetes deployments or AWS Auto Scaling Groups
            logger.info(f"Executed scaling action: {action['action']} to {action['to_instances']} instances")
            
            # Mock implementation
            if action["action"] == "scale_up":
                await self._provision_instances(action["to_instances"] - action["from_instances"])
            else:
                await self._terminate_instances(action["from_instances"] - action["to_instances"])
            
        except Exception as e:
            logger.error(f"Failed to execute scaling action: {e}")
            raise
    
    async def _provision_instances(self, count: int):
        """Provision new instances"""
        # Mock implementation - in production, call cloud provider APIs
        logger.info(f"Provisioning {count} new instances")
        await asyncio.sleep(1)  # Simulate provisioning time
    
    async def _terminate_instances(self, count: int):
        """Terminate instances"""
        # Mock implementation - in production, call cloud provider APIs
        logger.info(f"Terminating {count} instances")
        await asyncio.sleep(1)  # Simulate termination time

class CostOptimizationFramework:
    """Complete cost optimization framework"""
    
    def __init__(self):
        self.cost_tracker = CloudCostTracker()
        self.resource_optimizer = ResourceOptimizer(self.cost_tracker)
        self.auto_scalers = {}
        self.optimization_scheduler = None
        self.is_running = False
        
        # Cost monitoring
        self.cost_alerts = []
        self.budget_limits = {}
        
        # Recommendations tracking
        self.active_recommendations = []
        self.implemented_recommendations = []
    
    async def initialize(self):
        """Initialize the cost optimization framework"""
        try:
            logger.info("Initializing cost optimization framework...")
            
            # Setup default auto-scaling configurations
            default_config = AutoScalingConfig(
                min_instances=1,
                max_instances=10,
                target_cpu_utilization=70.0,
                target_memory_utilization=80.0,
                scale_up_threshold=80.0,
                scale_down_threshold=30.0,
                cooldown_period_minutes=5,
                enable_predictive_scaling=False
            )
            
            self.auto_scalers["inference"] = AutoScaler(default_config)
            
            # Setup budget alerts
            self.budget_limits = {
                "daily": 1000.0,
                "weekly": 5000.0,
                "monthly": 20000.0
            }
            
            logger.info("✅ Cost optimization framework initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize cost optimization framework: {e}")
            raise
    
    async def start_monitoring(self):
        """Start continuous cost monitoring and optimization"""
        try:
            self.is_running = True
            logger.info("Starting cost monitoring...")
            
            # Start monitoring tasks
            tasks = [
                asyncio.create_task(self._cost_monitoring_loop()),
                asyncio.create_task(self._auto_scaling_loop()),
                asyncio.create_task(self._optimization_scheduler_loop())
            ]
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Cost monitoring failed: {e}")
            raise
        finally:
            self.is_running = False
    
    async def _cost_monitoring_loop(self):
        """Continuous cost monitoring loop"""
        while self.is_running:
            try:
                # Simulate cost data collection
                await self._collect_cost_data()
                
                # Check budget alerts
                await self._check_budget_alerts()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Cost monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _auto_scaling_loop(self):
        """Auto-scaling monitoring loop"""
        while self.is_running:
            try:
                # Simulate current metrics
                current_metrics = {
                    "cpu_utilization": np.random.normal(60, 20),
                    "memory_utilization": np.random.normal(55, 15),
                    "queue_length": np.random.poisson(5),
                    "response_time": np.random.exponential(0.1)
                }
                
                # Evaluate scaling for each auto-scaler
                for service_name, auto_scaler in self.auto_scalers.items():
                    scaling_action = await auto_scaler.evaluate_scaling(current_metrics)
                    
                    if scaling_action:
                        logger.info(f"Auto-scaling action for {service_name}: {scaling_action}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(60)
    
    async def _optimization_scheduler_loop(self):
        """Optimization scheduler loop"""
        while self.is_running:
            try:
                # Run daily optimization analysis
                recommendations = await self.resource_optimizer.optimize_resources(
                    workload_type="inference",
                    performance_requirements={"target_cpu_utilization": 70},
                    cost_constraints={"max_monthly_cost": 10000}
                )
                
                if recommendations:
                    logger.info(f"Generated {len(recommendations)} optimization recommendations")
                    self.active_recommendations.extend(recommendations)
                
                # Sleep for 24 hours
                await asyncio.sleep(24 * 3600)
                
            except Exception as e:
                logger.error(f"Optimization scheduler error: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def _collect_cost_data(self):
        """Collect cost data from cloud providers"""
        # Simulate cost data collection
        regions = ["us-east-1", "us-west-2"]
        instance_types = ["t3.micro", "t3.small", "m5.large", "c5.xlarge"]
        
        for region in regions:
            for instance_type in instance_types:
                # Simulate resource usage
                usage = ResourceUsage(
                    timestamp=time.time(),
                    resource_type=ResourceType.CPU,
                    usage_percent=np.random.uniform(20, 90),
                    allocated_amount=100.0,
                    used_amount=np.random.uniform(20, 90),
                    cost_per_hour=self.cost_tracker.instance_pricing[region][instance_type],
                    instance_id=f"{instance_type}-{region}-{int(time.time() % 1000)}",
                    region=region
                )
                
                self.cost_tracker.record_usage(usage)
    
    async def _check_budget_alerts(self):
        """Check budget limits and generate alerts"""
        try:
            now = datetime.utcnow()
            
            # Check daily budget
            daily_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            daily_analysis = self.cost_tracker.analyze_costs(daily_start, now)
            
            if daily_analysis.total_cost > self.budget_limits["daily"]:
                alert = {
                    "type": "budget_exceeded",
                    "period": "daily",
                    "current_cost": daily_analysis.total_cost,
                    "budget_limit": self.budget_limits["daily"],
                    "timestamp": time.time()
                }
                self.cost_alerts.append(alert)
                logger.warning(f"Daily budget exceeded: ${daily_analysis.total_cost:.2f} > ${self.budget_limits['daily']:.2f}")
            
        except Exception as e:
            logger.error(f"Budget alert check failed: {e}")
    
    async def get_cost_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive cost dashboard data"""
        try:
            now = datetime.utcnow()
            
            # Different time periods
            periods = {
                "today": now.replace(hour=0, minute=0, second=0, microsecond=0),
                "this_week": now - timedelta(days=7),
                "this_month": now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            }
            
            dashboard = {
                "timestamp": now.isoformat(),
                "cost_analysis": {},
                "active_recommendations": len(self.active_recommendations),
                "implemented_recommendations": len(self.implemented_recommendations),
                "auto_scaling_status": {},
                "budget_status": {},
                "recent_alerts": self.cost_alerts[-10:]  # Last 10 alerts
            }
            
            # Cost analysis for each period
            for period_name, start_date in periods.items():
                analysis = self.cost_tracker.analyze_costs(start_date, now)
                dashboard["cost_analysis"][period_name] = {
                    "total_cost": analysis.total_cost,
                    "cost_breakdown": analysis.cost_breakdown,
                    "projected_savings": analysis.projected_savings,
                    "utilization": analysis.resource_utilization
                }
            
            # Auto-scaling status
            for service_name, auto_scaler in self.auto_scalers.items():
                dashboard["auto_scaling_status"][service_name] = {
                    "current_instances": auto_scaler.current_instances,
                    "min_instances": auto_scaler.config.min_instances,
                    "max_instances": auto_scaler.config.max_instances,
                    "recent_actions": len(auto_scaler.scaling_history)
                }
            
            # Budget status
            for period, limit in self.budget_limits.items():
                if period in dashboard["cost_analysis"]:
                    current_cost = dashboard["cost_analysis"][period]["total_cost"]
                    dashboard["budget_status"][period] = {
                        "current": current_cost,
                        "limit": limit,
                        "usage_percentage": (current_cost / limit) * 100,
                        "remaining": limit - current_cost
                    }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to generate cost dashboard: {e}")
            raise
    
    async def export_cost_report(self, output_path: str = None) -> str:
        """Export comprehensive cost report"""
        try:
            dashboard = await self.get_cost_dashboard()
            
            # Add detailed recommendations
            detailed_recommendations = []
            for rec in self.active_recommendations:
                detailed_recommendations.append(asdict(rec))
            
            report = {
                "report_timestamp": datetime.utcnow().isoformat(),
                "executive_summary": {
                    "total_monthly_cost": dashboard["cost_analysis"]["this_month"]["total_cost"],
                    "potential_savings": dashboard["cost_analysis"]["this_month"]["projected_savings"],
                    "optimization_opportunities": len(self.active_recommendations)
                },
                "cost_dashboard": dashboard,
                "detailed_recommendations": detailed_recommendations,
                "optimization_history": self.implemented_recommendations[-20:]  # Last 20 implementations
            }
            
            if not output_path:
                output_path = f"cost_optimization_report_{int(time.time())}.json"
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Cost optimization report exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export cost report: {e}")
            raise

def main():
    """Main function for cost optimization demonstration"""
    
    async def run_cost_optimization_demo():
        # Initialize framework
        cost_framework = CostOptimizationFramework()
        await cost_framework.initialize()
        
        print("💰 MLOps Cost Optimization Framework Demo")
        print("=" * 50)
        
        # Simulate some cost data
        print("\n1. Collecting cost data...")
        for _ in range(20):
            await cost_framework._collect_cost_data()
        
        # Analyze costs
        print("\n2. Analyzing costs...")
        now = datetime.utcnow()
        start_date = now - timedelta(hours=1)
        
        analysis = cost_framework.cost_tracker.analyze_costs(start_date, now)
        
        print(f"Total cost (last hour): ${analysis.total_cost:.2f}")
        print(f"Cost breakdown: {analysis.cost_breakdown}")
        print(f"Projected savings: ${analysis.projected_savings:.2f}")
        
        # Generate recommendations
        print("\n3. Generating optimization recommendations...")
        recommendations = await cost_framework.resource_optimizer.optimize_resources(
            workload_type="inference",
            performance_requirements={"target_cpu_utilization": 70},
            cost_constraints={"max_monthly_cost": 10000}
        )
        
        print(f"Generated {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec.reasoning}")
            print(f"     Expected savings: ${rec.expected_cost_savings:.2f}")
            print(f"     Confidence: {rec.confidence_score:.2f}")
        
        # Test auto-scaling
        print("\n4. Testing auto-scaling...")
        auto_scaler = cost_framework.auto_scalers["inference"]
        
        # Simulate high load
        high_load_metrics = {
            "cpu_utilization": 85,
            "memory_utilization": 90,
            "queue_length": 15,
            "response_time": 0.5
        }
        
        scaling_action = await auto_scaler.evaluate_scaling(high_load_metrics)
        
        if scaling_action:
            print(f"Auto-scaling triggered: {scaling_action['action']}")
            print(f"Scaling from {scaling_action['from_instances']} to {scaling_action['to_instances']} instances")
            print(f"Reason: {scaling_action['reason']}")
        else:
            print("No scaling action needed")
        
        # Generate cost dashboard
        print("\n5. Generating cost dashboard...")
        dashboard = await cost_framework.get_cost_dashboard()
        
        print("Cost Summary:")
        for period, data in dashboard["cost_analysis"].items():
            if data["total_cost"] > 0:
                print(f"  {period.title()}: ${data['total_cost']:.2f}")
        
        print(f"\nActive recommendations: {dashboard['active_recommendations']}")
        print(f"Budget status: {dashboard['budget_status']}")
        
        # Export report
        print("\n6. Exporting cost report...")
        report_path = await cost_framework.export_cost_report()
        print(f"Cost report exported to: {report_path}")
    
    # Run demo
    asyncio.run(run_cost_optimization_demo())

if __name__ == "__main__":
    main()