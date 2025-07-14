"""
Data Structures and Algorithms for MLOps Engineers
Focus on algorithms commonly used in ML systems and infrastructure
"""

from typing import List, Dict, Optional, Tuple, Any
import heapq
import hashlib
import time
import threading
import queue
from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np
import random

# ==============================================================================
# PROBLEM 1: CONSISTENT HASHING FOR MODEL SERVING
# ==============================================================================

class ConsistentHash:
    """
    Implement consistent hashing for distributing model serving requests
    across multiple servers with minimal reshuffling when servers are added/removed.
    
    Use case: Load balancing ML model requests across GPU servers
    """
    
    def __init__(self, replicas: int = 150):
        self.replicas = replicas
        self.ring = {}  # hash -> server
        self.sorted_hashes = []
        
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_server(self, server: str) -> None:
        """
        Add a server to the hash ring
        
        TODO: Implement this method
        1. Create multiple virtual nodes for the server
        2. Add them to the ring
        3. Keep sorted_hashes updated
        """
        pass
    
    def remove_server(self, server: str) -> None:
        """
        Remove a server from the hash ring
        
        TODO: Implement this method
        1. Remove all virtual nodes for the server
        2. Update sorted_hashes
        """
        pass
    
    def get_server(self, key: str) -> Optional[str]:
        """
        Get the server responsible for a given key
        
        TODO: Implement this method
        1. Hash the key
        2. Find the next server in clockwise direction
        3. Handle wraparound case
        """
        pass
    
    def get_servers_for_key(self, key: str, count: int = 3) -> List[str]:
        """
        Get multiple servers for replication
        Used for fault tolerance in model serving
        """
        # TODO: Implement to return 'count' unique servers
        pass

# ==============================================================================
# PROBLEM 2: LRU CACHE FOR MODEL ARTIFACTS
# ==============================================================================

class LRUCache:
    """
    Implement LRU cache for storing frequently accessed model artifacts
    
    Use case: Cache model weights, feature data, or inference results
    """
    
    class Node:
        def __init__(self, key=None, value=None):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> node
        
        # Create dummy head and tail nodes
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node):
        """Add node right after head"""
        # TODO: Implement doubly linked list insertion
        pass
    
    def _remove_node(self, node):
        """Remove an existing node from linked list"""
        # TODO: Implement doubly linked list removal
        pass
    
    def _move_to_head(self, node):
        """Move node to head (mark as recently used)"""
        # TODO: Implement move operation
        pass
    
    def _pop_tail(self):
        """Remove last node (least recently used)"""
        # TODO: Implement tail removal
        pass
    
    def get(self, key: str) -> Any:
        """
        Get value from cache and mark as recently used
        
        TODO: Implement get operation
        1. Check if key exists
        2. Move to head if exists
        3. Return value or None
        """
        pass
    
    def put(self, key: str, value: Any) -> None:
        """
        Put key-value pair in cache
        
        TODO: Implement put operation
        1. Check if key already exists
        2. If at capacity, remove LRU item
        3. Add new item to head
        """
        pass

# ==============================================================================
# PROBLEM 3: RATE LIMITER FOR API ENDPOINTS
# ==============================================================================

class TokenBucketRateLimiter:
    """
    Implement token bucket rate limiter for ML API endpoints
    
    Use case: Protect model serving APIs from excessive traffic
    """
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        capacity: Maximum number of tokens
        refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def _refill(self) -> None:
        """
        Refill tokens based on elapsed time
        
        TODO: Implement token refill logic
        1. Calculate elapsed time
        2. Add tokens based on refill rate
        3. Cap at capacity
        """
        pass
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens
        
        TODO: Implement token consumption
        1. Refill tokens first
        2. Check if enough tokens available
        3. Consume tokens if available
        4. Thread-safe implementation
        """
        pass
    
    def get_available_tokens(self) -> int:
        """Get current number of available tokens"""
        with self.lock:
            self._refill()
            return int(self.tokens)

# ==============================================================================
# PROBLEM 4: PRIORITY QUEUE FOR MODEL TRAINING JOBS
# ==============================================================================

@dataclass
class TrainingJob:
    job_id: str
    priority: int  # Lower number = higher priority
    gpu_hours: float
    created_at: float
    
    def __lt__(self, other):
        # For heapq - prioritize by priority, then by creation time
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at

class TrainingJobScheduler:
    """
    Priority queue scheduler for ML training jobs
    
    Use case: Schedule training jobs based on priority and resource requirements
    """
    
    def __init__(self):
        self.job_queue = []  # min-heap
        self.job_lookup = {}  # job_id -> job
        self.running_jobs = {}  # job_id -> start_time
        self.completed_jobs = []
        
    def submit_job(self, job: TrainingJob) -> None:
        """
        Submit a new training job
        
        TODO: Implement job submission
        1. Add to priority queue
        2. Add to lookup table
        """
        pass
    
    def get_next_job(self) -> Optional[TrainingJob]:
        """
        Get the highest priority job
        
        TODO: Implement job retrieval
        1. Pop from heap
        2. Update lookup table
        3. Handle empty queue
        """
        pass
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job
        
        TODO: Implement job cancellation
        1. Remove from lookup
        2. Mark as cancelled (can't remove from heap efficiently)
        """
        pass
    
    def update_priority(self, job_id: str, new_priority: int) -> bool:
        """
        Update job priority
        
        TODO: Implement priority update
        This is tricky with heapq - consider rebuild or lazy deletion
        """
        pass

# ==============================================================================
# PROBLEM 5: BLOOM FILTER FOR DUPLICATE DETECTION
# ==============================================================================

class BloomFilter:
    """
    Implement Bloom filter for efficient duplicate detection
    
    Use case: Detect duplicate training data or feature vectors
    """
    
    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        """
        Calculate optimal size and number of hash functions
        
        TODO: Implement size calculation
        m = -(n * ln(p)) / (ln(2)^2)
        k = (m/n) * ln(2)
        """
        self.expected_items = expected_items
        self.false_positive_rate = false_positive_rate
        
        # Calculate optimal parameters
        self.size = self._calculate_size()
        self.hash_count = self._calculate_hash_count()
        
        # Bit array
        self.bit_array = [False] * self.size
        self.items_added = 0
    
    def _calculate_size(self) -> int:
        """Calculate optimal bit array size"""
        # TODO: Implement optimal size calculation
        pass
    
    def _calculate_hash_count(self) -> int:
        """Calculate optimal number of hash functions"""
        # TODO: Implement optimal hash count calculation
        pass
    
    def _hash(self, item: str, seed: int) -> int:
        """Hash function with seed"""
        # TODO: Implement hash function with different seeds
        pass
    
    def add(self, item: str) -> None:
        """
        Add item to bloom filter
        
        TODO: Implement add operation
        1. Generate k hash values
        2. Set corresponding bits
        """
        pass
    
    def contains(self, item: str) -> bool:
        """
        Check if item might be in the set
        
        TODO: Implement contains check
        1. Generate k hash values
        2. Check if all corresponding bits are set
        """
        pass
    
    def estimated_false_positive_rate(self) -> float:
        """Calculate current false positive rate"""
        # TODO: Implement FPR estimation based on current state
        pass

# ==============================================================================
# PROBLEM 6: SLIDING WINDOW FOR METRICS CALCULATION
# ==============================================================================

class SlidingWindowMetrics:
    """
    Implement sliding window for real-time metrics calculation
    
    Use case: Calculate moving averages, error rates, throughput over time windows
    """
    
    def __init__(self, window_size_seconds: int):
        self.window_size = window_size_seconds
        self.data_points = deque()  # (timestamp, value) tuples
        self.sum = 0.0
        self.count = 0
    
    def add_data_point(self, value: float, timestamp: float = None) -> None:
        """
        Add a new data point
        
        TODO: Implement data point addition
        1. Use current time if timestamp not provided
        2. Add to deque
        3. Remove old data points outside window
        4. Update sum and count
        """
        pass
    
    def _remove_old_data(self, current_time: float) -> None:
        """Remove data points outside the window"""
        # TODO: Implement old data removal
        pass
    
    def get_average(self) -> float:
        """Get average over current window"""
        # TODO: Implement average calculation
        pass
    
    def get_count(self) -> int:
        """Get count of data points in current window"""
        return self.count
    
    def get_rate_per_second(self) -> float:
        """Get rate (count per second) over current window"""
        # TODO: Implement rate calculation
        pass

# ==============================================================================
# PROBLEM 7: DISTRIBUTED LOCKING FOR MODEL UPDATES
# ==============================================================================

class DistributedLock:
    """
    Implement distributed locking for coordinating model updates across servers
    
    Use case: Ensure only one process updates shared model artifacts at a time
    """
    
    def __init__(self, lock_name: str, timeout: int = 30):
        self.lock_name = lock_name
        self.timeout = timeout
        self.lock_id = None
        self.acquired = False
    
    def acquire(self) -> bool:
        """
        Acquire distributed lock
        
        TODO: Implement lock acquisition
        This would typically use Redis, ZooKeeper, or etcd
        For this exercise, simulate with local state
        """
        pass
    
    def release(self) -> bool:
        """
        Release distributed lock
        
        TODO: Implement lock release
        1. Verify we own the lock
        2. Release the lock
        3. Update local state
        """
        pass
    
    def extend_lock(self, additional_time: int) -> bool:
        """
        Extend lock timeout
        
        TODO: Implement lock extension
        Useful for long-running operations
        """
        pass
    
    def __enter__(self):
        """Context manager entry"""
        if self.acquire():
            return self
        else:
            raise Exception(f"Failed to acquire lock: {self.lock_name}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()

# ==============================================================================
# PROBLEM 8: GRAPH ALGORITHMS FOR DEPENDENCY RESOLUTION
# ==============================================================================

class DAGScheduler:
    """
    Implement topological sorting for ML pipeline dependency resolution
    
    Use case: Schedule ML pipeline tasks based on dependencies
    """
    
    def __init__(self):
        self.graph = defaultdict(list)  # adjacency list
        self.in_degree = defaultdict(int)
        self.nodes = set()
    
    def add_dependency(self, task: str, depends_on: str) -> None:
        """
        Add dependency: task depends on depends_on
        
        TODO: Implement dependency addition
        1. Add edge to graph
        2. Update in-degree count
        3. Track all nodes
        """
        pass
    
    def topological_sort(self) -> List[str]:
        """
        Return tasks in topological order (dependencies first)
        
        TODO: Implement Kahn's algorithm
        1. Find nodes with in-degree 0
        2. Process nodes and reduce in-degree of neighbors
        3. Detect cycles if any
        """
        pass
    
    def find_parallel_tasks(self) -> List[List[str]]:
        """
        Group tasks that can run in parallel
        
        TODO: Implement parallel task grouping
        Return list of task groups that can run simultaneously
        """
        pass
    
    def detect_cycle(self) -> bool:
        """
        Detect if there's a cycle in dependencies
        
        TODO: Implement cycle detection using DFS
        """
        pass

# ==============================================================================
# PROBLEM 9: LOAD BALANCING ALGORITHMS
# ==============================================================================

class LoadBalancer:
    """
    Implement various load balancing algorithms for model serving
    """
    
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.current_index = 0  # for round robin
        self.server_weights = {server: 1.0 for server in servers}
        self.server_connections = {server: 0 for server in servers}
        
    def round_robin(self) -> str:
        """
        Round robin load balancing
        
        TODO: Implement round robin algorithm
        """
        pass
    
    def weighted_round_robin(self) -> str:
        """
        Weighted round robin based on server capacity
        
        TODO: Implement weighted round robin
        """
        pass
    
    def least_connections(self) -> str:
        """
        Route to server with least active connections
        
        TODO: Implement least connections algorithm
        """
        pass
    
    def consistent_hash_routing(self, client_id: str) -> str:
        """
        Route based on consistent hashing (sticky sessions)
        
        TODO: Use ConsistentHash class implemented earlier
        """
        pass
    
    def update_server_weight(self, server: str, weight: float) -> None:
        """Update server weight based on performance metrics"""
        self.server_weights[server] = weight
    
    def add_connection(self, server: str) -> None:
        """Track new connection to server"""
        self.server_connections[server] += 1
    
    def remove_connection(self, server: str) -> None:
        """Track connection removal from server"""
        self.server_connections[server] = max(0, self.server_connections[server] - 1)

# ==============================================================================
# PROBLEM 10: FEATURE HASHING FOR DIMENSIONALITY REDUCTION
# ==============================================================================

class FeatureHasher:
    """
    Implement feature hashing (hashing trick) for high-dimensional sparse features
    
    Use case: Convert categorical features to fixed-size numeric vectors
    """
    
    def __init__(self, num_features: int, hash_function: str = 'murmurhash'):
        self.num_features = num_features
        self.hash_function = hash_function
    
    def _hash(self, feature: str) -> int:
        """
        Hash feature to index
        
        TODO: Implement hash function
        1. Hash feature string
        2. Map to valid index range
        """
        pass
    
    def _sign_hash(self, feature: str) -> int:
        """
        Hash feature to sign (+1 or -1)
        Used to reduce hash collisions impact
        
        TODO: Implement sign hash
        """
        pass
    
    def transform(self, features: List[str]) -> List[float]:
        """
        Transform list of categorical features to numeric vector
        
        TODO: Implement feature hashing
        1. Initialize zero vector
        2. For each feature, hash to index and sign
        3. Add/subtract 1 based on sign
        """
        pass
    
    def transform_dict(self, feature_dict: Dict[str, float]) -> List[float]:
        """
        Transform dictionary of features with values
        
        TODO: Implement weighted feature hashing
        Similar to above but use feature values instead of 1
        """
        pass

# ==============================================================================
# INTERVIEW QUESTIONS AND TESTING
# ==============================================================================

def test_algorithms():
    """
    Test all implemented algorithms
    This would be used in interviews to verify correctness
    """
    
    print("Testing Data Structures and Algorithms for MLOps...")
    
    # Test Consistent Hashing
    print("\n1. Testing Consistent Hashing:")
    ch = ConsistentHash()
    servers = ["server1", "server2", "server3"]
    for server in servers:
        ch.add_server(server)
    
    # Test key distribution
    keys = [f"user_{i}" for i in range(1000)]
    distribution = {}
    for key in keys:
        server = ch.get_server(key)
        distribution[server] = distribution.get(server, 0) + 1
    
    print(f"Key distribution: {distribution}")
    
    # Test LRU Cache
    print("\n2. Testing LRU Cache:")
    cache = LRUCache(3)
    cache.put("model1", "weights1")
    cache.put("model2", "weights2")
    cache.put("model3", "weights3")
    print(f"Get model1: {cache.get('model1')}")
    cache.put("model4", "weights4")  # Should evict model2
    print(f"Get model2: {cache.get('model2')}")  # Should return None
    
    # Test Rate Limiter
    print("\n3. Testing Rate Limiter:")
    limiter = TokenBucketRateLimiter(capacity=10, refill_rate=2)
    
    # Consume tokens
    for i in range(5):
        result = limiter.consume(2)
        print(f"Consume 2 tokens: {result}, Available: {limiter.get_available_tokens()}")
    
    # More tests for other algorithms...

"""
INTERVIEW QUESTIONS FOR EACH ALGORITHM:

CONSISTENT HASHING:
Q: How does consistent hashing minimize data movement when servers are added/removed?
Q: What are virtual nodes and why are they important?
Q: How would you handle hotspots in consistent hashing?

LRU CACHE:
Q: What's the time complexity of get and put operations?
Q: How would you make this thread-safe?
Q: What are alternatives to LRU (LFU, ARC, etc.)?

RATE LIMITER:
Q: Compare token bucket vs leaky bucket algorithms
Q: How would you implement distributed rate limiting?
Q: What are the trade-offs between accuracy and performance?

PRIORITY QUEUE:
Q: How do you handle priority updates efficiently?
Q: What happens if two jobs have the same priority?
Q: How would you implement fair scheduling?

BLOOM FILTER:
Q: What are the trade-offs between space and false positive rate?
Q: When would you choose Bloom filter over hash set?
Q: How do you handle deletions in Bloom filters?

SLIDING WINDOW:
Q: How do you optimize for memory usage with large windows?
Q: What's the difference between tumbling and sliding windows?
Q: How do you handle out-of-order data?

DISTRIBUTED LOCKING:
Q: How do you prevent deadlocks in distributed systems?
Q: What happens if a process holding a lock crashes?
Q: Compare different consensus algorithms (Raft, Paxos)

DAG SCHEDULING:
Q: How do you optimize parallel execution of independent tasks?
Q: What happens when a task fails in the middle of execution?
Q: How do you handle dynamic dependencies?

LOAD BALANCING:
Q: When would you use each load balancing algorithm?
Q: How do you handle server failures?
Q: What metrics would you use to evaluate load balancer performance?

FEATURE HASHING:
Q: How does feature hashing handle hash collisions?
Q: What are the trade-offs between hash space size and collision rate?
Q: How do you choose the right hash function?
"""

if __name__ == "__main__":
    test_algorithms()