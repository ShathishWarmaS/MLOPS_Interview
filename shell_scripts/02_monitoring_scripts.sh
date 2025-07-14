#!/bin/bash

# ==============================================================================
# MLOps Monitoring and Alerting Scripts
# Challenge: Implement comprehensive monitoring for ML systems
# ==============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITORING_CONFIG="${SCRIPT_DIR}/monitoring.conf"
METRICS_DIR="${SCRIPT_DIR}/metrics"
ALERTS_DIR="${SCRIPT_DIR}/alerts"
LOG_FILE="${SCRIPT_DIR}/monitoring.log"

# Create directories
mkdir -p "$METRICS_DIR" "$ALERTS_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ==============================================================================
# LOGGING UTILITIES
# ==============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_metric() {
    local metric_name="$1"
    local value="$2"
    local timestamp="${3:-$(date +%s)}"
    
    echo "${timestamp},${metric_name},${value}" >> "${METRICS_DIR}/${metric_name}.csv"
}

# ==============================================================================
# SYSTEM METRICS COLLECTION
# ==============================================================================

collect_system_metrics() {
    log "Collecting system metrics..."
    
    # CPU Usage
    local cpu_usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    log_metric "cpu_usage" "$cpu_usage"
    
    # Memory Usage
    local memory_usage
    memory_usage=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')
    log_metric "memory_usage" "$memory_usage"
    
    # Disk Usage
    local disk_usage
    disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    log_metric "disk_usage" "$disk_usage"
    
    # Load Average
    local load_avg
    load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    log_metric "load_average" "$load_avg"
    
    log "System metrics collected: CPU: ${cpu_usage}%, Memory: ${memory_usage}%, Disk: ${disk_usage}%"
}

# ==============================================================================
# MODEL PERFORMANCE MONITORING
# ==============================================================================

check_model_latency() {
    local endpoint="$1"
    local timeout="${2:-30}"
    
    log "Checking model latency for endpoint: $endpoint"
    
    # Send test request and measure response time
    local start_time response_time http_code
    start_time=$(date +%s.%N)
    
    # TODO: Implement actual model prediction request
    # This should send a real prediction request to your model
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
                     --max-time "$timeout" \
                     -X POST \
                     -H "Content-Type: application/json" \
                     -d '{"features": [1,2,3,4,5]}' \
                     "$endpoint/predict" || echo "000")
    
    local end_time
    end_time=$(date +%s.%N)
    response_time=$(echo "$end_time - $start_time" | bc)
    
    if [[ "$http_code" == "200" ]]; then
        log_metric "model_latency_ms" "$(echo "$response_time * 1000" | bc)"
        log_metric "model_availability" "1"
        log "Model latency: ${response_time}s"
        return 0
    else
        log_metric "model_availability" "0"
        log "Model endpoint unavailable (HTTP: $http_code)"
        return 1
    fi
}

check_model_accuracy() {
    local model_name="$1"
    local test_data_path="$2"
    
    log "Checking model accuracy for: $model_name"
    
    # TODO: Implement accuracy calculation
    # This should:
    # 1. Load ground truth data
    # 2. Get model predictions
    # 3. Calculate accuracy metrics
    # 4. Compare with thresholds
    
    # Mock implementation
    local accuracy
    accuracy=$(echo "scale=4; $RANDOM / 32767 * 0.1 + 0.9" | bc)  # Random accuracy between 0.9-1.0
    
    log_metric "model_accuracy" "$accuracy"
    log "Model accuracy: $accuracy"
    
    # Check if accuracy is below threshold
    local threshold=0.85
    if (( $(echo "$accuracy < $threshold" | bc -l) )); then
        generate_alert "model_accuracy_low" "Model accuracy ($accuracy) below threshold ($threshold)"
        return 1
    fi
    
    return 0
}

monitor_prediction_drift() {
    local model_name="$1"
    local reference_data="$2"
    local current_data="$3"
    
    log "Monitoring prediction drift for: $model_name"
    
    # TODO: Implement drift detection
    # Methods to implement:
    # 1. Statistical tests (KS test, Chi-square)
    # 2. Distribution comparison
    # 3. Feature importance drift
    # 4. Population stability index (PSI)
    
    # Mock implementation - calculate simple mean difference
    if [[ -f "$reference_data" && -f "$current_data" ]]; then
        local ref_mean curr_mean drift_score
        ref_mean=$(awk '{sum+=$1} END {print sum/NR}' "$reference_data" 2>/dev/null || echo "0")
        curr_mean=$(awk '{sum+=$1} END {print sum/NR}' "$current_data" 2>/dev/null || echo "0")
        drift_score=$(echo "scale=4; ($curr_mean - $ref_mean) / $ref_mean" | bc -l 2>/dev/null || echo "0")
        
        log_metric "prediction_drift" "${drift_score#-}"  # Remove negative sign
        log "Prediction drift score: $drift_score"
        
        # Check drift threshold
        local drift_threshold=0.1
        if (( $(echo "${drift_score#-} > $drift_threshold" | bc -l) )); then
            generate_alert "prediction_drift" "Prediction drift detected: $drift_score"
            return 1
        fi
    else
        log "Warning: Reference or current data files not found"
    fi
    
    return 0
}

# ==============================================================================
# DATA QUALITY MONITORING
# ==============================================================================

check_data_freshness() {
    local data_source="$1"
    local max_age_hours="${2:-24}"
    
    log "Checking data freshness for: $data_source"
    
    # TODO: Implement for different data sources
    # - Database tables
    # - File systems
    # - S3 buckets
    # - Streaming sources
    
    case "$data_source" in
        file:*)
            local file_path="${data_source#file:}"
            check_file_freshness "$file_path" "$max_age_hours"
            ;;
        s3:*)
            local s3_path="${data_source#s3:}"
            check_s3_freshness "$s3_path" "$max_age_hours"
            ;;
        db:*)
            local db_table="${data_source#db:}"
            check_db_freshness "$db_table" "$max_age_hours"
            ;;
        *)
            log "Unknown data source type: $data_source"
            return 1
            ;;
    esac
}

check_file_freshness() {
    local file_path="$1"
    local max_age_hours="$2"
    
    if [[ ! -f "$file_path" ]]; then
        generate_alert "data_missing" "Data file not found: $file_path"
        return 1
    fi
    
    local file_age_hours
    file_age_hours=$(echo "($(date +%s) - $(stat -c %Y "$file_path")) / 3600" | bc)
    
    log_metric "data_age_hours" "$file_age_hours"
    
    if (( file_age_hours > max_age_hours )); then
        generate_alert "data_stale" "Data file is stale: ${file_age_hours}h old (max: ${max_age_hours}h)"
        return 1
    fi
    
    log "Data freshness OK: ${file_age_hours}h old"
    return 0
}

check_s3_freshness() {
    local s3_path="$1"
    local max_age_hours="$2"
    
    # TODO: Implement S3 freshness check
    # Use AWS CLI to check object modification time
    
    log "Checking S3 freshness for: $s3_path"
    
    if command -v aws &> /dev/null; then
        local last_modified
        last_modified=$(aws s3api head-object --bucket "${s3_path%%/*}" --key "${s3_path#*/}" \
                       --query 'LastModified' --output text 2>/dev/null || echo "")
        
        if [[ -n "$last_modified" ]]; then
            local age_hours
            age_hours=$(echo "($(date +%s) - $(date -d "$last_modified" +%s)) / 3600" | bc)
            log_metric "s3_data_age_hours" "$age_hours"
            
            if (( age_hours > max_age_hours )); then
                generate_alert "s3_data_stale" "S3 data is stale: ${age_hours}h old"
                return 1
            fi
        else
            generate_alert "s3_data_missing" "S3 object not found: $s3_path"
            return 1
        fi
    else
        log "AWS CLI not available, skipping S3 check"
    fi
    
    return 0
}

validate_data_schema() {
    local data_file="$1"
    local schema_file="$2"
    
    log "Validating data schema for: $data_file"
    
    # TODO: Implement schema validation
    # Methods:
    # 1. JSON Schema validation
    # 2. Column count and type checks
    # 3. Required field validation
    # 4. Range and format validation
    
    # Mock implementation - check if file exists and has expected columns
    if [[ ! -f "$data_file" ]]; then
        generate_alert "schema_validation_failed" "Data file not found: $data_file"
        return 1
    fi
    
    # Simple CSV column count check
    if [[ "$data_file" == *.csv ]]; then
        local actual_columns expected_columns
        actual_columns=$(head -1 "$data_file" | tr ',' '\n' | wc -l)
        expected_columns=$(grep -c "column" "$schema_file" 2>/dev/null || echo "0")
        
        if [[ "$expected_columns" -gt 0 && "$actual_columns" -ne "$expected_columns" ]]; then
            generate_alert "schema_mismatch" "Column count mismatch: expected $expected_columns, got $actual_columns"
            return 1
        fi
    fi
    
    log "Schema validation passed"
    return 0
}

# ==============================================================================
# KUBERNETES MONITORING
# ==============================================================================

monitor_kubernetes_resources() {
    local namespace="${1:-default}"
    
    log "Monitoring Kubernetes resources in namespace: $namespace"
    
    # Check pod status
    check_pod_status "$namespace"
    
    # Check resource usage
    check_resource_usage "$namespace"
    
    # Check service availability
    check_service_availability "$namespace"
    
    # Check HPA status
    check_hpa_status "$namespace"
}

check_pod_status() {
    local namespace="$1"
    
    # Get pod status
    local pod_status
    pod_status=$(kubectl get pods -n "$namespace" -o json 2>/dev/null | \
                jq -r '.items[] | select(.metadata.labels.app=="ml-model") | 
                       .status.phase' 2>/dev/null || echo "")
    
    if [[ -n "$pod_status" ]]; then
        local running_pods failed_pods
        running_pods=$(echo "$pod_status" | grep -c "Running" || echo "0")
        failed_pods=$(echo "$pod_status" | grep -c -E "(Failed|Error)" || echo "0")
        
        log_metric "pods_running" "$running_pods"
        log_metric "pods_failed" "$failed_pods"
        
        if [[ "$failed_pods" -gt 0 ]]; then
            generate_alert "pods_failed" "$failed_pods pods in failed state"
        fi
        
        log "Pod status: $running_pods running, $failed_pods failed"
    else
        log "No pods found or kubectl unavailable"
    fi
}

check_resource_usage() {
    local namespace="$1"
    
    # TODO: Implement resource usage monitoring
    # Check CPU and memory usage of pods
    # Compare with resource limits
    # Alert on high utilization
    
    log "Checking resource usage for namespace: $namespace"
    
    # Mock implementation
    local cpu_usage=75
    local memory_usage=60
    
    log_metric "k8s_cpu_usage" "$cpu_usage"
    log_metric "k8s_memory_usage" "$memory_usage"
    
    if [[ "$cpu_usage" -gt 90 ]]; then
        generate_alert "high_cpu_usage" "CPU usage is ${cpu_usage}%"
    fi
    
    if [[ "$memory_usage" -gt 90 ]]; then
        generate_alert "high_memory_usage" "Memory usage is ${memory_usage}%"
    fi
}

check_hpa_status() {
    local namespace="$1"
    
    log "Checking HPA status for namespace: $namespace"
    
    # TODO: Implement HPA monitoring
    # Check current replicas vs desired
    # Monitor scaling events
    # Alert on scaling issues
    
    if command -v kubectl &> /dev/null; then
        local hpa_info
        hpa_info=$(kubectl get hpa -n "$namespace" -o json 2>/dev/null | \
                  jq -r '.items[] | "\(.metadata.name) \(.status.currentReplicas) \(.spec.maxReplicas)"' 2>/dev/null || echo "")
        
        if [[ -n "$hpa_info" ]]; then
            while read -r name current max; do
                log_metric "hpa_current_replicas_${name}" "$current"
                
                if [[ "$current" -eq "$max" ]]; then
                    generate_alert "hpa_max_replicas" "HPA $name reached maximum replicas: $max"
                fi
            done <<< "$hpa_info"
        fi
    fi
}

# ==============================================================================
# ALERTING SYSTEM
# ==============================================================================

generate_alert() {
    local alert_type="$1"
    local message="$2"
    local severity="${3:-WARNING}"
    local timestamp=$(date +%s)
    
    log "ALERT [$severity] $alert_type: $message"
    
    # Save alert to file
    local alert_file="${ALERTS_DIR}/${alert_type}_${timestamp}.json"
    cat > "$alert_file" << EOF
{
    "timestamp": $timestamp,
    "alert_type": "$alert_type",
    "severity": "$severity",
    "message": "$message",
    "hostname": "$(hostname)",
    "script": "$(basename "$0")"
}
EOF
    
    # Send notifications
    send_alert_notification "$alert_type" "$message" "$severity"
    
    # Log metric
    log_metric "alert_${alert_type}" "1"
}

send_alert_notification() {
    local alert_type="$1"
    local message="$2"
    local severity="$3"
    
    # TODO: Implement multiple notification channels
    # - Slack webhook
    # - Email
    # - PagerDuty
    # - SMS
    # - Webhook endpoints
    
    # Slack notification example
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        send_slack_alert "$alert_type" "$message" "$severity"
    fi
    
    # Email notification example
    if [[ -n "${EMAIL_RECIPIENTS:-}" ]]; then
        send_email_alert "$alert_type" "$message" "$severity"
    fi
}

send_slack_alert() {
    local alert_type="$1"
    local message="$2"
    local severity="$3"
    
    local color="warning"
    [[ "$severity" == "CRITICAL" ]] && color="danger"
    [[ "$severity" == "INFO" ]] && color="good"
    
    local payload
    payload=$(cat << EOF
{
    "attachments": [
        {
            "color": "$color",
            "title": "MLOps Alert: $alert_type",
            "fields": [
                {
                    "title": "Severity",
                    "value": "$severity",
                    "short": true
                },
                {
                    "title": "Host",
                    "value": "$(hostname)",
                    "short": true
                },
                {
                    "title": "Message",
                    "value": "$message",
                    "short": false
                }
            ],
            "ts": $(date +%s)
        }
    ]
}
EOF
    )
    
    curl -X POST -H 'Content-type: application/json' \
         --data "$payload" \
         "${SLACK_WEBHOOK_URL}" &> /dev/null || true
}

# ==============================================================================
# REPORTING AND DASHBOARDS
# ==============================================================================

generate_health_report() {
    local output_file="${1:-health_report_$(date +%Y%m%d_%H%M%S).html}"
    
    log "Generating health report: $output_file"
    
    # TODO: Implement comprehensive health report
    # Include:
    # - System metrics summary
    # - Model performance trends
    # - Alert history
    # - Resource utilization
    # - Recommendations
    
    cat > "$output_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>MLOps Health Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { background: #f5f5f5; padding: 10px; margin: 5px; border-radius: 5px; }
        .alert { background: #ffe6e6; border-left: 5px solid #ff0000; padding: 10px; margin: 5px; }
        .good { color: green; }
        .warning { color: orange; }
        .critical { color: red; }
    </style>
</head>
<body>
    <h1>MLOps Health Report</h1>
    <p>Generated: $(date)</p>
    
    <h2>System Metrics</h2>
    <div class="metric">
        <strong>CPU Usage:</strong> <span id="cpu">Loading...</span>
    </div>
    
    <h2>Model Performance</h2>
    <div class="metric">
        <strong>Model Latency:</strong> <span id="latency">Loading...</span>
    </div>
    
    <h2>Recent Alerts</h2>
    <div id="alerts">Loading...</div>
    
    <script>
        // TODO: Add JavaScript to load real-time data
        // This would typically fetch data from an API
    </script>
</body>
</html>
EOF
    
    log "Health report generated: $output_file"
}

# ==============================================================================
# MAIN MONITORING LOOP
# ==============================================================================

run_monitoring_loop() {
    local interval="${1:-60}"  # Default 60 seconds
    
    log "Starting monitoring loop with ${interval}s interval"
    
    while true; do
        log "Running monitoring cycle..."
        
        # Collect system metrics
        collect_system_metrics
        
        # Check model performance
        if [[ -n "${MODEL_ENDPOINT:-}" ]]; then
            check_model_latency "$MODEL_ENDPOINT"
        fi
        
        # Check data freshness
        if [[ -n "${DATA_SOURCES:-}" ]]; then
            IFS=',' read -ra SOURCES <<< "$DATA_SOURCES"
            for source in "${SOURCES[@]}"; do
                check_data_freshness "$source"
            done
        fi
        
        # Monitor Kubernetes resources
        if [[ -n "${K8S_NAMESPACE:-}" ]]; then
            monitor_kubernetes_resources "$K8S_NAMESPACE"
        fi
        
        log "Monitoring cycle completed, sleeping for ${interval}s"
        sleep "$interval"
    done
}

# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    monitor              Start continuous monitoring loop
    check-system         Check system metrics once
    check-model URL      Check model endpoint
    check-data SOURCE    Check data source freshness
    check-k8s NAMESPACE  Check Kubernetes resources
    generate-report      Generate health report
    list-alerts          List recent alerts

Options:
    -i, --interval SEC   Monitoring interval in seconds (default: 60)
    -c, --config FILE    Configuration file
    -o, --output FILE    Output file for reports
    -h, --help          Show this help

Examples:
    $0 monitor --interval 30
    $0 check-model http://ml-service:8000
    $0 check-data file:/data/input.csv
    $0 generate-report --output report.html

EOF
}

main() {
    local command=""
    local interval=60
    local output_file=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--interval)
                interval="$2"
                shift 2
                ;;
            -c|--config)
                MONITORING_CONFIG="$2"
                shift 2
                ;;
            -o|--output)
                output_file="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            monitor)
                command="monitor"
                shift
                ;;
            check-system)
                command="check-system"
                shift
                ;;
            check-model)
                command="check-model"
                MODEL_ENDPOINT="$2"
                shift 2
                ;;
            check-data)
                command="check-data"
                DATA_SOURCE="$2"
                shift 2
                ;;
            check-k8s)
                command="check-k8s"
                K8S_NAMESPACE="$2"
                shift 2
                ;;
            generate-report)
                command="generate-report"
                shift
                ;;
            list-alerts)
                command="list-alerts"
                shift
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Load configuration if exists
    if [[ -f "$MONITORING_CONFIG" ]]; then
        # shellcheck source=/dev/null
        source "$MONITORING_CONFIG"
    fi
    
    # Execute command
    case "$command" in
        monitor)
            run_monitoring_loop "$interval"
            ;;
        check-system)
            collect_system_metrics
            ;;
        check-model)
            if [[ -n "$MODEL_ENDPOINT" ]]; then
                check_model_latency "$MODEL_ENDPOINT"
            else
                echo "Model endpoint required"
                exit 1
            fi
            ;;
        check-data)
            if [[ -n "$DATA_SOURCE" ]]; then
                check_data_freshness "$DATA_SOURCE"
            else
                echo "Data source required"
                exit 1
            fi
            ;;
        check-k8s)
            if [[ -n "$K8S_NAMESPACE" ]]; then
                monitor_kubernetes_resources "$K8S_NAMESPACE"
            else
                echo "Kubernetes namespace required"
                exit 1
            fi
            ;;
        generate-report)
            generate_health_report "$output_file"
            ;;
        list-alerts)
            find "$ALERTS_DIR" -name "*.json" -mtime -1 | \
                xargs cat | jq -r '. | "\(.timestamp | strftime("%Y-%m-%d %H:%M:%S")) [\(.severity)] \(.alert_type): \(.message)"' 2>/dev/null | \
                sort || echo "No recent alerts found"
            ;;
        *)
            echo "No command specified"
            usage
            exit 1
            ;;
    esac
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi