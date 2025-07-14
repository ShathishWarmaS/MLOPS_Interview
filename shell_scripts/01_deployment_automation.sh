#!/bin/bash

# ==============================================================================
# MLOps Deployment Automation Script
# Challenge: Automate complete model deployment workflow
# ==============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/deployment.conf"
LOG_FILE="${SCRIPT_DIR}/deployment.log"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# LOGGING AND UTILITIES
# ==============================================================================

log() {
    local level="$1"
    shift
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

log_info() {
    log "INFO" "$@"
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_warn() {
    log "WARN" "$@"
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    log "ERROR" "$@"
    echo -e "${RED}[ERROR]${NC} $*"
}

log_success() {
    log "SUCCESS" "$@"
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

# ==============================================================================
# CONFIGURATION MANAGEMENT
# ==============================================================================

load_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        create_sample_config
        exit 1
    fi
    
    # Source configuration file
    # shellcheck source=/dev/null
    source "$CONFIG_FILE"
    
    # Validate required variables
    validate_config
}

create_sample_config() {
    log_info "Creating sample configuration file..."
    cat > "$CONFIG_FILE" << 'EOF'
# Model Deployment Configuration

# Docker Configuration
DOCKER_REGISTRY="gcr.io/your-project"
IMAGE_NAME="ml-model"
DOCKERFILE_PATH="./Dockerfile"

# Kubernetes Configuration
NAMESPACE="ml-production"
DEPLOYMENT_NAME="ml-model-deployment"
SERVICE_NAME="ml-model-service"
REPLICAS=3

# Health Check Configuration
HEALTH_CHECK_URL="http://localhost:8000/health"
HEALTH_CHECK_TIMEOUT=300
HEALTH_CHECK_INTERVAL=10

# Rollback Configuration
ROLLBACK_ON_FAILURE=true
PREVIOUS_VERSION_KEEP=3

# Notification Configuration
SLACK_WEBHOOK_URL=""
EMAIL_RECIPIENTS=""

# Environment Specific
ENVIRONMENT="staging"  # staging, production
EOF
    log_info "Sample configuration created at: $CONFIG_FILE"
}

validate_config() {
    local required_vars=(
        "DOCKER_REGISTRY"
        "IMAGE_NAME"
        "NAMESPACE"
        "DEPLOYMENT_NAME"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Required configuration variable $var is not set"
            exit 1
        fi
    done
    
    log_info "Configuration validation passed"
}

# ==============================================================================
# DOCKER OPERATIONS
# ==============================================================================

build_docker_image() {
    local version="$1"
    local image_tag="${DOCKER_REGISTRY}/${IMAGE_NAME}:${version}"
    
    log_info "Building Docker image: $image_tag"
    
    # TODO: Implement Docker build with optimizations
    # - Multi-stage builds
    # - Build cache optimization
    # - Security scanning
    
    if docker build -t "$image_tag" -f "$DOCKERFILE_PATH" .; then
        log_success "Docker image built successfully"
        return 0
    else
        log_error "Docker image build failed"
        return 1
    fi
}

push_docker_image() {
    local version="$1"
    local image_tag="${DOCKER_REGISTRY}/${IMAGE_NAME}:${version}"
    
    log_info "Pushing Docker image: $image_tag"
    
    # TODO: Implement push with retry logic
    local max_retries=3
    local retry_count=0
    
    while [[ $retry_count -lt $max_retries ]]; do
        if docker push "$image_tag"; then
            log_success "Docker image pushed successfully"
            return 0
        else
            ((retry_count++))
            log_warn "Push failed, retry $retry_count/$max_retries"
            sleep $((retry_count * 5))
        fi
    done
    
    log_error "Failed to push Docker image after $max_retries retries"
    return 1
}

scan_image_security() {
    local version="$1"
    local image_tag="${DOCKER_REGISTRY}/${IMAGE_NAME}:${version}"
    
    log_info "Scanning image for security vulnerabilities: $image_tag"
    
    # TODO: Implement security scanning
    # Examples: Trivy, Snyk, Clair
    
    # Mock implementation
    if command -v trivy &> /dev/null; then
        trivy image --exit-code 1 "$image_tag"
        return $?
    else
        log_warn "Security scanner not available, skipping scan"
        return 0
    fi
}

# ==============================================================================
# KUBERNETES OPERATIONS
# ==============================================================================

update_kubernetes_deployment() {
    local version="$1"
    local image_tag="${DOCKER_REGISTRY}/${IMAGE_NAME}:${version}"
    
    log_info "Updating Kubernetes deployment with image: $image_tag"
    
    # TODO: Implement Kubernetes deployment update
    # - Rolling update strategy
    # - Deployment status monitoring
    # - Rollout verification
    
    if kubectl set image deployment/"$DEPLOYMENT_NAME" \
        "$IMAGE_NAME"="$image_tag" \
        --namespace="$NAMESPACE"; then
        
        log_info "Deployment update initiated"
        wait_for_rollout "$DEPLOYMENT_NAME"
        return $?
    else
        log_error "Failed to update deployment"
        return 1
    fi
}

wait_for_rollout() {
    local deployment_name="$1"
    local timeout=600  # 10 minutes
    
    log_info "Waiting for rollout to complete..."
    
    if kubectl rollout status deployment/"$deployment_name" \
        --namespace="$NAMESPACE" \
        --timeout="${timeout}s"; then
        log_success "Rollout completed successfully"
        return 0
    else
        log_error "Rollout failed or timed out"
        return 1
    fi
}

get_current_deployment_version() {
    kubectl get deployment "$DEPLOYMENT_NAME" \
        --namespace="$NAMESPACE" \
        -o jsonpath='{.spec.template.spec.containers[0].image}' \
        2>/dev/null | grep -o '[^:]*$' || echo "unknown"
}

rollback_deployment() {
    log_warn "Rolling back deployment..."
    
    if kubectl rollout undo deployment/"$DEPLOYMENT_NAME" \
        --namespace="$NAMESPACE"; then
        
        wait_for_rollout "$DEPLOYMENT_NAME"
        log_success "Rollback completed"
        return 0
    else
        log_error "Rollback failed"
        return 1
    fi
}

# ==============================================================================
# HEALTH CHECKS
# ==============================================================================

perform_health_checks() {
    local service_url="$1"
    local timeout="${HEALTH_CHECK_TIMEOUT:-300}"
    local interval="${HEALTH_CHECK_INTERVAL:-10}"
    local end_time=$((SECONDS + timeout))
    
    log_info "Performing health checks for: $service_url"
    
    while [[ $SECONDS -lt $end_time ]]; do
        if check_service_health "$service_url"; then
            log_success "Health check passed"
            return 0
        fi
        
        log_info "Health check failed, retrying in ${interval}s..."
        sleep "$interval"
    done
    
    log_error "Health check timeout after ${timeout}s"
    return 1
}

check_service_health() {
    local url="$1"
    
    # TODO: Implement comprehensive health checks
    # - HTTP endpoint availability
    # - Response time checks
    # - Functional tests
    # - Load testing
    
    if curl -s -f --max-time 30 "$url" > /dev/null; then
        return 0
    else
        return 1
    fi
}

run_smoke_tests() {
    local base_url="$1"
    
    log_info "Running smoke tests..."
    
    # TODO: Implement smoke tests
    # - Basic functionality tests
    # - API endpoint validation
    # - Performance benchmarks
    
    local test_endpoints=(
        "$base_url/health"
        "$base_url/ready"
        "$base_url/metrics"
    )
    
    for endpoint in "${test_endpoints[@]}"; do
        if ! curl -s -f --max-time 10 "$endpoint" > /dev/null; then
            log_error "Smoke test failed for: $endpoint"
            return 1
        fi
    done
    
    log_success "All smoke tests passed"
    return 0
}

# ==============================================================================
# NOTIFICATION SYSTEM
# ==============================================================================

send_notification() {
    local status="$1"
    local message="$2"
    local version="${3:-unknown}"
    
    # Send Slack notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        send_slack_notification "$status" "$message" "$version"
    fi
    
    # Send email notification
    if [[ -n "${EMAIL_RECIPIENTS:-}" ]]; then
        send_email_notification "$status" "$message" "$version"
    fi
}

send_slack_notification() {
    local status="$1"
    local message="$2"
    local version="$3"
    
    local color="good"
    [[ "$status" == "FAILED" ]] && color="danger"
    [[ "$status" == "WARNING" ]] && color="warning"
    
    local payload=$(cat << EOF
{
    "attachments": [
        {
            "color": "$color",
            "title": "Model Deployment $status",
            "fields": [
                {
                    "title": "Environment",
                    "value": "$ENVIRONMENT",
                    "short": true
                },
                {
                    "title": "Version",
                    "value": "$version",
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
        "$SLACK_WEBHOOK_URL" &> /dev/null
}

send_email_notification() {
    local status="$1"
    local message="$2"
    local version="$3"
    
    # TODO: Implement email notification
    # Use mail command or external service
    
    log_info "Email notification sent: $status - $message"
}

# ==============================================================================
# MAIN DEPLOYMENT WORKFLOW
# ==============================================================================

deploy_model() {
    local version="$1"
    local previous_version
    
    log_info "Starting deployment of version: $version"
    
    # Get current version for potential rollback
    previous_version=$(get_current_deployment_version)
    log_info "Current deployment version: $previous_version"
    
    # Build and push Docker image
    if ! build_docker_image "$version"; then
        send_notification "FAILED" "Docker build failed for version $version" "$version"
        exit 1
    fi
    
    if ! scan_image_security "$version"; then
        log_error "Security scan failed"
        send_notification "FAILED" "Security scan failed for version $version" "$version"
        exit 1
    fi
    
    if ! push_docker_image "$version"; then
        send_notification "FAILED" "Docker push failed for version $version" "$version"
        exit 1
    fi
    
    # Update Kubernetes deployment
    if ! update_kubernetes_deployment "$version"; then
        log_error "Kubernetes deployment failed"
        
        if [[ "${ROLLBACK_ON_FAILURE:-true}" == "true" ]]; then
            rollback_deployment
            send_notification "FAILED" "Deployment failed, rolled back to $previous_version" "$version"
        else
            send_notification "FAILED" "Deployment failed for version $version" "$version"
        fi
        exit 1
    fi
    
    # Perform health checks
    local service_url="http://${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local:80"
    if ! perform_health_checks "$service_url"; then
        log_error "Health checks failed"
        
        if [[ "${ROLLBACK_ON_FAILURE:-true}" == "true" ]]; then
            rollback_deployment
            send_notification "FAILED" "Health checks failed, rolled back to $previous_version" "$version"
        else
            send_notification "FAILED" "Health checks failed for version $version" "$version"
        fi
        exit 1
    fi
    
    # Run smoke tests
    if ! run_smoke_tests "$service_url"; then
        log_error "Smoke tests failed"
        
        if [[ "${ROLLBACK_ON_FAILURE:-true}" == "true" ]]; then
            rollback_deployment
            send_notification "FAILED" "Smoke tests failed, rolled back to $previous_version" "$version"
        else
            send_notification "FAILED" "Smoke tests failed for version $version" "$version"
        fi
        exit 1
    fi
    
    log_success "Deployment completed successfully!"
    send_notification "SUCCESS" "Deployment completed successfully" "$version"
    
    # Cleanup old versions
    cleanup_old_versions
}

cleanup_old_versions() {
    local keep_versions="${PREVIOUS_VERSION_KEEP:-3}"
    
    log_info "Cleaning up old Docker images (keeping last $keep_versions versions)"
    
    # TODO: Implement cleanup logic
    # - Remove old Docker images
    # - Clean up old Kubernetes resources
    # - Archive old deployment configs
    
    # Mock implementation
    log_info "Cleanup completed"
}

# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    deploy VERSION      Deploy model with specified version
    rollback           Rollback to previous version
    status             Show current deployment status
    health-check       Perform health check on current deployment
    cleanup            Clean up old versions and resources

Options:
    -c, --config FILE   Use custom configuration file
    -e, --env ENV       Set environment (staging, production)
    -v, --verbose       Enable verbose logging
    -h, --help          Show this help message

Examples:
    $0 deploy v1.2.3
    $0 --env production deploy v1.2.3
    $0 rollback
    $0 status

EOF
}

main() {
    local command=""
    local version=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--verbose)
                set -x
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            deploy)
                command="deploy"
                version="$2"
                shift 2
                ;;
            rollback)
                command="rollback"
                shift
                ;;
            status)
                command="status"
                shift
                ;;
            health-check)
                command="health-check"
                shift
                ;;
            cleanup)
                command="cleanup"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Load configuration
    load_config
    
    # Execute command
    case "$command" in
        deploy)
            if [[ -z "$version" ]]; then
                log_error "Version is required for deploy command"
                usage
                exit 1
            fi
            deploy_model "$version"
            ;;
        rollback)
            rollback_deployment
            ;;
        status)
            current_version=$(get_current_deployment_version)
            log_info "Current deployment version: $current_version"
            ;;
        health-check)
            service_url="http://${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local:80"
            perform_health_checks "$service_url"
            ;;
        cleanup)
            cleanup_old_versions
            ;;
        *)
            log_error "No command specified"
            usage
            exit 1
            ;;
    esac
}

# ==============================================================================
# INTERVIEW QUESTIONS
# ==============================================================================

# 1. Error Handling:
#    Q: How do you handle partial failures in deployment?
#    A: Use atomic operations, rollback mechanisms, and proper error codes

# 2. Concurrency:
#    Q: How do you prevent multiple deployments from running simultaneously?
#    A: Use file locks, distributed locks, or deployment queues

# 3. Security:
#    Q: How do you secure secrets in deployment scripts?
#    A: Use secret management systems, avoid hardcoding, use encryption

# 4. Monitoring:
#    Q: How do you monitor deployment progress?
#    A: Real-time logging, metrics collection, status endpoints

# 5. Testing:
#    Q: How do you test deployment scripts?
#    A: Unit tests, integration tests, dry-run mode, staging environments

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi