#!/bin/bash

# n8n MLOps Platform Setup Script
# This script sets up the development environment for the n8n MLOps Platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Node.js version
check_node_version() {
    if command_exists node; then
        local node_version=$(node -v | cut -d'v' -f2)
        local major_version=$(echo $node_version | cut -d'.' -f1)
        
        if [ "$major_version" -ge 18 ]; then
            print_success "Node.js version $node_version detected"
            return 0
        else
            print_error "Node.js version $node_version is too old. Please install Node.js 18 or later"
            return 1
        fi
    else
        print_error "Node.js is not installed. Please install Node.js 18 or later"
        return 1
    fi
}

# Function to check Docker
check_docker() {
    if command_exists docker; then
        if docker info >/dev/null 2>&1; then
            print_success "Docker is installed and running"
            return 0
        else
            print_error "Docker is installed but not running. Please start Docker"
            return 1
        fi
    else
        print_error "Docker is not installed. Please install Docker"
        return 1
    fi
}

# Function to check Docker Compose
check_docker_compose() {
    if command_exists docker-compose || docker compose version >/dev/null 2>&1; then
        print_success "Docker Compose is available"
        return 0
    else
        print_error "Docker Compose is not available. Please install Docker Compose"
        return 1
    fi
}

# Function to install Node.js dependencies
install_dependencies() {
    print_status "Installing Node.js dependencies..."
    
    if [ -f "package-lock.json" ]; then
        npm ci
    else
        npm install
    fi
    
    print_success "Dependencies installed successfully"
}

# Function to setup Lerna
setup_lerna() {
    print_status "Setting up Lerna monorepo..."
    
    # Bootstrap packages
    npx lerna bootstrap
    
    print_success "Lerna setup completed"
}

# Function to build packages
build_packages() {
    print_status "Building packages..."
    
    npm run build
    
    print_success "Packages built successfully"
}

# Function to setup environment files
setup_environment() {
    print_status "Setting up environment files..."
    
    # Copy example environment file
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Environment file created from .env.example"
            print_warning "Please update .env file with your configuration"
        else
            print_warning ".env.example file not found. Creating basic .env file"
            cat > .env << EOF
# n8n MLOps Platform Configuration
N8N_ENCRYPTION_KEY=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
SLACK_WEBHOOK_URL=https://hooks.slack.com/your/webhook/url
MLFLOW_TRACKING_URI=http://localhost:5000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin_password
EOF
        fi
    else
        print_success "Environment file already exists"
    fi
}

# Function to setup database
setup_database() {
    print_status "Setting up database..."
    
    # Start database services
    docker-compose -f docker/docker-compose.yml up -d postgres redis
    
    # Wait for database to be ready
    print_status "Waiting for database to be ready..."
    sleep 10
    
    # Run database migrations
    npm run db:migrate
    
    # Seed database with sample data
    npm run db:seed
    
    print_success "Database setup completed"
}

# Function to start development environment
start_dev_environment() {
    print_status "Starting development environment..."
    
    # Start infrastructure services
    docker-compose -f docker/docker-compose.yml up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 30
    
    print_success "Development environment started"
    print_status "Services available at:"
    echo "  - n8n Web UI: http://localhost:5678"
    echo "  - MLflow: http://localhost:5000"
    echo "  - Grafana: http://localhost:3000"
    echo "  - Jupyter: http://localhost:8888"
    echo "  - MinIO: http://localhost:9001"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    npm run test
    
    print_success "All tests passed"
}

# Function to setup Git hooks
setup_git_hooks() {
    print_status "Setting up Git hooks..."
    
    if command_exists git && [ -d ".git" ]; then
        # Install husky
        npx husky install
        
        # Setup pre-commit hook
        npx husky add .husky/pre-commit "npm run lint:fix && npm run test"
        
        # Setup pre-push hook
        npx husky add .husky/pre-push "npm run typecheck && npm run test:coverage"
        
        print_success "Git hooks setup completed"
    else
        print_warning "Git repository not found. Skipping Git hooks setup"
    fi
}

# Function to create directories
create_directories() {
    print_status "Creating required directories..."
    
    mkdir -p logs
    mkdir -p data/uploads
    mkdir -p data/exports
    mkdir -p models
    mkdir -p reports
    mkdir -p tmp
    
    print_success "Directories created"
}

# Function to display help
show_help() {
    echo "n8n MLOps Platform Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h           Show this help message"
    echo "  --skip-deps          Skip Node.js dependencies installation"
    echo "  --skip-build         Skip package building"
    echo "  --skip-db            Skip database setup"
    echo "  --skip-docker        Skip Docker environment setup"
    echo "  --skip-tests         Skip running tests"
    echo "  --dev-only           Setup for development only (skip production setup)"
    echo "  --production         Setup for production deployment"
    echo ""
    echo "Examples:"
    echo "  $0                   Full setup"
    echo "  $0 --dev-only        Development setup only"
    echo "  $0 --skip-tests      Setup without running tests"
    echo ""
}

# Main setup function
main() {
    local skip_deps=false
    local skip_build=false
    local skip_db=false
    local skip_docker=false
    local skip_tests=false
    local dev_only=false
    local production=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --skip-deps)
                skip_deps=true
                shift
                ;;
            --skip-build)
                skip_build=true
                shift
                ;;
            --skip-db)
                skip_db=true
                shift
                ;;
            --skip-docker)
                skip_docker=true
                shift
                ;;
            --skip-tests)
                skip_tests=true
                shift
                ;;
            --dev-only)
                dev_only=true
                shift
                ;;
            --production)
                production=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    echo "🚀 Setting up n8n MLOps Platform..."
    echo ""
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    check_node_version || exit 1
    
    if [ "$skip_docker" = false ]; then
        check_docker || exit 1
        check_docker_compose || exit 1
    fi
    
    # Create directories
    create_directories
    
    # Setup environment
    setup_environment
    
    # Install dependencies
    if [ "$skip_deps" = false ]; then
        install_dependencies
        setup_lerna
    fi
    
    # Build packages
    if [ "$skip_build" = false ]; then
        build_packages
    fi
    
    # Setup database
    if [ "$skip_db" = false ] && [ "$skip_docker" = false ]; then
        setup_database
    fi
    
    # Setup Git hooks (development only)
    if [ "$dev_only" = true ] || [ "$production" = false ]; then
        setup_git_hooks
    fi
    
    # Run tests
    if [ "$skip_tests" = false ]; then
        run_tests
    fi
    
    # Start development environment
    if [ "$dev_only" = true ] && [ "$skip_docker" = false ]; then
        start_dev_environment
    fi
    
    echo ""
    print_success "🎉 n8n MLOps Platform setup completed successfully!"
    echo ""
    
    if [ "$dev_only" = true ]; then
        echo "Development environment is ready!"
        echo ""
        echo "Next steps:"
        echo "1. Update .env file with your configuration"
        echo "2. Access the web UI at http://localhost:5678"
        echo "3. Start building your ML workflows!"
        echo ""
        echo "Useful commands:"
        echo "  npm run dev        - Start development servers"
        echo "  npm run test       - Run tests"
        echo "  npm run lint       - Run linting"
        echo "  npm run build      - Build packages"
        echo ""
    elif [ "$production" = true ]; then
        echo "Production setup completed!"
        echo ""
        echo "Next steps:"
        echo "1. Configure production environment variables"
        echo "2. Setup SSL certificates"
        echo "3. Deploy to Kubernetes cluster"
        echo "4. Configure monitoring and alerting"
        echo ""
    else
        echo "Basic setup completed!"
        echo ""
        echo "Run the script with --dev-only for development setup"
        echo "or --production for production deployment setup"
        echo ""
    fi
}

# Run main function
main "$@"