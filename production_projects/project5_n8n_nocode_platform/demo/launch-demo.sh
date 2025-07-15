#!/bin/bash

echo "🚀 n8n MLOps Platform - Complete Demo Launcher"
echo "=============================================="

# Function to check if Python packages are installed
check_python_packages() {
    echo "🔍 Checking Python packages..."
    python3 -c "
import sys
packages = ['pandas', 'numpy', 'sklearn']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg} - OK')
    except ImportError:
        missing.append(pkg)
        print(f'❌ {pkg} - Missing')

if missing:
    print(f'\\n📦 Install missing packages with:')
    print(f'pip3 install {\" \".join(missing)}')
    sys.exit(1)
else:
    print('\\n✅ All required packages are installed!')
"
    if [ $? -ne 0 ]; then
        echo "❌ Missing Python packages. Installing..."
        pip3 install pandas numpy scikit-learn matplotlib seaborn
    fi
}

# Function to run ML backend
run_ml_backend() {
    echo "🧠 Running ML backend pipeline..."
    python3 ml_backend.py run
    
    if [ $? -eq 0 ]; then
        echo "✅ ML backend completed successfully!"
    else
        echo "❌ ML backend failed"
        return 1
    fi
}

# Function to start web demo
start_web_demo() {
    echo "🌐 Starting web demo..."
    
    # Try to start HTTP server
    if command -v python3 >/dev/null 2>&1; then
        echo "📡 Starting HTTP server on port 8081..."
        python3 -m http.server 8081 &
        SERVER_PID=$!
        
        # Wait for server to start
        sleep 2
        
        echo "🎉 Demo is ready!"
        echo ""
        echo "📱 Available demos:"
        echo "  • Visual Workflow Builder: http://localhost:8081/simple-n8n-demo.html"
        echo "  • Working ML Pipeline:     http://localhost:8081/working-ml-demo.html"
        echo ""
        
        # Open demos in browser
        if command -v open >/dev/null 2>&1; then
            echo "🌐 Opening demos in browser..."
            open "http://localhost:8081/working-ml-demo.html"
            sleep 2
            open "http://localhost:8081/simple-n8n-demo.html"
        else
            echo "Please open the URLs above in your browser"
        fi
        
        echo "🎮 Demo Features:"
        echo "  ✨ Visual workflow builder with drag & drop"
        echo "  🧠 Real ML pipeline with fraud detection"
        echo "  📊 Live data processing and model training"
        echo "  📈 Interactive charts and metrics"
        echo "  🔄 Step-by-step pipeline execution"
        echo ""
        echo "Press Ctrl+C to stop the demo server"
        
        # Wait for user to stop
        trap "echo '🛑 Stopping demo server...'; kill $SERVER_PID 2>/dev/null; exit 0" INT
        wait $SERVER_PID
        
    else
        echo "❌ Python3 not found. Opening files directly..."
        open "working-ml-demo.html"
        open "simple-n8n-demo.html"
    fi
}

# Main execution
main() {
    cd "$(dirname "$0")"
    
    echo "📁 Working directory: $(pwd)"
    echo ""
    
    # Check Python packages
    check_python_packages
    echo ""
    
    # Run ML backend
    run_ml_backend
    echo ""
    
    # Start web demo
    start_web_demo
}

# Show help
show_help() {
    echo "n8n MLOps Platform Demo Launcher"
    echo ""
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help"
    echo "  --ml-only      Run only ML backend"
    echo "  --web-only     Start only web demo"
    echo "  --install      Install Python dependencies"
    echo ""
    echo "Examples:"
    echo "  $0             # Run complete demo"
    echo "  $0 --ml-only   # Run ML pipeline only"
    echo "  $0 --web-only  # Start web demo only"
}

# Parse arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --ml-only)
        cd "$(dirname "$0")"
        check_python_packages
        run_ml_backend
        ;;
    --web-only)
        cd "$(dirname "$0")"
        start_web_demo
        ;;
    --install)
        pip3 install pandas numpy scikit-learn matplotlib seaborn
        echo "✅ Dependencies installed!"
        ;;
    "")
        main
        ;;
    *)
        echo "❌ Unknown option: $1"
        show_help
        exit 1
        ;;
esac