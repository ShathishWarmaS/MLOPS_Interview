#!/bin/bash

echo "🚀 Starting n8n MLOps Platform Demo..."

# Check if Python is available
if command -v python3 >/dev/null 2>&1; then
    echo "📱 Starting demo server on http://localhost:8080"
    echo "🌐 Opening demo in browser..."
    
    # Start HTTP server in background
    python3 -m http.server 8080 &
    SERVER_PID=$!
    
    # Wait a moment for server to start
    sleep 2
    
    # Open in browser
    if command -v open >/dev/null 2>&1; then
        open "http://localhost:8080/simple-n8n-demo.html"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "http://localhost:8080/simple-n8n-demo.html"
    else
        echo "Please open http://localhost:8080/simple-n8n-demo.html in your browser"
    fi
    
    echo ""
    echo "✨ Demo Features:"
    echo "  - Visual workflow builder with drag & drop"
    echo "  - Pre-built ML nodes (Data Source, Feature Engineering, Model Training)"
    echo "  - Interactive node properties panel"
    echo "  - Real-time workflow execution simulation"
    echo "  - Connection lines between nodes"
    echo ""
    echo "🎮 Demo Instructions:"
    echo "  1. Drag nodes from left palette to canvas"
    echo "  2. Click nodes to select and edit properties"
    echo "  3. Click 'Execute' button to simulate workflow run"
    echo "  4. Watch real-time execution status in bottom bar"
    echo ""
    echo "Press Ctrl+C to stop the demo server"
    
    # Wait for user to stop
    trap "echo '🛑 Stopping demo server...'; kill $SERVER_PID; exit 0" INT
    wait $SERVER_PID
    
else
    echo "❌ Python3 not found. Opening file directly..."
    
    # Try to open file directly
    if command -v open >/dev/null 2>&1; then
        open "simple-n8n-demo.html"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "simple-n8n-demo.html"
    else
        echo "Please open simple-n8n-demo.html in your browser"
    fi
fi