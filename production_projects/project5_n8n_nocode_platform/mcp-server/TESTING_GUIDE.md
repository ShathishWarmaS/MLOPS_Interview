# MCP Server Testing Guide

## 🚀 How to Test the MCP Server

### Prerequisites
- Node.js 18+
- npm or yarn
- Docker (for ChromaDB)
- API keys for AI services

### Step 1: Setup Environment

```bash
# Navigate to the project directory
cd mcp-server

# Run the setup test
node test-setup.cjs
```

### Step 2: Install Dependencies

```bash
# Install all dependencies
npm install

# If npm install is slow, try:
npm install --timeout=300000
# or
yarn install
```

### Step 3: Configure Environment Variables

```bash
# Edit the .env file with your API keys
nano .env

# Required API keys:
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
GOOGLE_API_KEY=your-google-ai-key
PERPLEXITY_API_KEY=your-perplexity-key
JWT_SECRET=your-very-long-secure-jwt-secret-key
```

### Step 4: Start External Services

```bash
# Start ChromaDB (required for RAG functionality)
docker run -p 8000:8000 chromadb/chroma

# Optional: Start MongoDB (if you want to test database features)
docker run -p 27017:27017 mongo:latest

# Optional: Start Redis (if you want to test caching)
docker run -p 6379:6379 redis:latest
```

### Step 5: Test TypeScript Compilation

```bash
# Build the TypeScript code
npm run build

# If successful, you should see a 'dist' folder
ls -la dist/
```

### Step 6: Run the Server

```bash
# Start in development mode
npm run dev

# Or start production build
npm start
```

### Step 7: Test MCP Tools

The server exposes 10 MCP tools. Here's how to test them:

#### Method 1: Using MCP Client (Recommended)
If you have Claude Desktop or another MCP client:

1. Add the server to your MCP client configuration
2. Test individual tools through the client interface

#### Method 2: Manual Testing with curl

```bash
# Test server health (if health endpoint exists)
curl -X GET http://localhost:3000/health

# Test MCP server directly (stdin/stdout protocol)
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | node dist/index.js
```

#### Method 3: Using the Test Script

```bash
# Run the quick test
node quick-test.js
```

## 🧪 Testing Individual Components

### 1. Chat with AI Tool
```json
{
  "name": "chat_with_ai",
  "arguments": {
    "message": "Hello, how are you?",
    "models": ["chatgpt"],
    "temperature": 0.7
  }
}
```

### 2. RAG Query Tool
```json
{
  "name": "rag_query",
  "arguments": {
    "query": "What is machine learning?",
    "collection": "test_collection",
    "top_k": 3
  }
}
```

### 3. Perplexity Search Tool
```json
{
  "name": "perplexity_search",
  "arguments": {
    "query": "Latest AI developments 2024",
    "focus": "academic",
    "include_citations": true
  }
}
```

### 4. Multimodal Processing Tool
```json
{
  "name": "process_multimodal",
  "arguments": {
    "content_type": "text",
    "content_data": "This is a test text",
    "analysis_type": "analyze"
  }
}
```

## 🔍 Debugging Common Issues

### Issue 1: Dependencies Not Installing
```bash
# Clear npm cache
npm cache clean --force

# Try alternative package managers
yarn install
# or
pnpm install
```

### Issue 2: API Keys Not Working
```bash
# Check if .env file is loaded
node -e "require('dotenv').config(); console.log(process.env.OPENAI_API_KEY)"

# Verify API key format
curl -H "Authorization: Bearer YOUR_OPENAI_KEY" https://api.openai.com/v1/models
```

### Issue 3: ChromaDB Connection Issues
```bash
# Check if ChromaDB is running
curl http://localhost:8000/api/v1/heartbeat

# View ChromaDB logs
docker logs $(docker ps -q --filter ancestor=chromadb/chroma)
```

### Issue 4: TypeScript Compilation Errors
```bash
# Check TypeScript version
npx tsc --version

# Run TypeScript check without building
npx tsc --noEmit --skipLibCheck
```

## 📊 Performance Testing

### Memory Usage Test
```bash
# Monitor memory usage while running
node --max-old-space-size=4096 dist/index.js
```

### Load Testing
```bash
# Install load testing tool
npm install -g loadtest

# Test server under load (adjust URL as needed)
loadtest -n 100 -c 10 http://localhost:3000/health
```

## 🏗️ Integration Testing

### Test with Real MCP Client
1. Install Claude Desktop
2. Configure the MCP server in settings
3. Test all tools through the interface

### Test with Custom Client
```javascript
// simple-mcp-client.js
import { Client } from '@modelcontextprotocol/sdk/client/index.js';

const client = new Client({
  name: 'test-client',
  version: '1.0.0'
});

// Test connection and list tools
const tools = await client.listTools();
console.log('Available tools:', tools);
```

## 📋 Test Checklist

- [ ] Node.js version (18+)
- [ ] All files exist
- [ ] Dependencies installed
- [ ] API keys configured
- [ ] ChromaDB running
- [ ] TypeScript compiles
- [ ] Server starts successfully
- [ ] All 10 tools respond
- [ ] Logs show no errors
- [ ] Memory usage reasonable
- [ ] External services connected

## 🚨 Troubleshooting

### Common Error Messages

1. **"Configuration validation failed"**
   - Check your .env file
   - Ensure all required API keys are set

2. **"ChromaDB connection failed"**
   - Start ChromaDB: `docker run -p 8000:8000 chromadb/chroma`
   - Check if port 8000 is free

3. **"Module not found"**
   - Run `npm install` again
   - Check node_modules folder exists

4. **"TypeScript compilation failed"**
   - Check TypeScript version
   - Ensure all imports are correct

### Getting Help

1. Check the logs in the `logs/` directory
2. Enable debug mode: `LOG_LEVEL=debug npm run dev`
3. Test individual services separately
4. Check API service status pages

## 🎯 Next Steps After Testing

1. **Production Deployment**: Configure for production environment
2. **Monitoring**: Set up logging and metrics
3. **Security**: Review security configurations
4. **Performance**: Optimize for your use case
5. **Features**: Add custom tools and integrations

## 📈 Monitoring & Logging

### View Logs
```bash
# Application logs
tail -f logs/application.log

# Error logs
tail -f logs/error.log

# Debug logs (if enabled)
tail -f logs/debug.log
```

### Health Checks
```bash
# Basic health check
curl http://localhost:3000/health

# Detailed service status
curl http://localhost:3000/health/detailed
```

Remember: This is a comprehensive MCP server with multiple AI integrations. Take time to test each component thoroughly!