# MLOps MCP Server

A comprehensive Model Context Protocol (MCP) server for MLOps platform with multimodal AI integration supporting ChatGPT, Claude, Gemini, RAG, and Perplexity.

## Features

### 🤖 AI Integration
- **Multi-Model Support**: OpenAI GPT-4, Anthropic Claude, Google Gemini
- **Model Comparison**: Compare responses across different AI models
- **Context-Aware Generation**: RAG-enhanced responses with document context
- **Streaming Support**: Real-time response streaming

### 🔍 RAG (Retrieval-Augmented Generation)
- **Vector Database**: ChromaDB integration for efficient document storage
- **Document Processing**: Support for PDF, DOCX, TXT, MD, HTML files
- **Semantic Search**: Advanced similarity search with configurable thresholds
- **Hybrid Search**: Combine semantic and keyword search
- **Collection Management**: Organize documents into collections

### 🔎 Perplexity Integration
- **Web Search**: Real-time web search with AI-powered answers
- **Academic Focus**: Specialized search for academic content
- **Fact Checking**: Verify statements with source citations
- **Research Tools**: Comprehensive topic research capabilities

### 🎭 Multimodal Processing
- **Image Processing**: Analysis, OCR, classification, thumbnail generation
- **Video Processing**: Metadata extraction, thumbnail generation, content analysis
- **Audio Processing**: Transcription, sentiment analysis, metadata extraction
- **Document Processing**: Text extraction, summarization, classification

### 🚀 MLOps Integration
- **Model Deployment**: Deploy to Kubernetes, Docker, Serverless, Edge
- **Pipeline Management**: Create and manage ML pipelines
- **Experiment Tracking**: MLflow integration for experiment management
- **Model Monitoring**: Performance tracking and drift detection
- **Auto-scaling**: Configurable scaling policies

### 🔧 Advanced Features
- **Workflow Automation**: Create custom AI workflows
- **Data Analysis**: AI-powered dataset analysis and insights
- **Real-time Processing**: WebSocket support for real-time operations
- **Security**: JWT authentication, rate limiting, CORS protection
- **Monitoring**: Comprehensive logging and metrics

## Installation

### Prerequisites
- Node.js 18+
- MongoDB
- Redis
- ChromaDB (for RAG functionality)
- Docker (for containerized deployments)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mcp-server
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start ChromaDB** (for RAG functionality)
   ```bash
   docker run -p 8000:8000 chromadb/chroma
   ```

5. **Start the server**
   ```bash
   # Development
   npm run dev
   
   # Production
   npm run build
   npm start
   ```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
JWT_SECRET=your_jwt_secret

# Database URLs
DATABASE_URL=mongodb://localhost:27017/mlops-mcp
REDIS_HOST=localhost
CHROMA_URL=http://localhost:8000

# MLOps Configuration
MLFLOW_URL=http://localhost:5000
KUBEFLOW_URL=http://localhost:8080
```

### Service Configuration

The server automatically configures itself based on environment variables. See `src/config/config.ts` for all available options.

## API Tools

### Chat with AI
```json
{
  "name": "chat_with_ai",
  "arguments": {
    "message": "Explain machine learning",
    "models": ["chatgpt", "claude", "gemini"],
    "temperature": 0.7
  }
}
```

### RAG Query
```json
{
  "name": "rag_query",
  "arguments": {
    "query": "What is deep learning?",
    "collection": "ml_docs",
    "top_k": 5
  }
}
```

### Document Ingestion
```json
{
  "name": "rag_ingest",
  "arguments": {
    "documents": ["/path/to/document.pdf"],
    "collection": "ml_docs",
    "chunk_size": 1000
  }
}
```

### Perplexity Search
```json
{
  "name": "perplexity_search",
  "arguments": {
    "query": "Latest developments in AI",
    "focus": "academic",
    "include_citations": true
  }
}
```

### Multimodal Processing
```json
{
  "name": "process_multimodal",
  "arguments": {
    "content_type": "image",
    "content_data": "base64_encoded_image",
    "analysis_type": "describe"
  }
}
```

### Model Deployment
```json
{
  "name": "mlops_deploy_model",
  "arguments": {
    "model_name": "my_model",
    "deployment_target": "kubernetes",
    "scaling_config": {
      "min_replicas": 1,
      "max_replicas": 10
    }
  }
}
```

### Model Comparison
```json
{
  "name": "compare_ai_models",
  "arguments": {
    "prompt": "Explain quantum computing",
    "models": ["chatgpt", "claude", "gemini"],
    "evaluation_criteria": ["accuracy", "clarity", "depth"]
  }
}
```

## Architecture

### Core Components

- **AI Orchestrator**: Manages multiple AI model integrations
- **RAG Service**: Handles document storage and retrieval
- **Perplexity Service**: Web search and fact-checking
- **Media Processor**: Multimodal content processing
- **MLOps Integration**: Model deployment and pipeline management

### Data Flow

1. **Request Processing**: MCP server receives tool calls
2. **Service Routing**: Requests routed to appropriate service
3. **AI Processing**: AI models process requests with context
4. **Response Generation**: Structured responses returned to client
5. **Logging & Monitoring**: All operations logged and monitored

## Development

### Project Structure
```
src/
├── config/          # Configuration management
├── services/        # Core service implementations
├── utils/           # Utility functions
├── types/           # TypeScript type definitions
└── index.ts         # Main server entry point
```

### Development Commands
```bash
# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Lint code
npm run lint

# Format code
npm run format
```

### Testing

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t mlops-mcp-server .

# Run container
docker run -p 3000:3000 --env-file .env mlops-mcp-server
```

### Docker Compose

```yaml
version: '3.8'
services:
  mcp-server:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
    depends_on:
      - mongodb
      - redis
      - chroma
  
  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
  
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
  
  chroma:
    image: chromadb/chroma
    ports:
      - "8000:8000"
```

### Kubernetes Deployment

See `k8s/` directory for Kubernetes manifests.

## Monitoring

### Logging

- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Multiple Levels**: Error, warn, info, debug, verbose
- **File Rotation**: Automatic log file rotation
- **Context Tracking**: Request correlation and user tracking

### Metrics

- **Performance Metrics**: Response times, throughput
- **AI Model Usage**: Token usage, model performance
- **Resource Usage**: Memory, CPU, storage
- **Error Tracking**: Error rates and patterns

### Health Checks

```bash
# Check server health
curl http://localhost:3000/health

# Check service status
curl http://localhost:3000/health/detailed
```

## Security

### Authentication
- JWT-based authentication
- API key validation
- Rate limiting per user/IP

### Data Protection
- Input validation and sanitization
- Secure file upload handling
- Environment variable protection

### Network Security
- CORS configuration
- Helmet.js security headers
- Request size limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run tests and linting
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the examples

## Roadmap

- [ ] Additional AI model integrations
- [ ] Enhanced multimodal capabilities
- [ ] Advanced pipeline orchestration
- [ ] Real-time collaboration features
- [ ] Mobile SDK support
- [ ] Edge computing optimizations

## Changelog

See CHANGELOG.md for version history and updates.
