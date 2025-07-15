#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
  Tool,
} from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import { MultimodalAIOrchestrator } from './services/ai-orchestrator.js';
import { RAGService } from './services/rag-service.js';
import { PerplexityService } from './services/perplexity-service.js';
import { MLOpsIntegration } from './services/mlops-integration.js';
import { MediaProcessor } from './services/media-processor.js';
import { logger } from './utils/logger.js';
import { config } from './config/config.js';

/**
 * MCP Server for MLOps Platform with Multimodal AI Integration
 * Supports ChatGPT, Claude, Gemini, RAG, and Perplexity
 */
class MLOpsMCPServer {
  private server: Server;
  private aiOrchestrator: MultimodalAIOrchestrator;
  private ragService: RAGService;
  private perplexityService: PerplexityService;
  private mlopsIntegration: MLOpsIntegration;
  private mediaProcessor: MediaProcessor;

  constructor() {
    this.server = new Server(
      {
        name: 'mlops-mcp-server',
        version: '1.0.0',
        description: 'MCP Server for MLOps Platform with Multimodal AI Integration',
      },
      {
        capabilities: {
          tools: {},
          resources: {},
        },
      }
    );

    this.initializeServices();
    this.setupToolHandlers();
    this.setupErrorHandling();
  }

  private async initializeServices() {
    try {
      // Initialize AI orchestrator with multiple models
      this.aiOrchestrator = new MultimodalAIOrchestrator({
        openai: { apiKey: config.openai.apiKey },
        anthropic: { apiKey: config.anthropic.apiKey },
        google: { apiKey: config.google.apiKey },
      });

      // Initialize RAG service
      this.ragService = new RAGService({
        vectorStore: config.rag.vectorStore,
        embeddingModel: config.rag.embeddingModel,
      });

      // Initialize Perplexity service
      this.perplexityService = new PerplexityService({
        apiKey: config.perplexity.apiKey,
      });

      // Initialize MLOps integration
      this.mlopsIntegration = new MLOpsIntegration({
        mlflowUrl: config.mlops.mlflowUrl,
        kubeflowUrl: config.mlops.kubeflowUrl,
      });

      // Initialize media processor
      this.mediaProcessor = new MediaProcessor({
        tempDir: config.media.tempDir,
        maxFileSize: config.media.maxFileSize,
      });

      await this.ragService.initialize();
      await this.mlopsIntegration.initialize();

      logger.info('All services initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize services:', error);
      throw error;
    }
  }

  private setupToolHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          // AI Chat Tools
          {
            name: 'chat_with_ai',
            description: 'Chat with multiple AI models (ChatGPT, Claude, Gemini) and get comparative responses',
            inputSchema: {
              type: 'object',
              properties: {
                message: { type: 'string', description: 'The message to send to AI models' },
                models: { 
                  type: 'array', 
                  items: { type: 'string' },
                  description: 'AI models to use: chatgpt, claude, gemini, or all',
                  default: ['all']
                },
                system_prompt: { type: 'string', description: 'Optional system prompt' },
                temperature: { type: 'number', description: 'Temperature for response generation', default: 0.7 },
                max_tokens: { type: 'number', description: 'Maximum tokens in response', default: 1000 }
              },
              required: ['message']
            }
          },

          // Multimodal Processing Tools
          {
            name: 'process_multimodal',
            description: 'Process multimodal content (text, images, audio, video) with AI analysis',
            inputSchema: {
              type: 'object',
              properties: {
                content_type: { 
                  type: 'string', 
                  enum: ['text', 'image', 'audio', 'video', 'document'],
                  description: 'Type of content to process'
                },
                content_data: { type: 'string', description: 'Base64 encoded content or file path' },
                analysis_type: {
                  type: 'string',
                  enum: ['describe', 'analyze', 'extract_text', 'sentiment', 'classify', 'summarize'],
                  description: 'Type of analysis to perform'
                },
                model_preference: { type: 'string', description: 'Preferred AI model for analysis' }
              },
              required: ['content_type', 'content_data', 'analysis_type']
            }
          },

          // RAG Tools
          {
            name: 'rag_query',
            description: 'Query documents using RAG (Retrieval-Augmented Generation) with vector search',
            inputSchema: {
              type: 'object',
              properties: {
                query: { type: 'string', description: 'Natural language query' },
                collection: { type: 'string', description: 'Document collection to search' },
                top_k: { type: 'number', description: 'Number of top results to retrieve', default: 5 },
                include_sources: { type: 'boolean', description: 'Include source documents', default: true },
                ai_model: { type: 'string', description: 'AI model for response generation', default: 'chatgpt' }
              },
              required: ['query']
            }
          },

          {
            name: 'rag_ingest',
            description: 'Ingest documents into RAG vector database',
            inputSchema: {
              type: 'object',
              properties: {
                documents: { 
                  type: 'array',
                  items: { type: 'string' },
                  description: 'Document paths or URLs to ingest'
                },
                collection: { type: 'string', description: 'Collection name to store documents' },
                chunk_size: { type: 'number', description: 'Text chunk size for processing', default: 1000 },
                chunk_overlap: { type: 'number', description: 'Overlap between chunks', default: 200 }
              },
              required: ['documents']
            }
          },

          // Perplexity Search Tools
          {
            name: 'perplexity_search',
            description: 'Search and get AI-powered answers using Perplexity',
            inputSchema: {
              type: 'object',
              properties: {
                query: { type: 'string', description: 'Search query' },
                focus: { 
                  type: 'string',
                  enum: ['academic', 'writing', 'wolframalpha', 'youtube', 'reddit'],
                  description: 'Search focus area'
                },
                include_citations: { type: 'boolean', description: 'Include source citations', default: true },
                response_format: { type: 'string', enum: ['text', 'json'], default: 'text' }
              },
              required: ['query']
            }
          },

          // MLOps Integration Tools
          {
            name: 'mlops_deploy_model',
            description: 'Deploy ML model to production using MLOps pipeline',
            inputSchema: {
              type: 'object',
              properties: {
                model_name: { type: 'string', description: 'Name of the model to deploy' },
                model_version: { type: 'string', description: 'Model version' },
                deployment_target: { 
                  type: 'string',
                  enum: ['kubernetes', 'docker', 'serverless', 'edge'],
                  description: 'Deployment target environment'
                },
                scaling_config: {
                  type: 'object',
                  properties: {
                    min_replicas: { type: 'number', default: 1 },
                    max_replicas: { type: 'number', default: 10 },
                    cpu_request: { type: 'string', default: '100m' },
                    memory_request: { type: 'string', default: '256Mi' }
                  }
                },
                monitoring_enabled: { type: 'boolean', default: true }
              },
              required: ['model_name', 'deployment_target']
            }
          },

          {
            name: 'mlops_create_pipeline',
            description: 'Create MLOps pipeline with AI-assisted configuration',
            inputSchema: {
              type: 'object',
              properties: {
                pipeline_name: { type: 'string', description: 'Name of the pipeline' },
                pipeline_type: {
                  type: 'string',
                  enum: ['training', 'inference', 'data_processing', 'full_ml_lifecycle'],
                  description: 'Type of pipeline to create'
                },
                data_source: { type: 'string', description: 'Data source configuration' },
                model_config: {
                  type: 'object',
                  properties: {
                    algorithm: { type: 'string', description: 'ML algorithm to use' },
                    hyperparameters: { type: 'object', description: 'Hyperparameters configuration' },
                    validation_split: { type: 'number', default: 0.2 }
                  }
                },
                schedule: { type: 'string', description: 'Cron schedule for pipeline execution' },
                ai_optimization: { type: 'boolean', description: 'Enable AI-assisted optimization', default: true }
              },
              required: ['pipeline_name', 'pipeline_type']
            }
          },

          // AI Model Comparison Tools
          {
            name: 'compare_ai_models',
            description: 'Compare responses from multiple AI models for the same prompt',
            inputSchema: {
              type: 'object',
              properties: {
                prompt: { type: 'string', description: 'Prompt to send to all models' },
                models: { 
                  type: 'array',
                  items: { type: 'string' },
                  description: 'Models to compare: chatgpt, claude, gemini',
                  default: ['chatgpt', 'claude', 'gemini']
                },
                evaluation_criteria: {
                  type: 'array',
                  items: { type: 'string' },
                  description: 'Criteria for comparison: accuracy, creativity, clarity, usefulness',
                  default: ['accuracy', 'clarity', 'usefulness']
                },
                include_metrics: { type: 'boolean', description: 'Include response metrics', default: true }
              },
              required: ['prompt']
            }
          },

          // Workflow Automation Tools
          {
            name: 'create_ai_workflow',
            description: 'Create automated workflow combining multiple AI services',
            inputSchema: {
              type: 'object',
              properties: {
                workflow_name: { type: 'string', description: 'Name of the workflow' },
                steps: {
                  type: 'array',
                  items: {
                    type: 'object',
                    properties: {
                      service: { type: 'string', description: 'AI service to use' },
                      action: { type: 'string', description: 'Action to perform' },
                      parameters: { type: 'object', description: 'Parameters for the action' },
                      condition: { type: 'string', description: 'Condition for step execution' }
                    }
                  }
                },
                trigger: { type: 'string', description: 'Workflow trigger condition' },
                output_format: { type: 'string', description: 'Desired output format' }
              },
              required: ['workflow_name', 'steps']
            }
          },

          // Data Analysis Tools
          {
            name: 'analyze_data_with_ai',
            description: 'Analyze datasets using multiple AI models and generate insights',
            inputSchema: {
              type: 'object',
              properties: {
                data_source: { type: 'string', description: 'Data source (file path, URL, or dataset ID)' },
                analysis_type: {
                  type: 'string',
                  enum: ['descriptive', 'predictive', 'prescriptive', 'exploratory'],
                  description: 'Type of analysis to perform'
                },
                ai_models: {
                  type: 'array',
                  items: { type: 'string' },
                  description: 'AI models to use for analysis',
                  default: ['chatgpt', 'claude']
                },
                visualization: { type: 'boolean', description: 'Generate visualizations', default: true },
                export_format: { type: 'string', enum: ['json', 'csv', 'pdf'], default: 'json' }
              },
              required: ['data_source', 'analysis_type']
            }
          }
        ] as Tool[]
      };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'chat_with_ai':
            return await this.handleChatWithAI(args);
          
          case 'process_multimodal':
            return await this.handleProcessMultimodal(args);
          
          case 'rag_query':
            return await this.handleRAGQuery(args);
          
          case 'rag_ingest':
            return await this.handleRAGIngest(args);
          
          case 'perplexity_search':
            return await this.handlePerplexitySearch(args);
          
          case 'mlops_deploy_model':
            return await this.handleMLOpsDeployModel(args);
          
          case 'mlops_create_pipeline':
            return await this.handleMLOpsCreatePipeline(args);
          
          case 'compare_ai_models':
            return await this.handleCompareAIModels(args);
          
          case 'create_ai_workflow':
            return await this.handleCreateAIWorkflow(args);
          
          case 'analyze_data_with_ai':
            return await this.handleAnalyzeDataWithAI(args);
          
          default:
            throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
        }
      } catch (error) {
        logger.error(`Error executing tool ${name}:`, error);
        throw new McpError(
          ErrorCode.InternalError,
          `Failed to execute tool ${name}: ${error.message}`
        );
      }
    });
  }

  private async handleChatWithAI(args: any) {
    const { message, models = ['all'], system_prompt, temperature = 0.7, max_tokens = 1000 } = args;

    const responses = await this.aiOrchestrator.chatWithMultipleModels({
      message,
      models: models.includes('all') ? ['chatgpt', 'claude', 'gemini'] : models,
      systemPrompt: system_prompt,
      temperature,
      maxTokens: max_tokens
    });

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          query: message,
          responses,
          timestamp: new Date().toISOString(),
          models_used: models
        }, null, 2)
      }]
    };
  }

  private async handleProcessMultimodal(args: any) {
    const { content_type, content_data, analysis_type, model_preference } = args;

    const result = await this.mediaProcessor.processMultimodal({
      contentType: content_type,
      contentData: content_data,
      analysisType: analysis_type,
      modelPreference: model_preference
    });

    const aiAnalysis = await this.aiOrchestrator.analyzeMultimodal({
      contentType: content_type,
      processedData: result,
      analysisType: analysis_type,
      modelPreference: model_preference
    });

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          content_type,
          analysis_type,
          processed_data: result,
          ai_analysis: aiAnalysis,
          timestamp: new Date().toISOString()
        }, null, 2)
      }]
    };
  }

  private async handleRAGQuery(args: any) {
    const { query, collection, top_k = 5, include_sources = true, ai_model = 'chatgpt' } = args;

    const ragResult = await this.ragService.query({
      query,
      collection,
      topK: top_k,
      includeSources: include_sources
    });

    // Generate AI response using retrieved context
    const aiResponse = await this.aiOrchestrator.generateWithContext({
      query,
      context: ragResult.documents,
      model: ai_model
    });

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          query,
          ai_response: aiResponse,
          retrieved_documents: ragResult.documents,
          sources: include_sources ? ragResult.sources : undefined,
          relevance_scores: ragResult.scores,
          timestamp: new Date().toISOString()
        }, null, 2)
      }]
    };
  }

  private async handleRAGIngest(args: any) {
    const { documents, collection, chunk_size = 1000, chunk_overlap = 200 } = args;

    const result = await this.ragService.ingestDocuments({
      documents,
      collection,
      chunkSize: chunk_size,
      chunkOverlap: chunk_overlap
    });

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          ingested_documents: result.documentCount,
          chunks_created: result.chunkCount,
          collection,
          status: 'completed',
          timestamp: new Date().toISOString()
        }, null, 2)
      }]
    };
  }

  private async handlePerplexitySearch(args: any) {
    const { query, focus, include_citations = true, response_format = 'text' } = args;

    const result = await this.perplexityService.search({
      query,
      focus,
      includeCitations: include_citations
    });

    return {
      content: [{
        type: 'text',
        text: response_format === 'json' 
          ? JSON.stringify(result, null, 2)
          : `${result.answer}\n\n${include_citations ? 'Sources:\n' + result.citations.map(c => `- ${c}`).join('\n') : ''}`
      }]
    };
  }

  private async handleMLOpsDeployModel(args: any) {
    const { model_name, model_version, deployment_target, scaling_config, monitoring_enabled = true } = args;

    const deployment = await this.mlopsIntegration.deployModel({
      modelName: model_name,
      modelVersion: model_version,
      deploymentTarget: deployment_target,
      scalingConfig: scaling_config,
      monitoringEnabled: monitoring_enabled
    });

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          deployment_id: deployment.id,
          model_name,
          model_version,
          deployment_target,
          status: deployment.status,
          endpoints: deployment.endpoints,
          monitoring_dashboard: deployment.monitoringUrl,
          timestamp: new Date().toISOString()
        }, null, 2)
      }]
    };
  }

  private async handleMLOpsCreatePipeline(args: any) {
    const { pipeline_name, pipeline_type, data_source, model_config, schedule, ai_optimization = true } = args;

    // Use AI to optimize pipeline configuration
    let optimizedConfig = model_config;
    if (ai_optimization) {
      optimizedConfig = await this.aiOrchestrator.optimizePipelineConfig({
        pipelineType: pipeline_type,
        dataSource: data_source,
        modelConfig: model_config
      });
    }

    const pipeline = await this.mlopsIntegration.createPipeline({
      name: pipeline_name,
      type: pipeline_type,
      dataSource: data_source,
      modelConfig: optimizedConfig,
      schedule
    });

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          pipeline_id: pipeline.id,
          pipeline_name,
          pipeline_type,
          optimized_config: optimizedConfig,
          schedule,
          status: pipeline.status,
          dashboard_url: pipeline.dashboardUrl,
          ai_optimization_applied: ai_optimization,
          timestamp: new Date().toISOString()
        }, null, 2)
      }]
    };
  }

  private async handleCompareAIModels(args: any) {
    const { prompt, models = ['chatgpt', 'claude', 'gemini'], evaluation_criteria = ['accuracy', 'clarity', 'usefulness'], include_metrics = true } = args;

    const comparison = await this.aiOrchestrator.compareModels({
      prompt,
      models,
      evaluationCriteria: evaluation_criteria,
      includeMetrics: include_metrics
    });

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          prompt,
          models_compared: models,
          responses: comparison.responses,
          evaluation: comparison.evaluation,
          metrics: include_metrics ? comparison.metrics : undefined,
          recommendation: comparison.recommendation,
          timestamp: new Date().toISOString()
        }, null, 2)
      }]
    };
  }

  private async handleCreateAIWorkflow(args: any) {
    const { workflow_name, steps, trigger, output_format } = args;

    const workflow = await this.aiOrchestrator.createWorkflow({
      name: workflow_name,
      steps,
      trigger,
      outputFormat: output_format
    });

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          workflow_id: workflow.id,
          workflow_name,
          steps_count: steps.length,
          trigger,
          output_format,
          status: workflow.status,
          execution_url: workflow.executionUrl,
          timestamp: new Date().toISOString()
        }, null, 2)
      }]
    };
  }

  private async handleAnalyzeDataWithAI(args: any) {
    const { data_source, analysis_type, ai_models = ['chatgpt', 'claude'], visualization = true, export_format = 'json' } = args;

    const analysis = await this.aiOrchestrator.analyzeDataset({
      dataSource: data_source,
      analysisType: analysis_type,
      aiModels: ai_models,
      generateVisualization: visualization
    });

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          data_source,
          analysis_type,
          ai_models_used: ai_models,
          insights: analysis.insights,
          recommendations: analysis.recommendations,
          visualizations: visualization ? analysis.visualizations : undefined,
          export_format,
          timestamp: new Date().toISOString()
        }, null, 2)
      }]
    };
  }

  private setupErrorHandling() {
    this.server.onerror = (error) => {
      logger.error('MCP Server error:', error);
    };

    process.on('SIGINT', async () => {
      logger.info('Shutting down MCP server...');
      await this.cleanup();
      process.exit(0);
    });
  }

  private async cleanup() {
    try {
      await this.ragService.cleanup();
      await this.mlopsIntegration.cleanup();
      await this.mediaProcessor.cleanup();
      logger.info('Cleanup completed successfully');
    } catch (error) {
      logger.error('Error during cleanup:', error);
    }
  }

  public async start() {
    try {
      const transport = new StdioServerTransport();
      await this.server.connect(transport);
      logger.info('MCP Server started successfully');
    } catch (error) {
      logger.error('Failed to start MCP server:', error);
      throw error;
    }
  }
}

// Start the server
const server = new MLOpsMCPServer();
server.start().catch((error) => {
  logger.error('Failed to start MCP server:', error);
  process.exit(1);
});