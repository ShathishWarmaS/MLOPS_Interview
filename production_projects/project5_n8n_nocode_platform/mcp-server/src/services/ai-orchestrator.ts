import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { logger } from '../utils/logger.js';

interface AIConfig {
  openai?: { apiKey: string };
  anthropic?: { apiKey: string };
  google?: { apiKey: string };
}

interface ChatRequest {
  message: string;
  models: string[];
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
}

interface MultimodalRequest {
  contentType: string;
  processedData: any;
  analysisType: string;
  modelPreference?: string;
}

interface ModelComparison {
  prompt: string;
  models: string[];
  evaluationCriteria: string[];
  includeMetrics: boolean;
}

interface WorkflowRequest {
  name: string;
  steps: any[];
  trigger: string;
  outputFormat: string;
}

interface DataAnalysisRequest {
  dataSource: string;
  analysisType: string;
  aiModels: string[];
  generateVisualization: boolean;
}

export class MultimodalAIOrchestrator {
  private openai: OpenAI;
  private anthropic: Anthropic;
  private google: GoogleGenerativeAI;
  private workflows: Map<string, any> = new Map();

  constructor(config: AIConfig) {
    if (config.openai) {
      this.openai = new OpenAI({
        apiKey: config.openai.apiKey,
      });
    }

    if (config.anthropic) {
      this.anthropic = new Anthropic({
        apiKey: config.anthropic.apiKey,
      });
    }

    if (config.google) {
      this.google = new GoogleGenerativeAI(config.google.apiKey);
    }

    logger.info('AI Orchestrator initialized with available models');
  }

  async chatWithMultipleModels(request: ChatRequest) {
    const { message, models, systemPrompt, temperature = 0.7, maxTokens = 1000 } = request;
    const responses: Record<string, any> = {};

    const promises = models.map(async (model) => {
      try {
        let response;
        const startTime = Date.now();

        switch (model) {
          case 'chatgpt':
            if (this.openai) {
              const completion = await this.openai.chat.completions.create({
                model: 'gpt-4-turbo-preview',
                messages: [
                  ...(systemPrompt ? [{ role: 'system' as const, content: systemPrompt }] : []),
                  { role: 'user' as const, content: message }
                ],
                temperature,
                max_tokens: maxTokens,
              });
              response = {
                content: completion.choices[0].message.content,
                usage: completion.usage,
                model: completion.model,
                responseTime: Date.now() - startTime
              };
            }
            break;

          case 'claude':
            if (this.anthropic) {
              const completion = await this.anthropic.messages.create({
                model: 'claude-3-sonnet-20240229',
                max_tokens: maxTokens,
                temperature,
                system: systemPrompt,
                messages: [
                  { role: 'user', content: message }
                ],
              });
              response = {
                content: completion.content[0].type === 'text' ? completion.content[0].text : '',
                usage: completion.usage,
                model: completion.model,
                responseTime: Date.now() - startTime
              };
            }
            break;

          case 'gemini':
            if (this.google) {
              const geminiModel = this.google.getGenerativeModel({ model: 'gemini-pro' });
              const result = await geminiModel.generateContent({
                contents: [
                  ...(systemPrompt ? [{ role: 'user', parts: [{ text: systemPrompt }] }] : []),
                  { role: 'user', parts: [{ text: message }] }
                ],
                generationConfig: {
                  temperature,
                  maxOutputTokens: maxTokens,
                }
              });
              response = {
                content: result.response.text(),
                usage: result.response.usageMetadata,
                model: 'gemini-pro',
                responseTime: Date.now() - startTime
              };
            }
            break;

          default:
            throw new Error(`Unsupported model: ${model}`);
        }

        responses[model] = response;
      } catch (error) {
        logger.error(`Error with model ${model}:`, error);
        responses[model] = {
          error: error.message,
          model,
          responseTime: Date.now() - startTime
        };
      }
    });

    await Promise.all(promises);
    return responses;
  }

  async analyzeMultimodal(request: MultimodalRequest) {
    const { contentType, processedData, analysisType, modelPreference = 'chatgpt' } = request;

    try {
      let prompt = '';
      let additionalData = null;

      switch (contentType) {
        case 'image':
          prompt = `Analyze this image and provide a ${analysisType} analysis. The image has been processed with the following data: ${JSON.stringify(processedData)}`;
          additionalData = processedData.base64Data;
          break;

        case 'audio':
          prompt = `Analyze this audio content and provide a ${analysisType} analysis. Audio processing results: ${JSON.stringify(processedData)}`;
          break;

        case 'video':
          prompt = `Analyze this video content and provide a ${analysisType} analysis. Video processing results: ${JSON.stringify(processedData)}`;
          break;

        case 'document':
          prompt = `Analyze this document and provide a ${analysisType} analysis. Document content: ${processedData.text}`;
          break;

        case 'text':
          prompt = `Analyze this text and provide a ${analysisType} analysis. Text content: ${processedData.text}`;
          break;

        default:
          throw new Error(`Unsupported content type: ${contentType}`);
      }

      // Use preferred model or fallback to available models
      const models = modelPreference ? [modelPreference] : ['chatgpt', 'claude', 'gemini'];
      const responses = await this.chatWithMultipleModels({
        message: prompt,
        models,
        systemPrompt: `You are an expert in ${analysisType} analysis. Provide detailed, accurate, and actionable insights.`,
        temperature: 0.3,
        maxTokens: 1500
      });

      return {
        analysis_type: analysisType,
        content_type: contentType,
        results: responses,
        processed_data: processedData,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      logger.error('Error in multimodal analysis:', error);
      throw error;
    }
  }

  async generateWithContext(request: { query: string; context: any[]; model: string }) {
    const { query, context, model } = request;

    const contextText = context.map(doc => doc.content).join('\n\n');
    const prompt = `Based on the following context, please answer the query:

Context:
${contextText}

Query: ${query}

Please provide a comprehensive answer based on the context provided.`;

    const response = await this.chatWithMultipleModels({
      message: prompt,
      models: [model],
      systemPrompt: 'You are a helpful assistant that answers questions based on provided context. Always cite your sources when possible.',
      temperature: 0.3,
      maxTokens: 1500
    });

    return response[model];
  }

  async optimizePipelineConfig(request: { pipelineType: string; dataSource: string; modelConfig: any }) {
    const { pipelineType, dataSource, modelConfig } = request;

    const optimizationPrompt = `
    You are an MLOps expert. Optimize the following ML pipeline configuration:

    Pipeline Type: ${pipelineType}
    Data Source: ${dataSource}
    Current Model Config: ${JSON.stringify(modelConfig, null, 2)}

    Please provide optimized configuration including:
    1. Best practices for this pipeline type
    2. Recommended hyperparameters
    3. Validation strategies
    4. Performance optimizations
    5. Monitoring recommendations

    Return the response in JSON format.
    `;

    const response = await this.chatWithMultipleModels({
      message: optimizationPrompt,
      models: ['chatgpt'],
      systemPrompt: 'You are an expert MLOps engineer. Provide practical, production-ready recommendations.',
      temperature: 0.2,
      maxTokens: 2000
    });

    try {
      const optimizedConfig = JSON.parse(response.chatgpt.content);
      return optimizedConfig;
    } catch (error) {
      logger.error('Error parsing optimized config:', error);
      return modelConfig; // Return original config if parsing fails
    }
  }

  async compareModels(request: ModelComparison) {
    const { prompt, models, evaluationCriteria, includeMetrics } = request;

    // Get responses from all models
    const responses = await this.chatWithMultipleModels({
      message: prompt,
      models,
      temperature: 0.7,
      maxTokens: 1000
    });

    // Use ChatGPT to evaluate and compare responses
    const evaluationPrompt = `
    Compare and evaluate the following AI model responses based on these criteria: ${evaluationCriteria.join(', ')}

    Original Prompt: ${prompt}

    Responses:
    ${Object.entries(responses).map(([model, response]) => `
    ${model.toUpperCase()}:
    ${typeof response === 'object' && response.content ? response.content : 'Error: ' + (response.error || 'Unknown error')}
    `).join('\n')}

    Please provide:
    1. Evaluation scores (1-10) for each model on each criterion
    2. Overall ranking of models
    3. Detailed comparison highlighting strengths and weaknesses
    4. Recommendation for best model for this type of query

    Format your response as JSON with the following structure:
    {
      "evaluations": {
        "model_name": {
          "criterion_name": score,
          "overall_score": score,
          "strengths": ["strength1", "strength2"],
          "weaknesses": ["weakness1", "weakness2"]
        }
      },
      "ranking": ["model1", "model2", "model3"],
      "recommendation": "recommended_model",
      "reasoning": "detailed explanation"
    }
    `;

    const evaluation = await this.chatWithMultipleModels({
      message: evaluationPrompt,
      models: ['chatgpt'],
      systemPrompt: 'You are an expert AI evaluator. Provide objective, detailed comparisons.',
      temperature: 0.2,
      maxTokens: 2000
    });

    let parsedEvaluation;
    try {
      parsedEvaluation = JSON.parse(evaluation.chatgpt.content);
    } catch (error) {
      parsedEvaluation = { error: 'Failed to parse evaluation' };
    }

    return {
      responses,
      evaluation: parsedEvaluation,
      metrics: includeMetrics ? this.calculateResponseMetrics(responses) : undefined,
      recommendation: parsedEvaluation.recommendation || 'Unable to determine'
    };
  }

  async createWorkflow(request: WorkflowRequest) {
    const { name, steps, trigger, outputFormat } = request;

    const workflowId = `workflow_${Date.now()}`;
    const workflow = {
      id: workflowId,
      name,
      steps,
      trigger,
      outputFormat,
      status: 'created',
      createdAt: new Date().toISOString(),
      executionUrl: `/workflows/${workflowId}/execute`
    };

    this.workflows.set(workflowId, workflow);

    logger.info(`Created workflow: ${name} with ${steps.length} steps`);
    return workflow;
  }

  async executeWorkflow(workflowId: string, input: any) {
    const workflow = this.workflows.get(workflowId);
    if (!workflow) {
      throw new Error(`Workflow not found: ${workflowId}`);
    }

    const results = [];
    let currentInput = input;

    for (const step of workflow.steps) {
      try {
        const stepResult = await this.executeWorkflowStep(step, currentInput);
        results.push(stepResult);
        currentInput = stepResult.output;
      } catch (error) {
        logger.error(`Error executing step ${step.action}:`, error);
        results.push({ error: error.message, step: step.action });
        break;
      }
    }

    return {
      workflowId,
      results,
      status: 'completed',
      executedAt: new Date().toISOString()
    };
  }

  private async executeWorkflowStep(step: any, input: any) {
    const { service, action, parameters } = step;

    switch (service) {
      case 'openai':
        return await this.executeOpenAIStep(action, parameters, input);
      case 'anthropic':
        return await this.executeAnthropicStep(action, parameters, input);
      case 'google':
        return await this.executeGoogleStep(action, parameters, input);
      default:
        throw new Error(`Unsupported service: ${service}`);
    }
  }

  private async executeOpenAIStep(action: string, parameters: any, input: any) {
    // Implementation for OpenAI workflow steps
    switch (action) {
      case 'chat':
        const response = await this.openai.chat.completions.create({
          model: parameters.model || 'gpt-4',
          messages: [
            { role: 'user', content: input.message || input }
          ],
          ...parameters
        });
        return {
          service: 'openai',
          action,
          output: response.choices[0].message.content,
          usage: response.usage
        };
      default:
        throw new Error(`Unsupported OpenAI action: ${action}`);
    }
  }

  private async executeAnthropicStep(action: string, parameters: any, input: any) {
    // Implementation for Anthropic workflow steps
    switch (action) {
      case 'chat':
        const response = await this.anthropic.messages.create({
          model: parameters.model || 'claude-3-sonnet-20240229',
          max_tokens: parameters.max_tokens || 1000,
          messages: [
            { role: 'user', content: input.message || input }
          ],
          ...parameters
        });
        return {
          service: 'anthropic',
          action,
          output: response.content[0].type === 'text' ? response.content[0].text : '',
          usage: response.usage
        };
      default:
        throw new Error(`Unsupported Anthropic action: ${action}`);
    }
  }

  private async executeGoogleStep(action: string, parameters: any, input: any) {
    // Implementation for Google workflow steps
    switch (action) {
      case 'generate':
        const model = this.google.getGenerativeModel({ model: parameters.model || 'gemini-pro' });
        const response = await model.generateContent(input.message || input);
        return {
          service: 'google',
          action,
          output: response.response.text(),
          usage: response.response.usageMetadata
        };
      default:
        throw new Error(`Unsupported Google action: ${action}`);
    }
  }

  async analyzeDataset(request: DataAnalysisRequest) {
    const { dataSource, analysisType, aiModels, generateVisualization } = request;

    // First, let's analyze the data structure
    const dataAnalysisPrompt = `
    Analyze the following dataset and provide insights based on the analysis type: ${analysisType}

    Data Source: ${dataSource}

    Please provide:
    1. Dataset overview and structure
    2. Key statistical insights
    3. Patterns and trends
    4. Anomalies or outliers
    5. Recommendations for further analysis
    6. Suggested visualizations (if applicable)

    Analysis Type: ${analysisType}
    `;

    const responses = await this.chatWithMultipleModels({
      message: dataAnalysisPrompt,
      models: aiModels,
      systemPrompt: 'You are a data scientist expert. Provide detailed, actionable insights.',
      temperature: 0.3,
      maxTokens: 2000
    });

    const insights = Object.entries(responses).map(([model, response]) => ({
      model,
      analysis: response.content || response.error,
      confidence: response.error ? 0 : 0.8 // Simple confidence score
    }));

    // Generate recommendations based on all model responses
    const recommendations = await this.generateDataRecommendations(insights, analysisType);

    return {
      dataSource,
      analysisType,
      insights,
      recommendations,
      visualizations: generateVisualization ? await this.generateVisualizationSuggestions(insights) : undefined,
      timestamp: new Date().toISOString()
    };
  }

  private async generateDataRecommendations(insights: any[], analysisType: string) {
    const combinedInsights = insights.map(i => i.analysis).join('\n\n');
    
    const recommendationPrompt = `
    Based on the following data analysis insights, provide actionable recommendations:

    Analysis Type: ${analysisType}
    
    Combined Insights:
    ${combinedInsights}

    Please provide:
    1. Top 3 actionable recommendations
    2. Potential risks and mitigation strategies
    3. Next steps for deeper analysis
    4. Implementation priorities

    Format as JSON with clear structure.
    `;

    const response = await this.chatWithMultipleModels({
      message: recommendationPrompt,
      models: ['chatgpt'],
      systemPrompt: 'You are a senior data consultant. Provide practical, implementable recommendations.',
      temperature: 0.2,
      maxTokens: 1500
    });

    try {
      return JSON.parse(response.chatgpt.content);
    } catch (error) {
      return { error: 'Failed to parse recommendations' };
    }
  }

  private async generateVisualizationSuggestions(insights: any[]) {
    const visualizationPrompt = `
    Based on these data analysis insights, suggest appropriate visualizations:

    Insights:
    ${insights.map(i => i.analysis).join('\n\n')}

    Please provide visualization suggestions in JSON format:
    {
      "charts": [
        {
          "type": "chart_type",
          "title": "chart_title",
          "description": "why this chart is useful",
          "data_requirements": "what data is needed"
        }
      ]
    }
    `;

    const response = await this.chatWithMultipleModels({
      message: visualizationPrompt,
      models: ['chatgpt'],
      systemPrompt: 'You are a data visualization expert. Suggest the most effective charts.',
      temperature: 0.3,
      maxTokens: 1000
    });

    try {
      return JSON.parse(response.chatgpt.content);
    } catch (error) {
      return { error: 'Failed to generate visualization suggestions' };
    }
  }

  private calculateResponseMetrics(responses: Record<string, any>) {
    const metrics: Record<string, any> = {};

    Object.entries(responses).forEach(([model, response]) => {
      if (response.error) {
        metrics[model] = {
          error: true,
          errorMessage: response.error
        };
      } else {
        metrics[model] = {
          responseTime: response.responseTime,
          contentLength: response.content?.length || 0,
          tokensUsed: response.usage?.total_tokens || 0,
          model: response.model
        };
      }
    });

    return metrics;
  }
}