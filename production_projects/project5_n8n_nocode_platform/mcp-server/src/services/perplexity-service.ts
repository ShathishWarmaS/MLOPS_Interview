import axios from 'axios';
import { logger } from '../utils/logger.js';

interface PerplexityConfig {
  apiKey: string;
  baseUrl?: string;
}

interface SearchRequest {
  query: string;
  focus?: string;
  includeCitations?: boolean;
  model?: string;
  temperature?: number;
  maxTokens?: number;
}

interface SearchResponse {
  answer: string;
  citations: string[];
  sources: Array<{
    title: string;
    url: string;
    snippet: string;
  }>;
  query: string;
  focus?: string;
  responseTime: number;
}

interface ChatRequest {
  messages: Array<{
    role: 'system' | 'user' | 'assistant';
    content: string;
  }>;
  model?: string;
  temperature?: number;
  maxTokens?: number;
}

export class PerplexityService {
  private apiKey: string;
  private baseUrl: string;
  private httpClient: any;

  constructor(config: PerplexityConfig) {
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || 'https://api.perplexity.ai';
    
    this.httpClient = axios.create({
      baseURL: this.baseUrl,
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    });

    logger.info('Perplexity service initialized');
  }

  async search(request: SearchRequest): Promise<SearchResponse> {
    const {
      query,
      focus,
      includeCitations = true,
      model = 'llama-3.1-sonar-small-128k-online',
      temperature = 0.2,
      maxTokens = 1000
    } = request;

    const startTime = Date.now();

    try {
      // Prepare messages for Perplexity API
      const messages = [
        {
          role: 'system' as const,
          content: this.getSystemPrompt(focus, includeCitations)
        },
        {
          role: 'user' as const,
          content: query
        }
      ];

      const response = await this.httpClient.post('/chat/completions', {
        model,
        messages,
        temperature,
        max_tokens: maxTokens,
        stream: false
      });

      const responseTime = Date.now() - startTime;
      const answer = response.data.choices[0].message.content;

      // Parse citations from response
      const citations = this.extractCitations(answer);
      const sources = this.extractSources(response.data.citations || []);

      return {
        answer: this.cleanAnswer(answer),
        citations,
        sources,
        query,
        focus,
        responseTime
      };
    } catch (error) {
      logger.error('Error in Perplexity search:', error);
      throw error;
    }
  }

  async chat(request: ChatRequest): Promise<any> {
    const {
      messages,
      model = 'llama-3.1-sonar-small-128k-chat',
      temperature = 0.7,
      maxTokens = 1000
    } = request;

    const startTime = Date.now();

    try {
      const response = await this.httpClient.post('/chat/completions', {
        model,
        messages,
        temperature,
        max_tokens: maxTokens,
        stream: false
      });

      const responseTime = Date.now() - startTime;

      return {
        content: response.data.choices[0].message.content,
        usage: response.data.usage,
        model: response.data.model,
        responseTime
      };
    } catch (error) {
      logger.error('Error in Perplexity chat:', error);
      throw error;
    }
  }

  async searchWithContext(request: {
    query: string;
    context: string;
    focus?: string;
    includeCitations?: boolean;
  }): Promise<SearchResponse> {
    const { query, context, focus, includeCitations = true } = request;

    const contextualQuery = `Context: ${context}\n\nQuery: ${query}\n\nPlease answer the query based on the provided context and any additional relevant information you can find.`;

    return await this.search({
      query: contextualQuery,
      focus,
      includeCitations
    });
  }

  async multiSearch(request: {
    queries: string[];
    focus?: string;
    includeCitations?: boolean;
  }): Promise<SearchResponse[]> {
    const { queries, focus, includeCitations = true } = request;

    const promises = queries.map(query => 
      this.search({ query, focus, includeCitations })
    );

    try {
      const results = await Promise.all(promises);
      return results;
    } catch (error) {
      logger.error('Error in multi-search:', error);
      throw error;
    }
  }

  async summarizeSearch(request: {
    queries: string[];
    focus?: string;
    summaryPrompt?: string;
  }): Promise<{
    summary: string;
    individualResults: SearchResponse[];
    combinedSources: string[];
  }> {
    const { queries, focus, summaryPrompt } = request;

    // Perform multiple searches
    const individualResults = await this.multiSearch({
      queries,
      focus,
      includeCitations: true
    });

    // Combine all answers
    const combinedAnswers = individualResults.map((result, index) => 
      `Query ${index + 1}: ${queries[index]}\nAnswer: ${result.answer}\n`
    ).join('\n');

    // Create summary
    const summaryQuery = summaryPrompt || 
      `Please provide a comprehensive summary of the following search results:\n\n${combinedAnswers}`;

    const summaryResult = await this.chat({
      messages: [
        {
          role: 'system',
          content: 'You are a skilled researcher who can synthesize information from multiple sources into coherent summaries.'
        },
        {
          role: 'user',
          content: summaryQuery
        }
      ],
      temperature: 0.3,
      maxTokens: 1500
    });

    // Combine all sources
    const combinedSources = [...new Set(
      individualResults.flatMap(result => result.citations)
    )];

    return {
      summary: summaryResult.content,
      individualResults,
      combinedSources
    };
  }

  async factCheck(request: {
    statement: string;
    context?: string;
  }): Promise<{
    isFactual: boolean;
    confidence: number;
    explanation: string;
    sources: string[];
  }> {
    const { statement, context } = request;

    let query = `Please fact-check the following statement: "${statement}"`;
    if (context) {
      query += `\n\nContext: ${context}`;
    }
    query += '\n\nProvide a detailed analysis of the factual accuracy, including evidence and sources.';

    const result = await this.search({
      query,
      focus: 'academic',
      includeCitations: true
    });

    // Simple fact-checking logic (in production, you'd use more sophisticated NLP)
    const answer = result.answer.toLowerCase();
    const factualWords = ['true', 'correct', 'accurate', 'confirmed', 'verified'];
    const nonFactualWords = ['false', 'incorrect', 'inaccurate', 'misleading', 'wrong'];

    const factualScore = factualWords.reduce((score, word) => 
      score + (answer.includes(word) ? 1 : 0), 0
    );
    const nonFactualScore = nonFactualWords.reduce((score, word) => 
      score + (answer.includes(word) ? 1 : 0), 0
    );

    const isFactual = factualScore > nonFactualScore;
    const confidence = Math.abs(factualScore - nonFactualScore) / (factualScore + nonFactualScore + 1);

    return {
      isFactual,
      confidence,
      explanation: result.answer,
      sources: result.citations
    };
  }

  async researchTopic(request: {
    topic: string;
    aspects: string[];
    depth?: 'shallow' | 'moderate' | 'deep';
  }): Promise<{
    overview: string;
    aspectAnalysis: Record<string, SearchResponse>;
    synthesis: string;
    recommendations: string[];
  }> {
    const { topic, aspects, depth = 'moderate' } = request;

    // Get overview
    const overviewResult = await this.search({
      query: `Provide a comprehensive overview of ${topic}`,
      focus: 'academic',
      includeCitations: true
    });

    // Research each aspect
    const aspectAnalysis: Record<string, SearchResponse> = {};
    for (const aspect of aspects) {
      const aspectQuery = `Analyze ${aspect} in the context of ${topic}. Provide detailed insights and current trends.`;
      aspectAnalysis[aspect] = await this.search({
        query: aspectQuery,
        focus: 'academic',
        includeCitations: true
      });
    }

    // Create synthesis
    const synthesisQuery = `Based on the following research about ${topic}:\n\n` +
      `Overview: ${overviewResult.answer}\n\n` +
      aspects.map(aspect => `${aspect}: ${aspectAnalysis[aspect].answer}`).join('\n\n') +
      '\n\nProvide a synthesis that connects these different aspects and identifies key insights.';

    const synthesisResult = await this.chat({
      messages: [
        {
          role: 'system',
          content: 'You are an expert researcher who can synthesize complex information and identify key insights.'
        },
        {
          role: 'user',
          content: synthesisQuery
        }
      ],
      temperature: 0.3,
      maxTokens: 1500
    });

    // Generate recommendations
    const recommendationsResult = await this.chat({
      messages: [
        {
          role: 'system',
          content: 'You are a strategic advisor who provides actionable recommendations based on research.'
        },
        {
          role: 'user',
          content: `Based on the research about ${topic}, provide 5-7 actionable recommendations for further exploration or practical application.`
        }
      ],
      temperature: 0.4,
      maxTokens: 800
    });

    const recommendations = recommendationsResult.content
      .split('\n')
      .filter(line => line.trim().match(/^\d+\.|^-|^\*/))
      .map(line => line.replace(/^\d+\.|^-|^\*/, '').trim())
      .filter(rec => rec.length > 0);

    return {
      overview: overviewResult.answer,
      aspectAnalysis,
      synthesis: synthesisResult.content,
      recommendations
    };
  }

  private getSystemPrompt(focus?: string, includeCitations?: boolean): string {
    let basePrompt = 'You are a helpful AI assistant that provides accurate and up-to-date information.';
    
    if (focus) {
      switch (focus) {
        case 'academic':
          basePrompt += ' Focus on academic and scholarly sources. Provide detailed, well-researched answers.';
          break;
        case 'writing':
          basePrompt += ' Focus on writing and creative content. Provide insights for writers and content creators.';
          break;
        case 'wolframalpha':
          basePrompt += ' Focus on computational and mathematical information. Provide precise, factual answers.';
          break;
        case 'youtube':
          basePrompt += ' Focus on video content and tutorials. Provide information about video resources.';
          break;
        case 'reddit':
          basePrompt += ' Focus on community discussions and user-generated content. Provide diverse perspectives.';
          break;
      }
    }

    if (includeCitations) {
      basePrompt += ' Always include citations and source references in your response.';
    }

    return basePrompt;
  }

  private extractCitations(text: string): string[] {
    // Extract citations from text (looking for URLs, references, etc.)
    const urlRegex = /https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)/g;
    const urls = text.match(urlRegex) || [];
    
    // Also look for reference patterns like [1], (Source: ...)
    const refRegex = /\[\d+\]|\(Source: [^)]+\)/g;
    const refs = text.match(refRegex) || [];
    
    return [...urls, ...refs];
  }

  private extractSources(citations: any[]): Array<{
    title: string;
    url: string;
    snippet: string;
  }> {
    return citations.map(citation => ({
      title: citation.title || 'Unknown Title',
      url: citation.url || '#',
      snippet: citation.snippet || citation.text || ''
    }));
  }

  private cleanAnswer(answer: string): string {
    // Remove excessive citations markers and clean up the answer
    return answer
      .replace(/\[\d+\]/g, '')
      .replace(/\(Source: [^)]+\)/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  async healthCheck(): Promise<{
    status: string;
    apiKey: boolean;
    responseTime: number;
  }> {
    const startTime = Date.now();
    
    try {
      await this.chat({
        messages: [
          { role: 'user', content: 'Hello' }
        ],
        maxTokens: 10
      });
      
      return {
        status: 'healthy',
        apiKey: true,
        responseTime: Date.now() - startTime
      };
    } catch (error) {
      return {
        status: 'unhealthy',
        apiKey: false,
        responseTime: Date.now() - startTime
      };
    }
  }
}
