import { ChromaClient } from 'chromadb';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { DocxLoader } from 'langchain/document_loaders/fs/docx';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio';
import fs from 'fs';
import path from 'path';
import { logger } from '../utils/logger.js';

interface RAGConfig {
  vectorStore: string;
  embeddingModel: string;
  chromaUrl?: string;
  persistDirectory?: string;
}

interface QueryRequest {
  query: string;
  collection?: string;
  topK?: number;
  includeSources?: boolean;
  threshold?: number;
}

interface IngestRequest {
  documents: string[];
  collection?: string;
  chunkSize?: number;
  chunkOverlap?: number;
  metadata?: Record<string, any>;
}

interface RAGQueryResult {
  documents: Array<{
    content: string;
    metadata: Record<string, any>;
    score: number;
  }>;
  sources: string[];
  scores: number[];
  query: string;
  collection: string;
}

interface IngestResult {
  documentCount: number;
  chunkCount: number;
  collection: string;
  status: string;
}

export class RAGService {
  private chromaClient: ChromaClient;
  private embeddings: OpenAIEmbeddings;
  private textSplitter: RecursiveCharacterTextSplitter;
  private collections: Map<string, any> = new Map();
  private config: RAGConfig;

  constructor(config: RAGConfig) {
    this.config = config;
    this.chromaClient = new ChromaClient({
      path: config.chromaUrl || 'http://localhost:8000'
    });

    this.embeddings = new OpenAIEmbeddings({
      modelName: config.embeddingModel || 'text-embedding-ada-002'
    });

    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200
    });

    logger.info('RAG Service initialized');
  }

  async initialize() {
    try {
      // Test connection to Chroma
      await this.chromaClient.heartbeat();
      logger.info('Connected to Chroma vector database');

      // Load existing collections
      const existingCollections = await this.chromaClient.listCollections();
      existingCollections.forEach(collection => {
        this.collections.set(collection.name, collection);
      });

      logger.info(`Loaded ${existingCollections.length} existing collections`);
    } catch (error) {
      logger.error('Failed to initialize RAG service:', error);
      throw error;
    }
  }

  async query(request: QueryRequest): Promise<RAGQueryResult> {
    const {
      query,
      collection = 'default',
      topK = 5,
      includeSources = true,
      threshold = 0.7
    } = request;

    try {
      // Get or create collection
      const chromaCollection = await this.getOrCreateCollection(collection);

      // Generate query embedding
      const queryEmbedding = await this.embeddings.embedQuery(query);

      // Search in vector database
      const results = await chromaCollection.query({
        queryEmbeddings: [queryEmbedding],
        nResults: topK,
        include: ['metadatas', 'documents', 'distances']
      });

      // Process results
      const documents = results.documents[0]?.map((doc: string, index: number) => ({
        content: doc,
        metadata: results.metadatas?.[0]?.[index] || {},
        score: 1 - (results.distances?.[0]?.[index] || 0) // Convert distance to similarity
      })) || [];

      // Filter by threshold
      const filteredDocuments = documents.filter(doc => doc.score >= threshold);

      const sources = includeSources 
        ? filteredDocuments.map(doc => doc.metadata.source || 'Unknown')
        : [];

      const scores = filteredDocuments.map(doc => doc.score);

      return {
        documents: filteredDocuments,
        sources,
        scores,
        query,
        collection
      };
    } catch (error) {
      logger.error('Error querying RAG service:', error);
      throw error;
    }
  }

  async ingestDocuments(request: IngestRequest): Promise<IngestResult> {
    const {
      documents,
      collection = 'default',
      chunkSize = 1000,
      chunkOverlap = 200,
      metadata = {}
    } = request;

    try {
      // Update text splitter configuration
      this.textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize,
        chunkOverlap
      });

      // Get or create collection
      const chromaCollection = await this.getOrCreateCollection(collection);

      let totalChunks = 0;
      let processedDocuments = 0;

      for (const docPath of documents) {
        try {
          // Load document
          const loader = this.getDocumentLoader(docPath);
          const docs = await loader.load();

          // Split into chunks
          const chunks = await this.textSplitter.splitDocuments(docs);

          // Process chunks
          const chunkTexts = chunks.map(chunk => chunk.pageContent);
          const chunkMetadatas = chunks.map((chunk, index) => ({
            ...metadata,
            source: docPath,
            chunkIndex: index,
            totalChunks: chunks.length,
            ingestedAt: new Date().toISOString()
          }));

          // Generate embeddings
          const embeddings = await this.embeddings.embedDocuments(chunkTexts);

          // Generate unique IDs for chunks
          const ids = chunks.map((_, index) => `${docPath}_chunk_${index}_${Date.now()}`);

          // Add to vector database
          await chromaCollection.add({
            ids,
            embeddings,
            documents: chunkTexts,
            metadatas: chunkMetadatas
          });

          totalChunks += chunks.length;
          processedDocuments++;

          logger.info(`Ingested ${chunks.length} chunks from ${docPath}`);
        } catch (error) {
          logger.error(`Error processing document ${docPath}:`, error);
          // Continue with other documents
        }
      }

      return {
        documentCount: processedDocuments,
        chunkCount: totalChunks,
        collection,
        status: 'completed'
      };
    } catch (error) {
      logger.error('Error ingesting documents:', error);
      throw error;
    }
  }

  private getDocumentLoader(docPath: string) {
    const extension = path.extname(docPath).toLowerCase();
    
    if (docPath.startsWith('http://') || docPath.startsWith('https://')) {
      return new CheerioWebBaseLoader(docPath);
    }

    switch (extension) {
      case '.pdf':
        return new PDFLoader(docPath);
      case '.docx':
        return new DocxLoader(docPath);
      case '.txt':
      case '.md':
        return new TextLoader(docPath);
      default:
        throw new Error(`Unsupported document format: ${extension}`);
    }
  }

  private async getOrCreateCollection(name: string) {
    if (this.collections.has(name)) {
      return this.collections.get(name);
    }

    try {
      // Try to get existing collection
      const collection = await this.chromaClient.getCollection({ name });
      this.collections.set(name, collection);
      return collection;
    } catch (error) {
      // Create new collection if it doesn't exist
      const collection = await this.chromaClient.createCollection({
        name,
        metadata: {
          description: `RAG collection for ${name}`,
          createdAt: new Date().toISOString()
        }
      });
      this.collections.set(name, collection);
      logger.info(`Created new collection: ${name}`);
      return collection;
    }
  }

  async listCollections() {
    const collections = await this.chromaClient.listCollections();
    return collections.map(collection => ({
      name: collection.name,
      metadata: collection.metadata,
      count: collection.count || 0
    }));
  }

  async deleteCollection(name: string) {
    try {
      await this.chromaClient.deleteCollection({ name });
      this.collections.delete(name);
      logger.info(`Deleted collection: ${name}`);
      return { success: true, message: `Collection ${name} deleted` };
    } catch (error) {
      logger.error(`Error deleting collection ${name}:`, error);
      throw error;
    }
  }

  async getCollectionStats(name: string) {
    try {
      const collection = await this.getOrCreateCollection(name);
      const count = await collection.count();
      
      return {
        name,
        documentCount: count,
        createdAt: collection.metadata?.createdAt,
        description: collection.metadata?.description
      };
    } catch (error) {
      logger.error(`Error getting collection stats for ${name}:`, error);
      throw error;
    }
  }

  async searchSimilar(request: {
    text: string;
    collection?: string;
    topK?: number;
    threshold?: number;
  }) {
    const { text, collection = 'default', topK = 5, threshold = 0.7 } = request;

    try {
      const chromaCollection = await this.getOrCreateCollection(collection);
      const embedding = await this.embeddings.embedQuery(text);

      const results = await chromaCollection.query({
        queryEmbeddings: [embedding],
        nResults: topK,
        include: ['metadatas', 'documents', 'distances']
      });

      const documents = results.documents[0]?.map((doc: string, index: number) => ({
        content: doc,
        metadata: results.metadatas?.[0]?.[index] || {},
        similarity: 1 - (results.distances?.[0]?.[index] || 0)
      })) || [];

      return documents.filter(doc => doc.similarity >= threshold);
    } catch (error) {
      logger.error('Error in similarity search:', error);
      throw error;
    }
  }

  async updateDocumentMetadata(request: {
    documentId: string;
    collection?: string;
    metadata: Record<string, any>;
  }) {
    const { documentId, collection = 'default', metadata } = request;

    try {
      const chromaCollection = await this.getOrCreateCollection(collection);
      
      await chromaCollection.update({
        ids: [documentId],
        metadatas: [metadata]
      });

      logger.info(`Updated metadata for document ${documentId}`);
      return { success: true, documentId, metadata };
    } catch (error) {
      logger.error(`Error updating document metadata:`, error);
      throw error;
    }
  }

  async deleteDocument(request: {
    documentId: string;
    collection?: string;
  }) {
    const { documentId, collection = 'default' } = request;

    try {
      const chromaCollection = await this.getOrCreateCollection(collection);
      
      await chromaCollection.delete({
        ids: [documentId]
      });

      logger.info(`Deleted document ${documentId} from collection ${collection}`);
      return { success: true, documentId, collection };
    } catch (error) {
      logger.error(`Error deleting document:`, error);
      throw error;
    }
  }

  async hybridSearch(request: {
    query: string;
    collection?: string;
    topK?: number;
    alpha?: number; // Balance between semantic and keyword search
    keywordWeight?: number;
  }) {
    const { 
      query, 
      collection = 'default', 
      topK = 5, 
      alpha = 0.7,
      keywordWeight = 0.3 
    } = request;

    try {
      // Semantic search
      const semanticResults = await this.query({
        query,
        collection,
        topK: topK * 2, // Get more results for hybrid ranking
        includeSources: true
      });

      // Keyword search (simple implementation)
      const keywordResults = await this.keywordSearch({
        query,
        collection,
        topK: topK * 2
      });

      // Combine and rank results
      const hybridResults = this.combineSearchResults(
        semanticResults.documents,
        keywordResults,
        alpha,
        keywordWeight
      );

      return {
        documents: hybridResults.slice(0, topK),
        query,
        collection,
        searchType: 'hybrid'
      };
    } catch (error) {
      logger.error('Error in hybrid search:', error);
      throw error;
    }
  }

  private async keywordSearch(request: {
    query: string;
    collection: string;
    topK: number;
  }) {
    // Simple keyword search implementation
    // In production, you might want to use a more sophisticated search engine
    const { query, collection, topK } = request;
    
    try {
      const chromaCollection = await this.getOrCreateCollection(collection);
      
      // Get all documents (this is not efficient for large collections)
      const allResults = await chromaCollection.get({
        include: ['metadatas', 'documents']
      });

      const queryTerms = query.toLowerCase().split(' ');
      const scoredDocuments = allResults.documents?.map((doc: string, index: number) => {
        const docLower = doc.toLowerCase();
        const score = queryTerms.reduce((acc, term) => {
          const matches = (docLower.match(new RegExp(term, 'g')) || []).length;
          return acc + matches;
        }, 0);

        return {
          content: doc,
          metadata: allResults.metadatas?.[index] || {},
          keywordScore: score
        };
      }) || [];

      return scoredDocuments
        .filter(doc => doc.keywordScore > 0)
        .sort((a, b) => b.keywordScore - a.keywordScore)
        .slice(0, topK);
    } catch (error) {
      logger.error('Error in keyword search:', error);
      return [];
    }
  }

  private combineSearchResults(
    semanticResults: any[],
    keywordResults: any[],
    alpha: number,
    keywordWeight: number
  ) {
    const combinedResults = new Map();

    // Add semantic results
    semanticResults.forEach(result => {
      combinedResults.set(result.content, {
        ...result,
        semanticScore: result.score || 0,
        keywordScore: 0,
        combinedScore: 0
      });
    });

    // Add keyword results
    keywordResults.forEach(result => {
      if (combinedResults.has(result.content)) {
        const existing = combinedResults.get(result.content);
        existing.keywordScore = result.keywordScore || 0;
      } else {
        combinedResults.set(result.content, {
          ...result,
          semanticScore: 0,
          keywordScore: result.keywordScore || 0,
          combinedScore: 0
        });
      }
    });

    // Calculate combined scores
    const results = Array.from(combinedResults.values()).map(result => ({
      ...result,
      combinedScore: (alpha * result.semanticScore) + (keywordWeight * result.keywordScore)
    }));

    return results.sort((a, b) => b.combinedScore - a.combinedScore);
  }

  async getDocumentsByMetadata(request: {
    collection?: string;
    metadata: Record<string, any>;
    limit?: number;
  }) {
    const { collection = 'default', metadata, limit = 10 } = request;

    try {
      const chromaCollection = await this.getOrCreateCollection(collection);
      
      const results = await chromaCollection.get({
        where: metadata,
        limit,
        include: ['metadatas', 'documents']
      });

      return results.documents?.map((doc: string, index: number) => ({
        content: doc,
        metadata: results.metadatas?.[index] || {}
      })) || [];
    } catch (error) {
      logger.error('Error getting documents by metadata:', error);
      throw error;
    }
  }

  async cleanup() {
    try {
      // Clean up any resources
      this.collections.clear();
      logger.info('RAG service cleanup completed');
    } catch (error) {
      logger.error('Error during RAG service cleanup:', error);
    }
  }
}