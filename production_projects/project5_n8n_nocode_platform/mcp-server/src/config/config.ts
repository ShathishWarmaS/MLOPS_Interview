import dotenv from 'dotenv';
import path from 'path';

// Load environment variables
dotenv.config();

interface DatabaseConfig {
  url: string;
  options: {
    useNewUrlParser: boolean;
    useUnifiedTopology: boolean;
  };
}

interface RedisConfig {
  host: string;
  port: number;
  password?: string;
  db: number;
}

interface AIConfig {
  apiKey: string;
  baseUrl?: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
}

interface RAGConfig {
  vectorStore: string;
  embeddingModel: string;
  chromaUrl?: string;
  persistDirectory?: string;
  chunkSize: number;
  chunkOverlap: number;
}

interface MLOpsConfig {
  mlflowUrl: string;
  kubeflowUrl: string;
  dockerRegistry: string;
  kubernetesNamespace: string;
}

interface MediaConfig {
  tempDir: string;
  maxFileSize: number;
  supportedFormats: {
    image: string[];
    video: string[];
    audio: string[];
    document: string[];
  };
}

interface SecurityConfig {
  jwtSecret: string;
  jwtExpiresIn: string;
  bcryptRounds: number;
  rateLimiting: {
    windowMs: number;
    maxRequests: number;
  };
  cors: {
    origins: string[];
    credentials: boolean;
  };
}

interface ServerConfig {
  port: number;
  host: string;
  env: string;
  logLevel: string;
  enableMetrics: boolean;
  enableHealthCheck: boolean;
}

interface Config {
  server: ServerConfig;
  database: DatabaseConfig;
  redis: RedisConfig;
  openai: AIConfig;
  anthropic: AIConfig;
  google: AIConfig;
  perplexity: AIConfig;
  rag: RAGConfig;
  mlops: MLOpsConfig;
  media: MediaConfig;
  security: SecurityConfig;
}

// Helper function to get required environment variable
function getRequiredEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Required environment variable ${name} is not set`);
  }
  return value;
}

// Helper function to get environment variable with default
function getEnvWithDefault(name: string, defaultValue: string): string {
  return process.env[name] || defaultValue;
}

// Helper function to get boolean environment variable
function getBooleanEnv(name: string, defaultValue: boolean): boolean {
  const value = process.env[name];
  if (value === undefined) return defaultValue;
  return value.toLowerCase() === 'true';
}

// Helper function to get number environment variable
function getNumberEnv(name: string, defaultValue: number): number {
  const value = process.env[name];
  if (value === undefined) return defaultValue;
  const parsed = parseInt(value, 10);
  return isNaN(parsed) ? defaultValue : parsed;
}

// Helper function to get array environment variable
function getArrayEnv(name: string, defaultValue: string[]): string[] {
  const value = process.env[name];
  if (!value) return defaultValue;
  return value.split(',').map(item => item.trim());
}

// Main configuration object
export const config: Config = {
  server: {
    port: getNumberEnv('PORT', 3000),
    host: getEnvWithDefault('HOST', '0.0.0.0'),
    env: getEnvWithDefault('NODE_ENV', 'development'),
    logLevel: getEnvWithDefault('LOG_LEVEL', 'info'),
    enableMetrics: getBooleanEnv('ENABLE_METRICS', true),
    enableHealthCheck: getBooleanEnv('ENABLE_HEALTH_CHECK', true)
  },

  database: {
    url: getEnvWithDefault('DATABASE_URL', 'mongodb://localhost:27017/mlops-mcp'),
    options: {
      useNewUrlParser: true,
      useUnifiedTopology: true
    }
  },

  redis: {
    host: getEnvWithDefault('REDIS_HOST', 'localhost'),
    port: getNumberEnv('REDIS_PORT', 6379),
    password: process.env.REDIS_PASSWORD,
    db: getNumberEnv('REDIS_DB', 0)
  },

  openai: {
    apiKey: getRequiredEnv('OPENAI_API_KEY'),
    baseUrl: getEnvWithDefault('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
    model: getEnvWithDefault('OPENAI_MODEL', 'gpt-4-turbo-preview'),
    maxTokens: getNumberEnv('OPENAI_MAX_TOKENS', 1000),
    temperature: parseFloat(getEnvWithDefault('OPENAI_TEMPERATURE', '0.7'))
  },

  anthropic: {
    apiKey: getRequiredEnv('ANTHROPIC_API_KEY'),
    baseUrl: getEnvWithDefault('ANTHROPIC_BASE_URL', 'https://api.anthropic.com'),
    model: getEnvWithDefault('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229'),
    maxTokens: getNumberEnv('ANTHROPIC_MAX_TOKENS', 1000),
    temperature: parseFloat(getEnvWithDefault('ANTHROPIC_TEMPERATURE', '0.7'))
  },

  google: {
    apiKey: getRequiredEnv('GOOGLE_API_KEY'),
    baseUrl: getEnvWithDefault('GOOGLE_BASE_URL', 'https://generativelanguage.googleapis.com'),
    model: getEnvWithDefault('GOOGLE_MODEL', 'gemini-pro'),
    maxTokens: getNumberEnv('GOOGLE_MAX_TOKENS', 1000),
    temperature: parseFloat(getEnvWithDefault('GOOGLE_TEMPERATURE', '0.7'))
  },

  perplexity: {
    apiKey: getRequiredEnv('PERPLEXITY_API_KEY'),
    baseUrl: getEnvWithDefault('PERPLEXITY_BASE_URL', 'https://api.perplexity.ai'),
    model: getEnvWithDefault('PERPLEXITY_MODEL', 'llama-3.1-sonar-small-128k-online'),
    maxTokens: getNumberEnv('PERPLEXITY_MAX_TOKENS', 1000),
    temperature: parseFloat(getEnvWithDefault('PERPLEXITY_TEMPERATURE', '0.2'))
  },

  rag: {
    vectorStore: getEnvWithDefault('RAG_VECTOR_STORE', 'chroma'),
    embeddingModel: getEnvWithDefault('RAG_EMBEDDING_MODEL', 'text-embedding-ada-002'),
    chromaUrl: getEnvWithDefault('CHROMA_URL', 'http://localhost:8000'),
    persistDirectory: getEnvWithDefault('RAG_PERSIST_DIR', './data/chroma'),
    chunkSize: getNumberEnv('RAG_CHUNK_SIZE', 1000),
    chunkOverlap: getNumberEnv('RAG_CHUNK_OVERLAP', 200)
  },

  mlops: {
    mlflowUrl: getEnvWithDefault('MLFLOW_URL', 'http://localhost:5000'),
    kubeflowUrl: getEnvWithDefault('KUBEFLOW_URL', 'http://localhost:8080'),
    dockerRegistry: getEnvWithDefault('DOCKER_REGISTRY', 'localhost:5000'),
    kubernetesNamespace: getEnvWithDefault('KUBERNETES_NAMESPACE', 'default')
  },

  media: {
    tempDir: getEnvWithDefault('MEDIA_TEMP_DIR', './tmp/media'),
    maxFileSize: getNumberEnv('MEDIA_MAX_FILE_SIZE', 100 * 1024 * 1024), // 100MB
    supportedFormats: {
      image: getArrayEnv('SUPPORTED_IMAGE_FORMATS', ['jpg', 'jpeg', 'png', 'gif', 'webp', 'tiff', 'bmp']),
      video: getArrayEnv('SUPPORTED_VIDEO_FORMATS', ['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv']),
      audio: getArrayEnv('SUPPORTED_AUDIO_FORMATS', ['mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a']),
      document: getArrayEnv('SUPPORTED_DOCUMENT_FORMATS', ['pdf', 'docx', 'txt', 'md', 'html'])
    }
  },

  security: {
    jwtSecret: getRequiredEnv('JWT_SECRET'),
    jwtExpiresIn: getEnvWithDefault('JWT_EXPIRES_IN', '7d'),
    bcryptRounds: getNumberEnv('BCRYPT_ROUNDS', 12),
    rateLimiting: {
      windowMs: getNumberEnv('RATE_LIMIT_WINDOW_MS', 15 * 60 * 1000), // 15 minutes
      maxRequests: getNumberEnv('RATE_LIMIT_MAX_REQUESTS', 100)
    },
    cors: {
      origins: getArrayEnv('CORS_ORIGINS', ['http://localhost:3000', 'http://localhost:3001']),
      credentials: getBooleanEnv('CORS_CREDENTIALS', true)
    }
  }
};

// Validation function
export function validateConfig(): void {
  const errors: string[] = [];

  // Validate required API keys
  if (!config.openai.apiKey) errors.push('OpenAI API key is required');
  if (!config.anthropic.apiKey) errors.push('Anthropic API key is required');
  if (!config.google.apiKey) errors.push('Google API key is required');
  if (!config.perplexity.apiKey) errors.push('Perplexity API key is required');
  if (!config.security.jwtSecret) errors.push('JWT secret is required');

  // Validate numeric values
  if (config.server.port <= 0 || config.server.port > 65535) {
    errors.push('Server port must be between 1 and 65535');
  }

  if (config.media.maxFileSize <= 0) {
    errors.push('Media max file size must be greater than 0');
  }

  if (config.rag.chunkSize <= 0) {
    errors.push('RAG chunk size must be greater than 0');
  }

  if (config.rag.chunkOverlap < 0) {
    errors.push('RAG chunk overlap must be non-negative');
  }

  if (config.rag.chunkOverlap >= config.rag.chunkSize) {
    errors.push('RAG chunk overlap must be less than chunk size');
  }

  // Validate URLs
  const urlFields = [
    { name: 'OpenAI Base URL', value: config.openai.baseUrl },
    { name: 'Anthropic Base URL', value: config.anthropic.baseUrl },
    { name: 'Google Base URL', value: config.google.baseUrl },
    { name: 'Perplexity Base URL', value: config.perplexity.baseUrl },
    { name: 'Chroma URL', value: config.rag.chromaUrl },
    { name: 'MLflow URL', value: config.mlops.mlflowUrl },
    { name: 'Kubeflow URL', value: config.mlops.kubeflowUrl }
  ];

  for (const field of urlFields) {
    if (field.value && !isValidUrl(field.value)) {
      errors.push(`${field.name} is not a valid URL: ${field.value}`);
    }
  }

  // Validate temperature values
  const temperatureFields = [
    { name: 'OpenAI temperature', value: config.openai.temperature },
    { name: 'Anthropic temperature', value: config.anthropic.temperature },
    { name: 'Google temperature', value: config.google.temperature },
    { name: 'Perplexity temperature', value: config.perplexity.temperature }
  ];

  for (const field of temperatureFields) {
    if (field.value !== undefined && (field.value < 0 || field.value > 2)) {
      errors.push(`${field.name} must be between 0 and 2: ${field.value}`);
    }
  }

  // Validate supported formats
  const formatFields = [
    { name: 'Image formats', value: config.media.supportedFormats.image },
    { name: 'Video formats', value: config.media.supportedFormats.video },
    { name: 'Audio formats', value: config.media.supportedFormats.audio },
    { name: 'Document formats', value: config.media.supportedFormats.document }
  ];

  for (const field of formatFields) {
    if (!Array.isArray(field.value) || field.value.length === 0) {
      errors.push(`${field.name} must be a non-empty array`);
    }
  }

  if (errors.length > 0) {
    throw new Error(`Configuration validation failed:\n${errors.join('\n')}`);
  }
}

// Helper function to validate URLs
function isValidUrl(urlString: string): boolean {
  try {
    new URL(urlString);
    return true;
  } catch {
    return false;
  }
}

// Environment-specific configurations
export const isDevelopment = config.server.env === 'development';
export const isProduction = config.server.env === 'production';
export const isTesting = config.server.env === 'test';

// Feature flags
export const features = {
  enableMetrics: config.server.enableMetrics,
  enableHealthCheck: config.server.enableHealthCheck,
  enableCors: true,
  enableRateLimiting: isProduction,
  enableRequestLogging: !isTesting,
  enableSwagger: isDevelopment,
  enableDebugMode: isDevelopment
};

// Export specific configurations for easy access
export const {
  server: serverConfig,
  database: databaseConfig,
  redis: redisConfig,
  openai: openaiConfig,
  anthropic: anthropicConfig,
  google: googleConfig,
  perplexity: perplexityConfig,
  rag: ragConfig,
  mlops: mlopsConfig,
  media: mediaConfig,
  security: securityConfig
} = config;

// Initialize configuration on import
try {
  validateConfig();
  console.log('Configuration validated successfully');
} catch (error) {
  console.error('Configuration validation failed:', error.message);
  process.exit(1);
}
