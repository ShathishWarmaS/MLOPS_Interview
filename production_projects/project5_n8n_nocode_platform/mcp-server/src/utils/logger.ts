import winston from 'winston';
import path from 'path';

// Define log levels
const logLevels = {
  error: 0,
  warn: 1,
  info: 2,
  http: 3,
  verbose: 4,
  debug: 5,
  silly: 6
};

// Define colors for each log level
const logColors = {
  error: 'red',
  warn: 'yellow',
  info: 'green',
  http: 'magenta',
  verbose: 'grey',
  debug: 'blue',
  silly: 'cyan'
};

// Add colors to winston
winston.addColors(logColors);

// Create custom format
const customFormat = winston.format.combine(
  winston.format.timestamp({
    format: 'YYYY-MM-DD HH:mm:ss'
  }),
  winston.format.errors({ stack: true }),
  winston.format.printf(({ level, message, timestamp, stack, ...meta }) => {
    const metaString = Object.keys(meta).length > 0 ? JSON.stringify(meta, null, 2) : '';
    if (stack) {
      return `${timestamp} [${level.toUpperCase()}]: ${message}\n${stack}${metaString ? `\n${metaString}` : ''}`;
    }
    return `${timestamp} [${level.toUpperCase()}]: ${message}${metaString ? `\n${metaString}` : ''}`;
  })
);

// Create console format with colors
const consoleFormat = winston.format.combine(
  winston.format.colorize({ all: true }),
  winston.format.timestamp({
    format: 'YYYY-MM-DD HH:mm:ss'
  }),
  winston.format.errors({ stack: true }),
  winston.format.printf(({ level, message, timestamp, stack, ...meta }) => {
    const metaString = Object.keys(meta).length > 0 ? JSON.stringify(meta, null, 2) : '';
    if (stack) {
      return `${timestamp} [${level}]: ${message}\n${stack}${metaString ? `\n${metaString}` : ''}`;
    }
    return `${timestamp} [${level}]: ${message}${metaString ? `\n${metaString}` : ''}`;
  })
);

// Get log level from environment or default to 'info'
const getLogLevel = (): string => {
  const level = process.env.LOG_LEVEL || 'info';
  return Object.keys(logLevels).includes(level) ? level : 'info';
};

// Create log directory if it doesn't exist
const logDir = path.join(process.cwd(), 'logs');
if (!require('fs').existsSync(logDir)) {
  require('fs').mkdirSync(logDir, { recursive: true });
}

// Create transports
const transports: winston.transport[] = [
  // Console transport
  new winston.transports.Console({
    level: getLogLevel(),
    format: consoleFormat,
    handleExceptions: true,
    handleRejections: true
  }),

  // File transport for all logs
  new winston.transports.File({
    filename: path.join(logDir, 'application.log'),
    level: 'info',
    format: customFormat,
    maxsize: 10 * 1024 * 1024, // 10MB
    maxFiles: 5,
    handleExceptions: true,
    handleRejections: true
  }),

  // File transport for errors only
  new winston.transports.File({
    filename: path.join(logDir, 'error.log'),
    level: 'error',
    format: customFormat,
    maxsize: 10 * 1024 * 1024, // 10MB
    maxFiles: 5,
    handleExceptions: true,
    handleRejections: true
  })
];

// Add debug file transport in development
if (process.env.NODE_ENV === 'development') {
  transports.push(
    new winston.transports.File({
      filename: path.join(logDir, 'debug.log'),
      level: 'debug',
      format: customFormat,
      maxsize: 10 * 1024 * 1024, // 10MB
      maxFiles: 3
    })
  );
}

// Create logger instance
export const logger = winston.createLogger({
  level: getLogLevel(),
  levels: logLevels,
  format: customFormat,
  transports,
  exitOnError: false
});

// Create a stream for HTTP request logging (for use with morgan)
export const logStream = {
  write: (message: string) => {
    logger.http(message.trim());
  }
};

// Helper functions for structured logging
export const loggerHelpers = {
  // Log with context
  logWithContext: (level: string, message: string, context: Record<string, any>) => {
    logger.log(level, message, context);
  },

  // Log API request
  logApiRequest: (method: string, url: string, statusCode: number, responseTime: number, userId?: string) => {
    logger.info('API Request', {
      method,
      url,
      statusCode,
      responseTime: `${responseTime}ms`,
      userId: userId || 'anonymous',
      type: 'api_request'
    });
  },

  // Log API error
  logApiError: (method: string, url: string, statusCode: number, error: Error, userId?: string) => {
    logger.error('API Error', {
      method,
      url,
      statusCode,
      error: error.message,
      stack: error.stack,
      userId: userId || 'anonymous',
      type: 'api_error'
    });
  },

  // Log database operation
  logDbOperation: (operation: string, collection: string, duration: number, success: boolean, error?: Error) => {
    const level = success ? 'info' : 'error';
    logger.log(level, `Database Operation: ${operation}`, {
      collection,
      duration: `${duration}ms`,
      success,
      error: error?.message,
      type: 'db_operation'
    });
  },

  // Log AI model usage
  logAiModelUsage: (model: string, operation: string, tokens: number, duration: number, success: boolean, error?: Error) => {
    const level = success ? 'info' : 'error';
    logger.log(level, `AI Model Usage: ${model}`, {
      operation,
      tokens,
      duration: `${duration}ms`,
      success,
      error: error?.message,
      type: 'ai_model_usage'
    });
  },

  // Log security event
  logSecurityEvent: (event: string, userId: string, details: Record<string, any>) => {
    logger.warn('Security Event', {
      event,
      userId,
      ...details,
      type: 'security_event'
    });
  },

  // Log performance metric
  logPerformance: (metric: string, value: number, unit: string, context?: Record<string, any>) => {
    logger.info('Performance Metric', {
      metric,
      value,
      unit,
      ...context,
      type: 'performance_metric'
    });
  },

  // Log user action
  logUserAction: (action: string, userId: string, details: Record<string, any>) => {
    logger.info('User Action', {
      action,
      userId,
      ...details,
      type: 'user_action'
    });
  },

  // Log system event
  logSystemEvent: (event: string, details: Record<string, any>) => {
    logger.info('System Event', {
      event,
      ...details,
      type: 'system_event'
    });
  },

  // Log workflow execution
  logWorkflow: (workflowId: string, step: string, status: 'started' | 'completed' | 'failed', details?: Record<string, any>) => {
    const level = status === 'failed' ? 'error' : 'info';
    logger.log(level, `Workflow ${status}: ${step}`, {
      workflowId,
      step,
      status,
      ...details,
      type: 'workflow_execution'
    });
  },

  // Log RAG operation
  logRagOperation: (operation: string, collection: string, documents: number, duration: number, success: boolean, error?: Error) => {
    const level = success ? 'info' : 'error';
    logger.log(level, `RAG Operation: ${operation}`, {
      collection,
      documents,
      duration: `${duration}ms`,
      success,
      error: error?.message,
      type: 'rag_operation'
    });
  },

  // Log MLOps operation
  logMlopsOperation: (operation: string, model: string, duration: number, success: boolean, error?: Error) => {
    const level = success ? 'info' : 'error';
    logger.log(level, `MLOps Operation: ${operation}`, {
      model,
      duration: `${duration}ms`,
      success,
      error: error?.message,
      type: 'mlops_operation'
    });
  },

  // Log media processing
  logMediaProcessing: (contentType: string, operation: string, fileSize: number, duration: number, success: boolean, error?: Error) => {
    const level = success ? 'info' : 'error';
    logger.log(level, `Media Processing: ${operation}`, {
      contentType,
      fileSize: `${fileSize} bytes`,
      duration: `${duration}ms`,
      success,
      error: error?.message,
      type: 'media_processing'
    });
  }
};

// Export specific log level functions for convenience
export const {
  error: logError,
  warn: logWarn,
  info: logInfo,
  debug: logDebug,
  verbose: logVerbose
} = logger;

// Performance monitoring helper
export class PerformanceMonitor {
  private startTime: number;
  private name: string;
  private context: Record<string, any>;

  constructor(name: string, context: Record<string, any> = {}) {
    this.name = name;
    this.context = context;
    this.startTime = Date.now();
    logger.debug(`Performance monitor started: ${name}`, context);
  }

  end(additionalContext?: Record<string, any>) {
    const duration = Date.now() - this.startTime;
    const finalContext = { ...this.context, ...additionalContext };
    
    loggerHelpers.logPerformance(this.name, duration, 'ms', finalContext);
    
    return duration;
  }

  lap(lapName: string) {
    const duration = Date.now() - this.startTime;
    logger.debug(`Performance lap: ${this.name} - ${lapName}`, {
      ...this.context,
      lapDuration: `${duration}ms`
    });
    return duration;
  }
}

// Request correlation ID middleware helper
export const createCorrelationId = () => {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
};

// Enhanced logger with correlation ID
export const createContextLogger = (correlationId: string, userId?: string) => {
  return {
    error: (message: string, meta?: any) => logger.error(message, { correlationId, userId, ...meta }),
    warn: (message: string, meta?: any) => logger.warn(message, { correlationId, userId, ...meta }),
    info: (message: string, meta?: any) => logger.info(message, { correlationId, userId, ...meta }),
    debug: (message: string, meta?: any) => logger.debug(message, { correlationId, userId, ...meta }),
    verbose: (message: string, meta?: any) => logger.verbose(message, { correlationId, userId, ...meta })
  };
};

// Cleanup function for graceful shutdown
export const cleanupLogger = async () => {
  return new Promise<void>((resolve) => {
    logger.on('finish', () => {
      resolve();
    });
    logger.end();
  });
};

// Error handling for logger
logger.on('error', (err) => {
  console.error('Logger error:', err);
});

// Log application startup
logger.info('Logger initialized', {
  logLevel: getLogLevel(),
  nodeEnv: process.env.NODE_ENV || 'development',
  logDir,
  type: 'system_startup'
});

// Export logger as default
export default logger;
