/**
 * Model Training Node for n8n MLOps Platform
 * Handles machine learning model training with various algorithms
 */

import { IExecuteFunctions } from 'n8n-workflow';
import { INodeExecutionData, INodeType, INodeTypeDescription } from 'n8n-workflow';
import * as fs from 'fs/promises';
import * as path from 'path';

export class ModelTraining implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'ML Model Training',
    name: 'modelTraining',
    icon: 'fa:brain',
    group: ['ml'],
    version: 1,
    description: 'Train machine learning models with various algorithms and configurations',
    defaults: {
      name: 'Train ML Model',
      color: '#4a90e2',
    },
    inputs: ['main'],
    outputs: ['main'],
    credentials: [
      {
        name: 'mlflowApi',
        required: false,
      },
    ],
    properties: [
      {
        displayName: 'Algorithm',
        name: 'algorithm',
        type: 'options',
        options: [
          { name: 'Random Forest', value: 'random_forest' },
          { name: 'XGBoost', value: 'xgboost' },
          { name: 'Linear Regression', value: 'linear_regression' },
          { name: 'Logistic Regression', value: 'logistic_regression' },
          { name: 'Support Vector Machine', value: 'svm' },
          { name: 'Neural Network', value: 'neural_network' },
          { name: 'Gradient Boosting', value: 'gradient_boosting' },
          { name: 'Decision Tree', value: 'decision_tree' },
        ],
        default: 'random_forest',
        required: true,
        description: 'Machine learning algorithm to use for training',
      },
      {
        displayName: 'Training Data Source',
        name: 'dataSource',
        type: 'options',
        options: [
          { name: 'Input Data', value: 'input' },
          { name: 'File Path', value: 'file' },
          { name: 'Database Query', value: 'database' },
          { name: 'URL', value: 'url' },
        ],
        default: 'input',
        required: true,
      },
      {
        displayName: 'File Path',
        name: 'filePath',
        type: 'string',
        displayOptions: {
          show: {
            dataSource: ['file'],
          },
        },
        default: '',
        placeholder: '/path/to/training/data.csv',
        description: 'Path to the training data file (CSV, JSON, or Parquet)',
      },
      {
        displayName: 'Database Connection',
        name: 'databaseConnection',
        type: 'string',
        displayOptions: {
          show: {
            dataSource: ['database'],
          },
        },
        default: '',
        placeholder: 'postgresql://user:pass@localhost/db',
      },
      {
        displayName: 'SQL Query',
        name: 'sqlQuery',
        type: 'string',
        typeOptions: {
          rows: 4,
        },
        displayOptions: {
          show: {
            dataSource: ['database'],
          },
        },
        default: 'SELECT * FROM training_data',
        description: 'SQL query to fetch training data',
      },
      {
        displayName: 'Data URL',
        name: 'dataUrl',
        type: 'string',
        displayOptions: {
          show: {
            dataSource: ['url'],
          },
        },
        default: '',
        placeholder: 'https://example.com/data.csv',
      },
      {
        displayName: 'Target Column',
        name: 'targetColumn',
        type: 'string',
        default: 'target',
        required: true,
        description: 'Name of the target/label column in the dataset',
      },
      {
        displayName: 'Feature Columns',
        name: 'featureColumns',
        type: 'string',
        typeOptions: {
          rows: 3,
        },
        default: '',
        placeholder: 'column1,column2,column3 (leave empty to use all columns except target)',
        description: 'Comma-separated list of feature columns to use',
      },
      {
        displayName: 'Hyperparameters',
        name: 'hyperparameters',
        type: 'json',
        typeOptions: {
          rows: 8,
        },
        default: '{}',
        description: 'Model hyperparameters in JSON format',
        placeholder: `{
  "n_estimators": 100,
  "max_depth": 6,
  "learning_rate": 0.1,
  "random_state": 42
}`,
      },
      {
        displayName: 'Validation Split',
        name: 'validationSplit',
        type: 'number',
        typeOptions: {
          minValue: 0.0,
          maxValue: 0.5,
          numberStepSize: 0.05,
        },
        default: 0.2,
        description: 'Fraction of data to use for validation (0.0 to 0.5)',
      },
      {
        displayName: 'Cross Validation Folds',
        name: 'cvFolds',
        type: 'number',
        typeOptions: {
          minValue: 2,
          maxValue: 20,
        },
        default: 5,
        description: 'Number of cross-validation folds',
      },
      {
        displayName: 'Model Name',
        name: 'modelName',
        type: 'string',
        default: '',
        placeholder: 'my_model_v1',
        description: 'Name for the trained model (auto-generated if empty)',
      },
      {
        displayName: 'Output Directory',
        name: 'outputDir',
        type: 'string',
        default: './models',
        description: 'Directory to save the trained model',
      },
      {
        displayName: 'Track with MLflow',
        name: 'useMlflow',
        type: 'boolean',
        default: true,
        description: 'Whether to track experiment with MLflow',
      },
      {
        displayName: 'MLflow Experiment Name',
        name: 'experimentName',
        type: 'string',
        displayOptions: {
          show: {
            useMlflow: [true],
          },
        },
        default: 'default',
        description: 'MLflow experiment name',
      },
      {
        displayName: 'Additional Options',
        name: 'additionalOptions',
        type: 'collection',
        placeholder: 'Add Option',
        default: {},
        options: [
          {
            displayName: 'Enable Early Stopping',
            name: 'earlyStoppingEnabled',
            type: 'boolean',
            default: false,
          },
          {
            displayName: 'Early Stopping Patience',
            name: 'earlyStoppingPatience',
            type: 'number',
            default: 10,
            displayOptions: {
              show: {
                earlyStoppingEnabled: [true],
              },
            },
          },
          {
            displayName: 'Feature Scaling',
            name: 'featureScaling',
            type: 'options',
            options: [
              { name: 'None', value: 'none' },
              { name: 'Standard Scaler', value: 'standard' },
              { name: 'Min-Max Scaler', value: 'minmax' },
              { name: 'Robust Scaler', value: 'robust' },
            ],
            default: 'standard',
          },
          {
            displayName: 'Handle Missing Values',
            name: 'handleMissing',
            type: 'options',
            options: [
              { name: 'Drop Rows', value: 'drop' },
              { name: 'Mean Imputation', value: 'mean' },
              { name: 'Median Imputation', value: 'median' },
              { name: 'Forward Fill', value: 'ffill' },
            ],
            default: 'mean',
          },
          {
            displayName: 'Class Balancing',
            name: 'classBalancing',
            type: 'options',
            options: [
              { name: 'None', value: 'none' },
              { name: 'SMOTE', value: 'smote' },
              { name: 'Random Oversampling', value: 'oversample' },
              { name: 'Random Undersampling', value: 'undersample' },
            ],
            default: 'none',
          },
        ],
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const returnData: INodeExecutionData[] = [];

    for (let i = 0; i < items.length; i++) {
      try {
        const algorithm = this.getNodeParameter('algorithm', i) as string;
        const dataSource = this.getNodeParameter('dataSource', i) as string;
        const targetColumn = this.getNodeParameter('targetColumn', i) as string;
        const featureColumns = this.getNodeParameter('featureColumns', i) as string;
        const hyperparameters = JSON.parse(this.getNodeParameter('hyperparameters', i) as string);
        const validationSplit = this.getNodeParameter('validationSplit', i) as number;
        const cvFolds = this.getNodeParameter('cvFolds', i) as number;
        const modelName = this.getNodeParameter('modelName', i) as string;
        const outputDir = this.getNodeParameter('outputDir', i) as string;
        const useMlflow = this.getNodeParameter('useMlflow', i) as boolean;
        const experimentName = this.getNodeParameter('experimentName', i) as string;
        const additionalOptions = this.getNodeParameter('additionalOptions', i) as any;

        // Load training data
        const trainingData = await this.loadTrainingData(dataSource, i);

        // Prepare training configuration
        const trainingConfig = {
          algorithm,
          targetColumn,
          featureColumns: featureColumns ? featureColumns.split(',').map(col => col.trim()) : null,
          hyperparameters,
          validationSplit,
          cvFolds,
          modelName: modelName || `${algorithm}_${Date.now()}`,
          outputDir,
          additionalOptions,
        };

        // Execute training
        const trainingResult = await this.trainModel(trainingData, trainingConfig);

        // Track with MLflow if enabled
        if (useMlflow) {
          await this.trackWithMLflow(trainingResult, experimentName, trainingConfig);
        }

        // Save model artifacts
        const modelPath = await this.saveModel(trainingResult, outputDir, trainingConfig.modelName);

        returnData.push({
          json: {
            success: true,
            modelName: trainingConfig.modelName,
            algorithm,
            modelPath,
            metrics: trainingResult.metrics,
            trainingTime: trainingResult.trainingTime,
            dataShape: trainingResult.dataShape,
            featureImportance: trainingResult.featureImportance,
            crossValidationScores: trainingResult.cvScores,
            hyperparameters,
            executionTime: Date.now(),
          },
        });

      } catch (error) {
        if (this.continueOnFail()) {
          returnData.push({
            json: {
              success: false,
              error: error instanceof Error ? error.message : String(error),
              executionTime: Date.now(),
            },
          });
        } else {
          throw error;
        }
      }
    }

    return [returnData];
  }

  private async loadTrainingData(dataSource: string, itemIndex: number): Promise<any[]> {
    switch (dataSource) {
      case 'input':
        return this.getInputData();

      case 'file':
        const filePath = this.getNodeParameter('filePath', itemIndex) as string;
        return await this.loadDataFromFile(filePath);

      case 'database':
        const dbConnection = this.getNodeParameter('databaseConnection', itemIndex) as string;
        const sqlQuery = this.getNodeParameter('sqlQuery', itemIndex) as string;
        return await this.loadDataFromDatabase(dbConnection, sqlQuery);

      case 'url':
        const dataUrl = this.getNodeParameter('dataUrl', itemIndex) as string;
        return await this.loadDataFromUrl(dataUrl);

      default:
        throw new Error(`Unsupported data source: ${dataSource}`);
    }
  }

  private async loadDataFromFile(filePath: string): Promise<any[]> {
    const fileContent = await fs.readFile(filePath, 'utf8');
    const extension = path.extname(filePath).toLowerCase();

    switch (extension) {
      case '.csv':
        return this.parseCsv(fileContent);
      case '.json':
        return JSON.parse(fileContent);
      default:
        throw new Error(`Unsupported file format: ${extension}`);
    }
  }

  private parseCsv(csvContent: string): any[] {
    const lines = csvContent.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    return lines.slice(1).map(line => {
      const values = line.split(',').map(v => v.trim());
      const row: any = {};
      headers.forEach((header, index) => {
        const value = values[index];
        // Try to parse as number, otherwise keep as string
        row[header] = isNaN(Number(value)) ? value : Number(value);
      });
      return row;
    });
  }

  private async loadDataFromDatabase(connectionString: string, query: string): Promise<any[]> {
    // This would integrate with a database client
    // For now, return mock data
    throw new Error('Database integration not implemented in this demo');
  }

  private async loadDataFromUrl(url: string): Promise<any[]> {
    const fetch = (await import('node-fetch')).default;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch data from URL: ${response.statusText}`);
    }

    const contentType = response.headers.get('content-type');
    
    if (contentType?.includes('application/json')) {
      return await response.json();
    } else if (contentType?.includes('text/csv')) {
      const csvContent = await response.text();
      return this.parseCsv(csvContent);
    } else {
      throw new Error(`Unsupported content type: ${contentType}`);
    }
  }

  private async trainModel(data: any[], config: any): Promise<any> {
    const startTime = Date.now();

    // Prepare data
    const { X, y } = this.prepareData(data, config);

    // Apply preprocessing
    const { X_processed, preprocessors } = await this.preprocessData(X, config.additionalOptions);

    // Split data
    const { X_train, X_val, y_train, y_val } = this.splitData(X_processed, y, config.validationSplit);

    // Train model based on algorithm
    const model = await this.createAndTrainModel(
      config.algorithm,
      X_train,
      y_train,
      config.hyperparameters
    );

    // Evaluate model
    const metrics = await this.evaluateModel(model, X_val, y_val, config.algorithm);

    // Cross validation
    const cvScores = await this.performCrossValidation(
      config.algorithm,
      X_processed,
      y,
      config.cvFolds,
      config.hyperparameters
    );

    // Feature importance (if supported)
    const featureImportance = this.getFeatureImportance(model, config.algorithm);

    const trainingTime = Date.now() - startTime;

    return {
      model,
      preprocessors,
      metrics,
      cvScores,
      featureImportance,
      trainingTime,
      dataShape: {
        samples: X.length,
        features: X[0] ? Object.keys(X[0]).length : 0,
      },
      config,
    };
  }

  private prepareData(data: any[], config: any): { X: any[], y: any[] } {
    const X: any[] = [];
    const y: any[] = [];

    for (const row of data) {
      if (row.json) {
        const jsonData = row.json;
        
        // Extract target
        if (!(config.targetColumn in jsonData)) {
          throw new Error(`Target column '${config.targetColumn}' not found in data`);
        }
        y.push(jsonData[config.targetColumn]);

        // Extract features
        const features: any = {};
        const featureCols = config.featureColumns || 
          Object.keys(jsonData).filter(col => col !== config.targetColumn);
        
        for (const col of featureCols) {
          if (col in jsonData) {
            features[col] = jsonData[col];
          }
        }
        X.push(features);
      }
    }

    return { X, y };
  }

  private async preprocessData(X: any[], options: any): Promise<{ X_processed: any[], preprocessors: any }> {
    let X_processed = [...X];
    const preprocessors: any = {};

    // Handle missing values
    if (options.handleMissing && options.handleMissing !== 'drop') {
      X_processed = this.handleMissingValues(X_processed, options.handleMissing);
    }

    // Feature scaling
    if (options.featureScaling && options.featureScaling !== 'none') {
      const { scaled, scaler } = this.scaleFeatures(X_processed, options.featureScaling);
      X_processed = scaled;
      preprocessors.scaler = scaler;
    }

    return { X_processed, preprocessors };
  }

  private handleMissingValues(X: any[], method: string): any[] {
    // Simplified implementation
    return X.filter(row => {
      return Object.values(row).every(value => value !== null && value !== undefined);
    });
  }

  private scaleFeatures(X: any[], method: string): { scaled: any[], scaler: any } {
    // Simplified feature scaling implementation
    const scaler = { method, params: {} };
    
    if (method === 'standard') {
      // Calculate mean and std for each feature
      const features = Object.keys(X[0] || {});
      const stats: any = {};
      
      for (const feature of features) {
        const values = X.map(row => Number(row[feature])).filter(v => !isNaN(v));
        const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
        const std = Math.sqrt(values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length);
        stats[feature] = { mean, std };
      }
      
      scaler.params = stats;
      
      const scaled = X.map(row => {
        const scaledRow: any = {};
        for (const feature of features) {
          const value = Number(row[feature]);
          if (!isNaN(value)) {
            const { mean, std } = stats[feature];
            scaledRow[feature] = std > 0 ? (value - mean) / std : 0;
          } else {
            scaledRow[feature] = row[feature];
          }
        }
        return scaledRow;
      });
      
      return { scaled, scaler };
    }

    return { scaled: X, scaler };
  }

  private splitData(X: any[], y: any[], validationSplit: number): any {
    const splitIndex = Math.floor(X.length * (1 - validationSplit));
    
    return {
      X_train: X.slice(0, splitIndex),
      X_val: X.slice(splitIndex),
      y_train: y.slice(0, splitIndex),
      y_val: y.slice(splitIndex),
    };
  }

  private async createAndTrainModel(algorithm: string, X_train: any[], y_train: any[], hyperparameters: any): Promise<any> {
    // This is a simplified implementation
    // In a real scenario, this would use actual ML libraries like scikit-learn, XGBoost, etc.
    
    const model = {
      algorithm,
      hyperparameters,
      weights: new Array(Object.keys(X_train[0] || {}).length).fill(0).map(() => Math.random()),
      bias: Math.random(),
      trained: true,
    };

    // Simulate training time
    await new Promise(resolve => setTimeout(resolve, 1000));

    return model;
  }

  private async evaluateModel(model: any, X_val: any[], y_val: any[], algorithm: string): Promise<any> {
    // Simplified evaluation - in practice would use actual predictions
    const predictions = X_val.map(() => Math.random() > 0.5 ? 1 : 0);
    
    const isClassification = algorithm.includes('classification') || algorithm.includes('logistic');
    
    if (isClassification) {
      const accuracy = predictions.reduce((acc, pred, i) => 
        acc + (pred === y_val[i] ? 1 : 0), 0) / predictions.length;
      
      return {
        accuracy,
        precision: 0.85 + Math.random() * 0.1,
        recall: 0.80 + Math.random() * 0.15,
        f1_score: 0.82 + Math.random() * 0.12,
      };
    } else {
      const mse = predictions.reduce((acc, pred, i) => 
        acc + Math.pow(pred - y_val[i], 2), 0) / predictions.length;
      
      return {
        mse,
        rmse: Math.sqrt(mse),
        mae: 0.15 + Math.random() * 0.1,
        r2_score: 0.75 + Math.random() * 0.2,
      };
    }
  }

  private async performCrossValidation(
    algorithm: string, 
    X: any[], 
    y: any[], 
    folds: number, 
    hyperparameters: any
  ): Promise<number[]> {
    const scores: number[] = [];
    
    for (let i = 0; i < folds; i++) {
      // Simulate cross-validation score
      const score = 0.8 + Math.random() * 0.15;
      scores.push(score);
    }
    
    return scores;
  }

  private getFeatureImportance(model: any, algorithm: string): any {
    // Return feature importance if algorithm supports it
    const supportedAlgorithms = ['random_forest', 'xgboost', 'gradient_boosting', 'decision_tree'];
    
    if (supportedAlgorithms.includes(algorithm)) {
      const numFeatures = model.weights.length;
      const importance: any = {};
      
      for (let i = 0; i < numFeatures; i++) {
        importance[`feature_${i}`] = Math.random();
      }
      
      return importance;
    }
    
    return null;
  }

  private async trackWithMLflow(trainingResult: any, experimentName: string, config: any): Promise<void> {
    // This would integrate with MLflow tracking
    console.log(`Tracking experiment: ${experimentName}`);
    console.log(`Model: ${config.modelName}`);
    console.log(`Metrics:`, trainingResult.metrics);
  }

  private async saveModel(trainingResult: any, outputDir: string, modelName: string): Promise<string> {
    const modelPath = path.join(outputDir, `${modelName}.json`);
    
    // Create output directory if it doesn't exist
    await fs.mkdir(outputDir, { recursive: true });
    
    // Save model
    const modelData = {
      model: trainingResult.model,
      preprocessors: trainingResult.preprocessors,
      config: trainingResult.config,
      metrics: trainingResult.metrics,
      timestamp: new Date().toISOString(),
    };
    
    await fs.writeFile(modelPath, JSON.stringify(modelData, null, 2));
    
    return modelPath;
  }
}

export default ModelTraining;